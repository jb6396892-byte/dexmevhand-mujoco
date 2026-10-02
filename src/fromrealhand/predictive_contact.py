"""Finite-candidate contact lookahead in separate MuJoCo data, not a safety proof."""
import time
import numpy as np
from .contact_guard import normal_joint_delta


class PredictiveContactFilter:
    def __init__(self, experiment, config, reference=None):
        from mujoco_py import MjSim
        self.exp, self.config = experiment, dict(config)
        self.reference = reference
        self.model = experiment.model
        # The model is shared read-only. All hypothetical state writes target this data only.
        self.sim = MjSim(self.model)
        self.hands = {self.model.geom_name2id(n) for n in experiment.hand_geoms}
        self.mugs = {self.model.geom_name2id(n) for n in experiment.mug_geoms}
        self.joints = {f:[int(self.model.jnt_qposadr[j]) for j in range(self.model.njnt)
                            if (self.model.joint_id2name(j) or '').startswith(f.upper()+'J')]
                       for f in ('th','ff','mf','rf','lf')}
        self.conversion = -self.model.actuator_biasprm[:,1]/(
            self.model.actuator_gainprm[:,0]*experiment.env.act_rng)
        self.offset = np.zeros(30)
        self.previous_prediction = None
        self.logs = []

    def initialize_branch(self):
        live, branch = self.exp.env.sim.data, self.sim.data
        self.sim.set_state(self.exp.env.sim.get_state())
        for name in ('ctrl','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','userdata'):
            value = getattr(live,name)
            if value is not None:
                getattr(branch,name)[:] = value
        self.sim.forward()
        branch.qacc_warmstart[:] = live.qacc_warmstart

    def contacts(self, directions=False, forces_needed=True):
        from mujoco_py import functions
        m, d = self.model, self.sim.data
        peak = 0.; forces = np.zeros(5); deepest = {}
        fingers = ('th','ff','mf','rf','lf')
        for i,c in enumerate(d.contact[:d.ncon]):
            a,b = int(c.geom1), int(c.geom2)
            if a in self.hands or b in self.hands:
                peak = max(peak, -float(c.dist))
            if a in self.mugs and b in self.hands:
                hand,sign = b,1.
            elif b in self.mugs and a in self.hands:
                hand,sign = a,-1.
            else:
                continue
            finger = (m.geom_id2name(hand) or '')[2:4]
            if finger not in fingers:
                continue
            if forces_needed:
                f = np.zeros(6); functions.mj_contactForce(m,d,i,f)
                forces[fingers.index(finger)] += max(0.,float(f[0]))
            depth = max(0., -float(c.dist))
            if directions and depth > max(self.config['target_depth_m'],deepest.get(finger,(0.,None))[0]):
                body = int(m.geom_bodyid[hand]); name = m.body_id2name(body)
                linear = d.get_body_jacp(name).reshape(3,m.nv)
                angular = d.get_body_jacr(name).reshape(3,m.nv)
                jac = (linear+np.cross(angular.T,np.asarray(c.pos)-d.body_xpos[body]).T)[:,:30].copy()
                mask = np.zeros(30); mask[self.joints[finger]] = 1.; jac *= mask
                delta = normal_joint_delta(jac, sign*np.asarray(c.frame).reshape(3,3)[0], depth,
                                          self.config['target_depth_m'],1.,self.config['joint_cap_rad'])
                deepest[finger] = (depth,delta)
        return peak,forces,deepest

    def predict(self, action, control_step=0, horizon=None, directions_needed=True):
        self.initialize_branch()
        e, d = self.exp.env, self.sim.data
        nsub = int(e.control_timestep/e.model_timestep)
        count = self.config['horizon_steps'] if horizon is None else horizon
        peak = 0.; directions = {}; first = None
        for step in range(count):
            future = np.asarray(action)
            if self.config.get('reference_lookahead',False):
                if self.reference is None:
                    raise ValueError('Future reference is required')
                idx = min(control_step+step,len(self.reference)-1)
                future = future+self.reference[idx]-self.reference[control_step]
            # Identical normalization to YCBRelocate._pre_action, once in the branch.
            d.ctrl[:] = e.act_mid+np.clip(future,-1.,1.)*e.act_rng
            for substep in range(nsub):
                self.sim.step()
                depth,forces,local = self.contacts(directions_needed,step==count-1 and substep==nsub-1)
                peak = max(peak,depth)
                for finger,value in local.items():
                    if value[0] > directions.get(finger,(0.,None))[0]:
                        directions[finger] = value
            if first is None:
                first = np.r_[d.qpos.copy(),d.qvel.copy()]
        return dict(depth=peak, forces=forces, directions=directions, first=first,
                    cup=d.qpos[30:33].copy())

    def apply(self, action, control_step=0):
        begin = time.monotonic()
        live = self.exp.env.sim.data
        original = np.r_[live.qpos.copy(),live.qvel.copy()]
        prediction_error = None if self.previous_prediction is None else float(np.max(np.abs(original-self.previous_prediction)))
        nominal = self.predict(action,control_step)
        cap = np.full(30,self.config['joint_cap_rad'])
        cap[:3] = self.config.get('root_cap_m',self.config['joint_cap_rad'])
        zero = np.zeros(30)
        candidates = [(zero,nominal)]
        active = nominal['depth'] > self.config['target_depth_m']
        if active or np.max(np.abs(self.offset)) > 1e-5:
            directions = sum((v[1] for v in nominal['directions'].values()),zero.copy())
            offsets = [self.offset*self.config['release']]
            if active:
                offsets += [np.clip(self.offset+gain*directions,-cap,cap) for gain in self.config['gains']]
                for joint,amplitude in self.config.get('coordinate_probes',[]):
                    index = int(self.model.jnt_qposadr[self.model.joint_name2id(joint)])
                    for sign in (-1.,1.):
                        probe = self.offset.copy()
                        probe[index] += sign*amplitude
                        offsets.append(np.clip(probe,-cap,cap))
            for offset in offsets:
                if np.max(np.abs(offset)) < 1e-9:
                    continue
                result = self.predict(action+self.conversion*offset,control_step,directions_needed=False)
                candidates.append((offset,result))
        def cost(item):
            offset,p = item
            excess = max(0.,p['depth']-self.config['target_depth_m'])/.0002
            lost = np.sum((nominal['forces']>.01)&(p['forces']<=.01))
            cup = np.linalg.norm(p['cup']-nominal['cup'])/.001
            return 100*excess**2+2.*lost+.1*cup**2+.01*np.sum((offset/cap)**2)
        offset,chosen = min(candidates,key=cost)
        self.offset = offset.copy()
        self.previous_prediction = chosen['first']
        if not np.array_equal(original,np.r_[live.qpos,live.qvel]):
            raise RuntimeError('Predictive filter changed live physical state')
        self.logs.append(dict(prediction_error=prediction_error, nominal_depth=nominal['depth'],
            chosen_depth=chosen['depth'], offset_max_rad=float(np.max(np.abs(offset[3:]))),
            root_offset_max_m=float(np.max(np.abs(offset[:3]))),
            candidates=len(candidates), elapsed_s=time.monotonic()-begin))
        return np.asarray(action)+self.conversion*offset


class PredictiveActions:
    def __init__(self, base, experiment, config):
        self.base = base
        self.filter = PredictiveContactFilter(experiment,config,base.reference)
    def __len__(self): return len(self.base)
    def __getitem__(self, step): return self.filter.apply(self.base[step],step)
