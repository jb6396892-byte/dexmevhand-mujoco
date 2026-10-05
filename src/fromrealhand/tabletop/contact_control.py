"""Contact-anchored tracking using visual object poses and hand proprioception."""
import numpy as np
from .control import VisualReference, rigid

FINGERS = ('th', 'ff', 'mf', 'rf', 'lf')


def opposition_from_vectors(vectors, threshold=.01, dot_threshold=-.3, min_opposed=2):
    th = np.asarray(vectors['th'], dtype=float)
    dots = {}
    for f in FINGERS[1:]:
        v = np.asarray(vectors[f], dtype=float)
        n, t = np.linalg.norm(v), np.linalg.norm(th)
        if not np.isfinite(v).all() or not np.isfinite(th).all():
            raise ValueError('Nonfinite contact vector')
        if n > threshold and t > threshold: dots[f] = float(np.dot(th,v)/(t*n))
    return dict(opposition=sum(dot < dot_threshold for dot in dots.values()) >= min_opposed, normal_dots=dots)


def opposing_contacts(sim, threshold=.01, dot_threshold=-.3, min_opposed=2):
    """Simulation contact normals are diagnostics, not real-video force labels."""
    import mujoco_py
    m, d = sim.model, sim.data
    vectors = {f: np.zeros(3) for f in FINGERS}
    pairs = []
    for i in range(d.ncon):
        c = d.contact[i]; gids = (int(c.geom1), int(c.geom2))
        names = [m.geom_id2name(g) or '' for g in gids]
        if not any(m.body_id2name(int(m.geom_bodyid[g])) == 'mug_0' for g in gids): continue
        for j, name in enumerate(names):
            if not name.startswith('C_') or name[2:4] not in vectors: continue
            wrench = np.zeros(6); mujoco_py.functions.mj_contactForce(m,d,i,wrench)
            force = max(0.,float(wrench[0]))
            vectors[name[2:4]] += (1. if j else -1.)*np.asarray(c.frame[:3])*force
            pairs.append(dict(hand=name,object=names[1-j],distance_m=float(c.dist),normal_force_n=force))
    result = opposition_from_vectors(vectors,threshold,dot_threshold,min_opposed)
    result['pairs'] = pairs
    return result


def object_pose(qpos):
    import transforms3d
    pose = np.eye(4)
    pose[:3, :3] = transforms3d.quaternions.quat2mat(qpos[33:37])
    pose[:3, 3] = qpos[30:33]
    return pose


def reference_contacts(reference_sim, states):
    """Use a separate reference simulation for FK, never write live robot state."""
    import mujoco_py
    scratch = mujoco_py.MjSim(reference_sim.model)
    tips, poses = [], []
    for state in states:
        scratch.data.qpos[:] = state['qpos']
        scratch.forward()
        pose = object_pose(state['qpos'])
        points = np.array([scratch.data.get_site_xpos('S_'+f+'tip').copy() for f in FINGERS])
        tips.append((points-pose[:3, 3]) @ pose[:3, :3])
        poses.append(pose)
    return np.asarray(tips), np.asarray(poses)


class ContactTracker:
    def __init__(self, adapter, local_tips, kp=160., damping=.4, root_gain=1., finger_gain=0.):
        self.adapter, self.local_tips = adapter, np.asarray(local_tips)
        self.kp, self.damping = float(kp), float(damping)
        self.root_gain, self.finger_gain = float(root_gain), float(finger_gain)
        parameters = [self.kp,self.damping,self.root_gain,self.finger_gain]
        if not np.isfinite(parameters).all() or min(parameters) < 0:
            raise ValueError('Invalid tracking gains')
        if self.local_tips.shape != (len(adapter.actions),5,3) or not np.isfinite(self.local_tips).all():
            raise ValueError('Invalid reference fingertip targets')
        self.last = {}
        self.carry = None

    def start_carry(self,sim,visual_pose,goal,anchor,lift_stop,stop):
        if not hasattr(self,'last_action'): raise ValueError('Carry needs an executed grasp action')
        pose=rigid(visual_pose); q=sim.data.qpos[:30].copy()
        displacement=np.asarray(goal)-pose[:3,3]
        if displacement.shape!=(3,) or not np.isfinite(displacement).all(): raise ValueError('Invalid carry goal')
        base=self.adapter.base[:3,:3]
        for offset in (np.array([0.,0.,.08]),displacement):
            target=q[:3]+base.T @ offset
            if np.any(target<sim.model.jnt_range[:3,0]) or np.any(target>sim.model.jnt_range[:3,1]):
                raise ValueError('Measured carry start outside reachable goal workspace')
        self.carry=dict(q=q,action=self.last_action.copy(),delta=displacement,anchor=anchor,
            lift_stop=lift_stop,stop=stop,goal=np.asarray(goal).copy(),correction=np.zeros(3),visual_frame=None)

    def reference(self,index,sim):
        if self.carry is None or index<self.carry['anchor']:
            return self.adapter.desired_qpos(index),self.adapter.action(index,sim.data.qpos[:30].copy())
        c=self.carry; lift=np.array([0.,0.,.08])
        if index<c['lift_stop']:
            t=np.clip((index-c['anchor'])/float(c['lift_stop']-c['anchor']),0.,1.)
            offset=lift*t**3*(10-15*t+6*t*t)
        else:
            t=np.clip((index-c['lift_stop'])/float(c['stop']-1-c['lift_stop']),0.,1.)
            offset=lift+(c['delta']-lift)*t**3*(10-15*t+6*t*t)
        offset=offset+c['correction']
        desired=c['q'].copy(); desired[:3]+=self.adapter.base[:3,:3].T @ offset
        factor=-sim.model.actuator_biasprm[:,1]/(sim.model.actuator_gainprm[:,0]*self.adapter_model_span(sim.model))
        action=c['action'].copy(); action[:6]+=factor[:6]*(desired[:6]-c['q'][:6])
        return desired,action

    def action(self, index, sim, visual_pose, visual_frame=None):
        m, d = sim.model, sim.data
        pose = rigid(visual_pose)
        if (self.carry is not None and index>=self.carry['stop']-200 and visual_frame is not None
                and visual_frame!=self.carry['visual_frame']):
            c=self.carry; error=c['goal']-pose[:3,3]
            if np.linalg.norm(error)>.08: raise ValueError('Visual carry correction outside local basin')
            increment=.25*error
            increment*=min(1.,.003/max(np.linalg.norm(increment),1e-12))
            correction=c['correction']+increment
            if np.linalg.norm(correction)>.05: raise ValueError('Visual carry correction budget exceeded')
            c['correction']=correction; c['visual_frame']=visual_frame
        desired,action = self.reference(index,sim)
        # Keep the reference feedforward, but add bounded task-space corrections.
        if np.any(desired[:6]<m.jnt_range[:6,0]) or np.any(desired[:6]>m.jnt_range[:6,1]):
            raise ValueError('Carry/reference target outside joint range')
        error = desired-d.qpos[:30]
        factor = -m.actuator_biasprm[:, 1]/(m.actuator_gainprm[:, 0]*self.adapter_model_span(m))
        action[:6] += factor[:6]*self.root_gain*error[:6]
        action[6:] += factor[6:]*self.finger_gain*error[6:]
        targets = self.local_tips[index] @ pose[:3, :3].T+pose[:3, 3]
        measured = np.array([d.get_site_xpos('S_'+f+'tip').copy() for f in FINGERS])
        torque = np.zeros(30)
        blend = float(np.clip(index*self.adapter.dt/2., 0., 1.))
        for n, finger in enumerate(FINGERS):
            if not self.kp: break
            jac = d.get_site_jacp('S_'+finger+'tip').reshape(3,m.nv)[:,:30]
            e = np.clip(targets[n]-measured[n], -.02, .02)
            velocity = jac @ d.qvel[:30]
            force = self.kp*e-self.damping*np.sqrt(self.kp)*velocity
            # Fingers and wrist rotation, not floating root translation: preserve the lift path.
            torque[6:] += blend*jac[:,6:].T @ force
        action += torque/(m.actuator_gainprm[:,0]*self.adapter_model_span(m))
        self.last = dict(tip_error_m=np.linalg.norm(targets-measured,axis=1).tolist(),
            root_error_m=float(np.linalg.norm(error[:3])), action_abs_max=float(np.max(np.abs(action))))
        if self.carry is not None:
            self.last['visual_carry_correction_m']=self.carry['correction'].tolist()
        if not np.isfinite(action).all() or np.max(np.abs(action)) > 1+1e-8:
            raise ValueError('Contact controller actuator saturation')
        self.last_action=action.copy()
        return action

    @staticmethod
    def adapter_model_span(model):
        return np.diff(model.actuator_ctrlrange,axis=1).ravel()/2
