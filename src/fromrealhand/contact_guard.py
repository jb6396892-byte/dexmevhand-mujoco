"""Small finger-only normal displacement feedback, not a hard safety guarantee."""
import numpy as np


def normal_joint_delta(jacobian, normal, depth, target, gain, cap):
    row=np.asarray(normal) @ np.asarray(jacobian)
    delta=gain*max(0.,depth-target)*row/(float(row @ row)+1e-6)
    return np.clip(delta,-cap,cap)


class ContactGuard:
    def __init__(self, experiment, configuration):
        self.e,self.configuration=experiment,configuration
        self.filtered=np.zeros(30)
        m=experiment.model
        self.conversion=-m.actuator_biasprm[:,1]/(m.actuator_gainprm[:,0]*experiment.env.act_rng)
        self.mug_ids={m.geom_name2id(name) for name in experiment.mug_geoms}
        self.finger_indices={}
        for prefix in ('th','ff','mf','rf','lf'):
            self.finger_indices[prefix]=[int(m.jnt_qposadr[i]) for i in range(m.njnt)
                                        if (m.joint_id2name(i) or '').startswith(prefix.upper()+'J')]

    def correction(self):
        m,d=self.e.model,self.e.env.sim.data
        config=self.configuration; deepest={}
        for c in d.contact[:d.ncon]:
            a,b=int(c.geom1),int(c.geom2)
            if a in self.mug_ids: hand,sign=b,1.
            elif b in self.mug_ids: hand,sign=a,-1.
            else: continue
            name=m.geom_id2name(hand) or ''; finger=name[2:4] if name.startswith('C_') else ''
            if finger not in self.finger_indices: continue
            depth=max(0.,-float(c.dist))
            if depth>deepest.get(finger,(0.,))[0]: deepest[finger]=(depth,c,hand,sign)
        correction=np.zeros(30)
        for finger,(depth,c,hand,sign) in deepest.items():
            if depth<=config['target_depth_m']: continue
            body=int(m.geom_bodyid[hand]); name=m.body_id2name(body)
            linear=d.get_body_jacp(name).reshape(3,m.nv)
            angular=d.get_body_jacr(name).reshape(3,m.nv)
            offset=np.asarray(c.pos)-d.body_xpos[body]
            jac=(linear+np.cross(angular.T,offset).T)[:,:30].copy()
            mask=np.zeros(30);mask[self.finger_indices[finger]]=1.
            jac*=mask
            correction+=normal_joint_delta(jac,sign*np.asarray(c.frame).reshape(3,3)[0],depth,
                       config['target_depth_m'],config['gain'],config['max_joint_delta_rad'])
        alpha=config['filter_alpha']
        self.filtered=(1-alpha)*self.filtered+alpha*correction
        return self.conversion*self.filtered

    def apply(self, action):
        return np.asarray(action)+self.correction()


class GuardedActions:
    def __init__(self, base, experiment, configuration):
        self.base=base;self.guard=ContactGuard(experiment,configuration)
    def __len__(self): return len(self.base)
    def __getitem__(self, step): return self.guard.apply(self.base[step])
