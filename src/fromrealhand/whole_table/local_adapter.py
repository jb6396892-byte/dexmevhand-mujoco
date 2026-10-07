"""Translate a local grasp frame at initialization; preserve learned 139/30 contract."""
import numpy as np
from .hand_scene import HandScene, TRANSLATION_JOINTS


def translate_initial_scene(env, delta, source_poses, scene, translate_object=True):
    import mujoco_py
    delta=np.asarray(delta,dtype=float)
    if delta.shape!=(3,) or not np.isfinite(delta).all() or abs(delta[2])>1e-12:
        raise ValueError('Expected finite tabletop XY frame displacement')
    sim,m,d=env.sim,env.sim.model,env.sim.data
    m.body_pos[m.body_name2id('forearm')]+=delta
    env.reference_base=env.reference_base.copy();env.reference_base[:3,3]+=delta
    for name in (('mug',) if translate_object else ()):
        q=d.get_joint_qpos(name+'_joint_0').copy();q[:3]+=delta
        d.set_joint_qpos(name+'_joint_0',q)
    state=sim.get_state();control=d.ctrl.copy()
    mujoco_py.functions.mj_setConst(m,d)
    sim.set_state(state);d.ctrl[:]=control;sim.forward()
    transformed=source_poses.copy();transformed[:,:3,3]+=delta
    scene['local_frame_translation_m']=delta.tolist()
    scene['frame_set_only_at_initialization']=True
    scene['cup_translated_at_initialization']=translate_object
    return transformed


class MotionBridge(HandScene):
    """Temporary physical navigation servo around a live local-policy environment."""
    def __init__(self,env,config,site='C_palm0',held_action=None):
        from ..tabletop.scene import mesh_local
        self.sim,self.config,self.env=env.sim,config,env
        self.site=site
        m,d=self.sim.model,self.sim.data
        self.qids=np.arange(30);self.vids=np.arange(30);self.aids=np.arange(30)
        self.tcols=np.array([int(m.jnt_qposadr[m.joint_name2id(n)]) for n in TRANSLATION_JOINTS])
        self.posture_cols=np.array([i for i in range(30) if i not in self.tcols])
        self.posture_target=d.qpos[:30].copy()
        self.basis=d.get_site_jacp('S_grasp').reshape(3,m.nv)[:,self.tcols].copy()
        self.inverse_basis=np.linalg.inv(self.basis)
        self.origin=self.position()-self.basis@d.qpos[self.tcols]
        self.backup={n:getattr(m,n).copy() for n in ('actuator_gainprm','actuator_biasprm',
            'actuator_ctrlrange','actuator_forcerange','actuator_forcelimited','jnt_range')}
        self.old_dt=float(m.opt.timestep)
        self.previous_ctrl=d.ctrl.copy()
        self.held_ctrl=None if held_action is None else env.mid+env.rng*np.asarray(held_action)
        m.opt.timestep=config['timestep_s']
        for i in range(30):
            if self.held_ctrl is not None and i>=6: continue
            kp,kd,force=(config['translation_gain'],config['translation_damping'],config['translation_force_n']) if i<3 else ((150.,15.,50.) if i<6 else (30.,1.,8.))
            m.actuator_gainprm[i,:]=0;m.actuator_gainprm[i,0]=kp
            m.actuator_biasprm[i,:]=0;m.actuator_biasprm[i,:3]=[0,-kp,-kd]
            m.actuator_ctrlrange[i]=[-2.,2.] if i<3 else self.backup['actuator_ctrlrange'][i]+[-.1,.1]
            m.actuator_forcelimited[i]=1;m.actuator_forcerange[i]=[-force,force]
        m.jnt_range[self.tcols]=[-1.5,1.5]
        self.kp=m.actuator_gainprm[:,0].copy()
        self.hand_bodies={m.body_name2id('forearm')}
        for bid in range(m.nbody):
            if m.body_parentid[bid] in self.hand_bodies:self.hand_bodies.add(bid)
        self.hand_geoms=[g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in self.hand_bodies]
        self.object_vertices={}
        for name in ('mug','banana','sugar_box','mustard_bottle','tomato_soup_can'):
            if name+'_0' not in m.body_names:continue
            bid=m.body_name2id(name+'_0')
            self.object_vertices[name]=np.concatenate([mesh_local(m,g)[0] for g in range(m.ngeom)
                if m.geom_bodyid[g]==bid and m.geom_type[g]==7])
        self.fixture_gids=[m.geom_name2id('table_collision')]+[m.geom_name2id(n) for n in m.geom_names
            if n and n.startswith('nav_obstacle_')]
        self.target=self.position();self.steps=0
        self.hand_envelope=self.hand_shapes()-self.position()

    def position(self):return self.sim.data.geom_xpos[self.sim.model.geom_name2id(self.site)].copy()

    def step(self,center):
        center=np.asarray(center,dtype=float)
        if center.shape!=(3,) or not np.isfinite(center).all() or np.any(center<self.config['workspace_min_m']) or np.any(center>self.config['workspace_max_m']):
            raise ValueError('Navigation center outside workspace')
        d=self.sim.data;m=self.sim.model
        desired=self.posture_target.copy();desired[self.tcols]=self.to_joints(center)
        ctrl=desired+d.qfrc_bias[:30]/self.kp
        if hasattr(self,'payload_wrench'):
            jp=d.get_site_jacp('S_grasp').reshape(3,m.nv)[:,:6]
            jr=d.get_site_jacr('S_grasp').reshape(3,m.nv)[:,:6]
            force,torque=self.payload_wrench
            ctrl[:6]+=(jp.T@force+jr.T@torque)/self.kp[:6]
        if self.held_ctrl is not None:ctrl[6:]=self.held_ctrl[6:]
        if hasattr(self,'handoff_seconds') and self.steps*self.config['timestep_s']<self.handoff_seconds:
            t=min(1.,(self.steps+1)*self.config['timestep_s']/self.handoff_seconds)
            blend=t*t*t*(10.+t*(-15.+6.*t))
            old=self.backup
            old_force=(old['actuator_gainprm'][:6,0]*self.previous_ctrl[:6]
                       +old['actuator_biasprm'][:6,0]+old['actuator_biasprm'][:6,1]*d.qpos[:6]
                       +old['actuator_biasprm'][:6,2]*d.qvel[:6])
            new_bias=m.actuator_biasprm[:6,1]*d.qpos[:6]+m.actuator_biasprm[:6,2]*d.qvel[:6]
            old_equivalent=(old_force-new_bias)/self.kp[:6]
            ctrl[:6]=old_equivalent*(1.-blend)+ctrl[:6]*blend
        if np.any(ctrl<m.actuator_ctrlrange[:,0]) or np.any(ctrl>m.actuator_ctrlrange[:,1]):
            raise ValueError('Navigation actuator authority exceeded')
        self.target=center.copy();d.ctrl[:]=ctrl;self.sim.step();self.steps+=1
        if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite hand state')

    def contacts(self):
        row=super().contacts()
        row['posture_error_rad']=float(np.max(np.abs(self.sim.data.qpos[self.posture_cols]-self.posture_target[self.posture_cols])))
        return row

    def restore(self):
        for key,value in self.backup.items():getattr(self.sim.model,key)[:]=value
        self.sim.model.opt.timestep=self.old_dt
