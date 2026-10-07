"""Grasp-retaining actuator transport. The cup remains an unmodified free body."""
import numpy as np
from .local_adapter import MotionBridge
from .navigation_runner import navigate
from .navigation import NavigationRejected
from ..tabletop.contact_control import opposing_contacts,FINGERS


def supported(env):
    row=env.contacts();row.update(opposing_contacts(env.sim));row.pop('pairs',None)
    row['supported']=bool(row['th_force_n']>.01 and sum(row[f+'_force_n']>.01 for f in FINGERS)>=3 and row['opposition'])
    return row


class LoadedMotion(MotionBridge):
    def __init__(self,env,config,held_action):
        super().__init__(env,config,held_action=held_action)
        m=self.sim.model
        mug=m.body_name2id('mug_0')
        force=-float(m.body_mass[mug])*m.opt.gravity.copy()
        lever=self.sim.data.xipos[mug]-self.sim.data.get_site_xpos('S_grasp')
        self.payload_wrench=(force,np.cross(lever,force))
        self.handoff_seconds=2.
        self.hand_bodies.add(mug)
        self.hand_geoms.extend([g for g in range(m.ngeom) if m.geom_bodyid[g]==mug])
        self.hand_envelope=self.hand_shapes()-self.position()
        self.lost_seconds=0.;self.max_penetration=0.;self.support_samples=0;self.samples=0

    def obstacles(self):return [r for r in super().obstacles() if r['name']!='mug']

    def contacts(self):
        row=super().contacts()
        q=supported(self.env)
        error=np.abs(self.sim.data.qpos[:30]-self.posture_target)
        row['posture_error_rad']=float(error[3:6].max())
        row['finger_compliance_error_rad']=float(error[6:].max())
        self.samples+=1;self.support_samples+=int(q['supported'])
        self.lost_seconds=0. if q['supported'] else self.lost_seconds+self.config['timestep_s']
        self.max_penetration=max(self.max_penetration,q['scene_penetration_m'])
        row['payload_valid']=(self.lost_seconds<.15 and self.max_penetration<=.001
                              and row['finger_compliance_error_rad']<=.05)
        row.update(payload_supported=q['supported'],payload_lost_seconds=self.lost_seconds,
                   grasp_penetration_m=self.max_penetration)
        return row


def carry(env,held_action,cup_goal,config,on_step=None):
    # Let the local lift finish dynamically before handing control to the transit servo.
    settle=[]
    def audit():
        q=supported(env)
        if q['scene_penetration_m']>.001 or q['non_target_contacts']:
            raise NavigationRejected('carry_handoff_physics_violation')
    for _ in range(int(round(3./env.dt))):
        env.step(held_action,audit);settle.append(supported(env))
    if not all(r['supported'] for r in settle[-20:]):
        raise NavigationRejected('carry_handoff_not_stably_grasped')
    bridge=LoadedMotion(env,config,held_action)
    mug=env.sim.model.body_name2id('mug_0')
    initial=env.sim.data.body_xpos[mug].copy()
    target=bridge.position()+np.asarray(cup_goal)-initial
    # Long-distance carrying is deliberately slower than empty-hand transit.
    bridge.config=dict(config,max_velocity_m_s=[.04,.04,.03],max_acceleration_m_s2=[.06,.06,.05],
                       max_jerk_m_s3=[.2,.2,.15])
    try:
        result=navigate(bridge,target,on_step)
        hold=[]
        for i in range(int(round(1./bridge.config['timestep_s']))):
            bridge.step(bridge.target);q=bridge.contacts()
            hold.append(q['payload_valid'] and supported(env)['supported'])
            if not q['payload_valid'] or q['hand_environment_contacts']:
                raise NavigationRejected('carry_final_hold_failed')
        final=env.sim.data.body_xpos[mug].copy();error=float(np.linalg.norm(final-cup_goal))
        result.update(cup_start_m=initial.tolist(),cup_goal_m=np.asarray(cup_goal).tolist(),
            cup_final_m=final.tolist(),cup_error_m=error,hold_seconds=1.,hold_supported_fraction=float(np.mean(hold)),
            support_fraction=bridge.support_samples/max(1,bridge.samples),max_grasp_penetration_m=bridge.max_penetration,
            cup_freejoint_preserved=True,object_pose_writes_during_execution=0,
            payload_feedforward_force_n=bridge.payload_wrench[0].tolist(),
            payload_feedforward_torque_nm=bridge.payload_wrench[1].tolist(),
            controller_handoff_seconds=bridge.handoff_seconds,
            pre_handoff_settling_seconds=3.,
            cup_pose_input='measured once at carry handoff for payload/goal calibration; live pose for safety only')
        result['passed']=bool(result['passed'] and error<=.02 and all(hold))
        return result
    except NavigationRejected as error:
        result=getattr(bridge,'last_navigation',{})
        result.update(passed=False,reason=str(error),payload_audit=supported(env),
                      cup_final_m=env.sim.data.body_xpos[mug].tolist())
        return result
    finally:bridge.restore()
