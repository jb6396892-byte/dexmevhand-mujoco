#!/usr/bin/env python3
"""Generate diagnostic inverse actions, then test free-object physical rollouts."""
import argparse
import csv
import json
import pickle
import sys
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import transforms3d
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fromrealhand.paths import configure_runtime_paths
from fromrealhand.action_recovery import force_to_action
contacts=import_module('21_diagnose_geometry').contacts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('geometry',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    configure_runtime_paths()
    from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate
    from mujoco_py import functions, MjRenderContextOffscreen
    args.output.mkdir(parents=True,exist_ok=True)
    class AuditedRelocate(YCBRelocate):
        def _pre_action(self, action, policy_step=False):
            if policy_step:
                self.substep_contacts = []
            self.substep_contacts.extend(contacts(self))
            return super()._pre_action(action, policy_step)

    data=np.load(args.geometry)
    source_time=np.arange(len(data['qpos']))/float(data['fps'])
    duration=source_time[-1]
    qcurve=PchipInterpolator(source_time,data['qpos'],axis=0)
    pcurve=PchipInterpolator(source_time,data['object_poses'][:,:3,3],axis=0)
    rotation=Slerp(source_time,Rotation.from_matrix(data['object_poses'][:,:3,:3]))
    def make_env():
        e=AuditedRelocate(has_renderer=False,object_name='mug',object_scale=.8,friction=(1,.5,.01),solref='-6000 -300',randomness_scale=.25)
        e.sim.model.body_pos[e.target_object_bid]=data['object_poses'][-1,:3,3]
        e.sim.model.body_quat[e.target_object_bid]=transforms3d.quaternions.mat2quat(data['object_poses'][-1,:3,:3])
        return e
    env=make_env(); dt=env.control_timestep
    times=np.arange(int(np.ceil(duration/dt))+101)*dt
    q=qcurve(np.minimum(times,duration))
    v=qcurve.derivative()(np.minimum(times,duration)); a=qcurve.derivative(2)(np.minimum(times,duration))
    v[times>=duration]=0; a[times>=duration]=0
    positions=pcurve(np.minimum(times,duration))
    quaternions=rotation(np.minimum(times,duration)).as_quat()[:,[3,0,1,2]]
    candidate={'observations':[],'actions':[],'rewards':[],'sim_data':[],'model_data':[env.dump_mujoco_model()]}
    max_kinematic_penetration=0.
    for i in range(len(times)):
        env.sim.data.qpos[:30]=q[i]; env.sim.data.qpos[30:33]=positions[i]; env.sim.data.qpos[33:37]=quaternions[i]
        env.sim.data.qvel[:]=0; env.sim.data.qvel[:30]=v[i]
        env.sim.forward()
        # forward refreshes geometry but overwrites qacc; restore trajectory acceleration for inverse dynamics.
        env.sim.data.qacc[:]=0; env.sim.data.qacc[:30]=a[i]
        observation=env._get_observations().copy()
        functions.mj_inverse(env.sim.model,env.sim.data)
        model=env.sim.model
        action=force_to_action(env.sim.data.qfrc_inverse[:30],q[i],v[i],model.actuator_gainprm[:,0],model.actuator_biasprm,
                               env.act_mid,env.act_rng)
        candidate['observations'].append(observation); candidate['actions'].append(action)
        candidate['rewards'].append(float(env.reward(None))); candidate['sim_data'].append(env.dump())
        max_kinematic_penetration=max(max_kinematic_penetration,max([max(0,-c['distance_m']) for c in contacts(env)] or [0]))
    for key in ['observations','actions','rewards']: candidate[key]=np.asarray(candidate[key])
    diagnostic_path=args.output/'relocate-mug-repair-diagnostic.pkl'
    with diagnostic_path.open('wb') as f: pickle.dump({'repair_diagnostic':candidate},f)
    runs=[]
    for name,speed,kp_factor in [('inverse_openloop',1.,0.),('feedback',1.,1.),('feedback_gain2',1.,2.),('feedback_slow',2.,1.)]:
        e=make_env(); e.pack(candidate['sim_data'][0]); e.sim.forward()
        initial_height=float(e.sim.data.body_xpos[e.obj_bid,2]); context=MjRenderContextOffscreen(e.sim)
        video=cv2.VideoWriter(str(args.output/(name+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),25,(640,480))
        if not video.isOpened(): raise RuntimeError('video writer failed')
        rows=[]; observations=[]; actions=[]; states=[]; rewards=[]
        streak=0; max_streak=0; error=None
        count=len(times) if kp_factor==0 else int(np.ceil(duration*speed/dt))+101
        for i in range(count):
            t=min(i*dt/speed,duration); desired=qcurve(t); velocity=qcurve.derivative()(t)/speed if t<duration else np.zeros(30)
            if kp_factor==0: action=candidate['actions'][i]
            else:
                kp=np.r_[np.full(6,300.),np.full(2,10.),np.full(22,2.)]*kp_factor
                kd=np.r_[np.full(6,15.),np.full(2,1.),np.full(22,.08)]*np.sqrt(kp_factor)
                force=e.sim.data.qfrc_bias[:30]+kp*(desired-e.sim.data.qpos[:30])+kd*(velocity-e.sim.data.qvel[:30])
                action=force_to_action(force,e.sim.data.qpos[:30],e.sim.data.qvel[:30],e.sim.model.actuator_gainprm[:,0],e.sim.model.actuator_biasprm,e.act_mid,e.act_rng)
            observations.append(e._get_observations().copy()); actions.append(action.copy()); states.append(e.dump())
            try: _,reward,_,_=e.step(action)
            except Exception as exc:
                error=str(exc); break
            rewards.append(reward)
            cs=e.substep_contacts+contacts(e); penetrating=max([max(0,-c['distance_m']) for c in cs] or [0])
            height=float(e.sim.data.body_xpos[e.obj_bid,2]); is_contact=any(c['normal_force_n']>1e-3 for c in cs)
            streak=streak+1 if is_contact and height>initial_height+.015 else 0; max_streak=max(max_streak,streak)
            rows.append({'step':i,'time_s':i*dt,'height_m':height,'contact':int(is_contact),'penetration_m':penetrating,
                         'target_distance_m':float(np.linalg.norm(e.sim.data.body_xpos[e.obj_bid]-e.sim.model.body_pos[e.target_object_bid])),
                         'hand_tracking_rmse':float(np.sqrt(np.mean((desired-e.sim.data.qpos[:30])**2)))})
            if i%4==0:
                context.render(640,480,camera_id=e.sim.model.camera_name2id('frontview'))
                video.write(cv2.cvtColor(context.read_pixels(640,480,depth=False)[::-1],cv2.COLOR_RGB2BGR))
        video.release()
        with (args.output/(name+'.csv')).open('w') as f:
            if rows:
                w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        summary={'mode':name,'steps':len(rows),'initial_height_m':initial_height,'max_height_m':max([r['height_m'] for r in rows] or [initial_height]),
                 'contact_steps':sum(r['contact'] for r in rows),'max_penetration_m':max([r['penetration_m'] for r in rows] or [0]),
                 'contact_lift_duration_s':max_streak*dt,'final_target_distance_m':rows[-1]['target_distance_m'] if rows else None,
                 'action_saturation_fraction':float(np.mean(np.abs(actions)>=.999)), 'error':error}
        summary['passed']=error is None and max_streak*dt>=.5 and summary['max_penetration_m']<=.005
        runs.append(summary)
        # A physics rollout is eligible for training only after its contact/lift gate passes.
        if summary['passed']:
            demo={'observations':np.asarray(observations),'actions':np.asarray(actions),'rewards':np.asarray(rewards),
                  'sim_data':states,'model_data':[e.dump_mujoco_model()]}
            with (args.output/(name+'_physical_demo.pkl')).open('wb') as f: pickle.dump({name:demo},f)
        print(json.dumps(summary,indent=2),flush=True)
    report={'source_fps':float(data['fps']),'control_timestep':dt,'diagnostic_demo':str(diagnostic_path),
            'diagnostic_demo_training_eligible':False,'kinematic_interpolation_max_penetration_m':max_kinematic_penetration,
            'runs':runs,'physical_gate_passed':any(r['passed'] for r in runs)}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['physical_gate_passed'] else 1

if __name__=='__main__': raise SystemExit(main())
