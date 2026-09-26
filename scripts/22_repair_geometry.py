#!/usr/bin/env python3
"""Transfer a real sequence into a reachable task frame and refine contacts."""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import transforms3d
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fromrealhand.paths import configure_runtime_paths
from fromrealhand.transforms import transform_points
from importlib import import_module
contacts = import_module('21_diagnose_geometry').contacts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=100)
    args=p.parse_args()
    configure_runtime_paths()
    from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate
    from hand_imitation.env.utils.mjcf_utils import xml_path_completion
    from hand_imitation.kinematics.retargeting_optimizer import ChainMatchingPositionKinematicsRetargeting
    from mujoco_py import MjRenderContextOffscreen
    seq=ROOT/'data/real_data/relocate_mug/seq_dexycb_001'
    args.output.mkdir(parents=True,exist_ok=True)
    indices=np.arange(20,74)
    camera=np.load(seq/'calib/camera_to_world.npy')
    joints=transform_points(camera,np.stack([np.load(seq/'hand_pose_mano'/('joints_%06d.npy'%i)) for i in indices]))
    frames=camera[None,None] @ np.stack([np.load(seq/'hand_pose_mano'/('results_global_%06d.npy'%i)) for i in indices])
    poses=camera[None] @ np.stack([np.load(seq/'object_pose'/('%06d.npy'%i)) for i in indices])
    env=YCBRelocate(has_renderer=False,object_name='mug',object_scale=.8,friction=(1,.5,.01),solref='-6000 -300',randomness_scale=.25)
    m=env.sim.model
    default_palm=env.sim.data.body_xpos[m.body_name2id('palm')].copy()
    origin=poses[0,:3,3].copy()
    target_origin=np.array([-.0625,0,.04065])
    source_direction=joints[0,0,:2]-origin[:2]
    desired_direction=default_palm[:2]-target_origin[:2]
    yaw=np.arctan2(desired_direction[1],desired_direction[0])-np.arctan2(source_direction[1],source_direction[0])
    rotation=transforms3d.euler.euler2mat(0,0,yaw)
    joints=.8*(joints-origin) @ rotation.T + target_origin
    frames[:,:,:3,3]=.8*(frames[:,:,:3,3]-origin) @ rotation.T + target_origin
    frames[:,:,:3,:3]=rotation[None,None] @ frames[:,:,:3,:3]
    poses[:,:3,3]=.8*(poses[:,:3,3]-origin) @ rotation.T + target_origin
    poses[:,:3,:3]=rotation[None] @ poses[:,:3,:3]
    env.sim.data.qpos[30:33]=poses[0,:3,3]
    env.sim.data.qpos[33:37]=transforms3d.quaternions.mat2quat(poses[0,:3,:3])
    env.sim.forward()
    bottom=np.inf
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == env.obj_bid and m.geom_type[g] == 7 and m.geom_contype[g]:
            mid=m.geom_dataid[g]
            v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
            world=v @ env.sim.data.geom_xmat[g].reshape(3,3).T+env.sim.data.geom_xpos[g]
            bottom=min(bottom,float(world[:,2].min()))
    dz=.001-bottom
    joints[:,:,2]+=dz; frames[:,:,2,3]+=dz; poses[:,2,3]+=dz
    links=['palm','thmiddle','ffmiddle','mfmiddle','rfmiddle','lfmiddle']
    human=[0,2,6,10,14,18]
    solver=ChainMatchingPositionKinematicsRetargeting(xml_path_completion('adroit/adroit_relocate.xml'),links,has_joint_limits=True,has_global_pose_limits=True)
    initial=np.asarray(solver.retarget(joints[:,human],frames,name='canonical_scale08',verbose=False))
    body_ids=[m.body_name2id(s) for s in links]
    tip_ids=[m.site_name2id('S_'+s+'tip') for s in ['th','ff','mf','rf','lf']]
    targets=joints[:,human+[4,8,12,16,20]]
    bounds=np.column_stack([np.maximum(m.jnt_range[:30,0],m.actuator_ctrlrange[:,0]),np.minimum(m.jnt_range[:30,1],m.actuator_ctrlrange[:,1])])
    qpos=[]; rows=[]; previous=initial[0]
    for i,seed in enumerate(initial):
        env.sim.data.qpos[30:33]=poses[i,:3,3]
        env.sim.data.qpos[33:37]=transforms3d.quaternions.mat2quat(poses[i,:3,:3])
        env.sim.data.qvel[:]=0
        def objective(q):
            env.sim.data.qpos[:30]=q
            env.sim.forward()
            points=np.vstack([env.sim.data.body_xpos[body_ids],env.sim.data.site_xpos[tip_ids]])
            fit=np.sum((points-targets[i])**2)
            collision=sum(max(0,-c['distance_m']-.0003)**2 for c in contacts(env))
            return fit+300*collision+2e-5*np.sum((q-seed)**2)+1e-4*np.sum((q-previous)**2)
        result=minimize(objective,np.clip(seed if i == 0 else previous,bounds[:,0],bounds[:,1]),method='SLSQP',bounds=bounds,
                        options={'maxiter':args.iterations,'ftol':1e-9,'eps':1e-5})
        objective(result.x)
        cs=contacts(env)
        points=np.vstack([env.sim.data.body_xpos[body_ids],env.sim.data.site_xpos[tip_ids]])
        rows.append({'frame':int(indices[i]),'converged':bool(result.success),'iterations':int(result.nit),
                     'max_penetration_m':max([max(0,-c['distance_m']) for c in cs] or [0]),
                     'mean_target_error_m':float(np.linalg.norm(points-targets[i],axis=1).mean())})
        previous=result.x.copy(); qpos.append(previous)
        print(rows[-1],flush=True)
    qpos=np.asarray(qpos)
    np.savez(args.output/'geometry.npz',qpos=qpos,object_poses=poses,source_frames=indices,targets=targets,
             source_origin=origin,target_origin=target_origin+np.array([0,0,dz]),rotation=rotation,scale=.8,fps=30.)
    with (args.output/'retargeting.pkl').open('wb') as f: pickle.dump(qpos,f)
    with (args.output/'frames.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    report={'yaw_degrees':float(np.rad2deg(yaw)),'table_support_shift_m':float(dz),'scale':.8,'fps':30,
            'max_penetration_m':max(r['max_penetration_m'] for r in rows),'frames_over_1mm':sum(r['max_penetration_m']>.001 for r in rows),
            'optimizer_failed_frames':[r['frame'] for r in rows if not r['converged']],
            'mean_target_error_m':float(np.mean([r['mean_target_error_m'] for r in rows])),
            'root_min':qpos[:,:6].min(0).tolist(),'root_max':qpos[:,:6].max(0).tolist()}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    context=MjRenderContextOffscreen(env.sim)
    m.body_pos[env.target_object_bid]=poses[-1,:3,3]
    m.body_quat[env.target_object_bid]=transforms3d.quaternions.mat2quat(poses[-1,:3,:3])
    for i in [0,10,20,30,40,53]:
        env.sim.data.qpos[:30]=qpos[i]; env.sim.data.qpos[30:33]=poses[i,:3,3]
        env.sim.data.qpos[33:37]=transforms3d.quaternions.mat2quat(poses[i,:3,:3]); env.sim.forward()
        context.render(640,480,camera_id=m.camera_name2id('frontview'))
        cv2.imwrite(str(args.output/('frame_%06d.jpg'%indices[i])),cv2.cvtColor(context.read_pixels(640,480,depth=False)[::-1],cv2.COLOR_RGB2BGR))
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
