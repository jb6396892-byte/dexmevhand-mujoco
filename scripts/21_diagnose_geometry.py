#!/usr/bin/env python3
"""Measure hand/mug intersections and compare source and simulator geometry."""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import transforms3d
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fromrealhand.paths import configure_runtime_paths
from fromrealhand.dexycb_io import project_points


def contacts(env):
    from mujoco_py import functions
    hands = set(env.robot_geom_names)
    mugs = set(env.body_geom_names)
    result = []
    for index, c in enumerate(env.sim.data.contact[:env.sim.data.ncon]):
        a, b = (env.sim.model.geom_id2name(int(g)) for g in (c.geom1, c.geom2))
        if (a in hands and b in mugs) or (b in hands and a in mugs):
            force = np.zeros(6)
            functions.mj_contactForce(env.sim.model, env.sim.data, index, force)
            result.append({'hand': a if a in hands else b, 'mug': b if b in mugs else a,
                           'distance_m': float(c.dist), 'normal_force_n': float(force[0]), 'position': c.pos.copy().tolist()})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--retargeting', type=Path)
    args = p.parse_args()
    configure_runtime_paths()
    from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate
    from mujoco_py import MjRenderContextOffscreen
    seq = ROOT / 'data/real_data/relocate_mug/seq_dexycb_001'
    args.output.mkdir(parents=True, exist_ok=True)
    q = pickle.load(open(args.retargeting or seq / 'retargeting_mano_aligned.pkl', 'rb'))
    transform = np.load(seq / 'calib/camera_to_mujoco.npy')
    inverse = np.linalg.inv(transform)
    k = np.load(seq / 'calib/camera_matrix.npy')
    env = YCBRelocate(has_renderer=False, object_name='mug', object_scale=.8,
                      friction=(1,.5,.01), solref='-6000 -300', randomness_scale=.25)
    ctx = MjRenderContextOffscreen(env.sim)
    m = env.sim.model
    source = trimesh.load('/media/smgbro/shared/DexYCB/dataset/models/025_mug/textured_simple.obj', process=False)
    source_vertices = np.asarray(source.vertices)
    visual_id = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == env.obj_bid and m.geom_type[i] == 7 and m.geom_group[i] == 1)
    mesh = m.geom_dataid[visual_id]
    vertices = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]].copy()
    rows, details, errors = [], [], []
    sample = {20, 30, 40, 50, 60, 73}
    for i in range(20, min(len(q),74)):
        pose = transform @ np.load(seq / 'object_pose' / ('%06d.npy' % i))
        env.sim.data.qpos[:30] = q[i]
        env.sim.data.qpos[30:33] = pose[:3,3]
        env.sim.data.qpos[33:37] = transforms3d.quaternions.mat2quat(pose[:3,:3])
        env.sim.data.qvel[:] = 0
        env.sim.forward()
        cs = contacts(env)
        depth = max([max(0,-c['distance_m']) for c in cs] or [0])
        rows.append({'frame':i,'time_s':i/30.,'contacts':len(cs),'max_penetration_m':depth,
                     'object_height_m':float(pose[2,3])})
        details.append({'frame':i,'contacts':cs})
        world_vertices = vertices @ env.sim.data.geom_xmat[visual_id].reshape(3,3).T + env.sim.data.geom_xpos[visual_id]
        local_vertices = (world_vertices-pose[:3,3]) @ pose[:3,:3]
        joints = np.load(seq / 'hand_pose_mano' / ('joints_%06d.npy' % i))
        label = np.load(seq / 'source_labels' / ('%06d.npz' % i))
        projected, valid = project_points(joints,k)
        annotated = label['joint_2d'].reshape(21,2)
        valid &= (annotated >= 0).all(axis=1)
        errors.extend(np.linalg.norm(projected[valid]-annotated[valid],axis=1).tolist())
        if i in sample or (depth > .001 and not any(r['max_penetration_m'] > .001 for r in rows[:-1])):
            image = cv2.imread(str(seq / 'rgb' / ('%06d.jpg' % i)))
            camera_pose = np.load(seq / 'object_pose' / ('%06d.npy' % i))
            for points,color in [(source_vertices[::20],(0,220,0)),(local_vertices[::10],(0,0,255))]:
                camera = points @ camera_pose[:3,:3].T + camera_pose[:3,3]
                pix,vis = project_points(camera,k)
                for x,y in pix[vis]:
                    if 0 <= x < 640 and 0 <= y < 480:
                        cv2.circle(image,(int(x),int(y)),1,color,-1)
            for x,y in projected[valid]:
                cv2.circle(image,(int(x),int(y)),3,(255,180,0),-1)
            ctx.render(640,480,camera_id=m.camera_name2id('frontview'))
            rendered = cv2.cvtColor(ctx.read_pixels(640,480,depth=False)[::-1],cv2.COLOR_RGB2BGR)
            cv2.putText(rendered,'frame %d penetration %.1f mm'%(i,depth*1000),(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
            cv2.imwrite(str(args.output / ('frame_%06d.jpg'%i)),np.hstack([image,rendered]))
    with (args.output/'frames.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output/'contacts.json').write_text(json.dumps(details,indent=2))
    distance=cKDTree(source_vertices).query(local_vertices/.8)[0]
    report={'frames':len(rows),'first_penetration_frame':next((r['frame'] for r in rows if r['max_penetration_m']>.001),None),
            'max_penetration_m':max(r['max_penetration_m'] for r in rows),
            'frames_over_1mm':sum(r['max_penetration_m']>.001 for r in rows),
            'source_mesh_bounds_m':source.bounds.tolist(),'sim_visual_bounds_m':np.array([local_vertices.min(0),local_vertices.max(0)]).tolist(),
            'unscaled_sim_to_source_mean_distance_m':float(distance.mean()),
            'unscaled_sim_to_source_max_distance_m':float(distance.max()),
            'source_joint_reprojection_mean_px':float(np.mean(errors)),
            'source_joint_reprojection_max_px':float(np.max(errors)),
            'source_fps':30,'legacy_demo_fps':25}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
