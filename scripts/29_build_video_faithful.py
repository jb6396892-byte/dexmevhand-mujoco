#!/usr/bin/env python3
"""Retarget all valid video frames with five-finger object-relative targets."""
import argparse
import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import transforms3d
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fromrealhand.paths import configure_runtime_paths
from fromrealhand.transforms import transform_points
from fromrealhand.video_fidelity import (HandLandmarks, TIP_INDICES, FINGER_NAMES,
                                        finger_directions, fidelity_metrics, object_relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/processed/seq_dexycb_001/video_faithful_v1/retarget')
    parser.add_argument('--iterations', type=int, default=90)
    parser.add_argument('--joint-margin', type=float, default=.04)
    parser.add_argument('--scene-collisions', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_runtime_paths()
    from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate
    contacts = import_module('21_diagnose_geometry').contacts
    sequence = ROOT/'data/real_data/relocate_mug/seq_dexycb_001'
    old_path = ROOT/'data/processed/seq_dexycb_001/repair_v2/geometry.npz'
    old = np.load(old_path)
    indices = np.array([i for i in range(3, 74)
                       if np.isfinite(np.load(sequence/'hand_pose_mano'/('joints_%06d.npy' % i))).all()])
    camera = np.load(sequence/'calib/camera_to_world.npy')
    joints = transform_points(camera, np.stack([np.load(sequence/'hand_pose_mano'/('joints_%06d.npy' % i)) for i in indices]))
    poses = camera[None] @ np.stack([np.load(sequence/'object_pose'/('%06d.npy' % i)) for i in indices])
    scale, origin, rotation, target = float(old['scale']), old['source_origin'], old['rotation'], old['target_origin']
    joints = scale * (joints-origin) @ rotation.T + target
    poses[:, :3, 3] = scale * (poses[:, :3, 3]-origin) @ rotation.T + target
    poses[:, :3, :3] = rotation[None] @ poses[:, :3, :3]
    env = YCBRelocate(has_renderer=False, object_name='mug', object_scale=scale,
                      friction=(1, .5, .01), solref='-6000 -300', randomness_scale=.25)
    model, data = env.sim.model, env.sim.data
    hand_ids = {i for i in range(model.ngeom) if model.geom_id2name(i) in set(env.robot_geom_names)}
    if args.scene_collisions:
        for i in range(model.ngeom):
            if i in hand_ids or model.geom_id2name(i) in set(env.body_geom_names):
                model.geom_margin[i] = .0002
                model.geom_gap[i] = 0.
    def scene_depths():
        return [max(0., -float(c.dist)) for c in data.contact[:data.ncon]
                if int(c.geom1) in hand_ids or int(c.geom2) in hand_ids]
    landmarks = HandLandmarks(model)
    bounds = np.column_stack([np.maximum(model.jnt_range[:30, 0], model.actuator_ctrlrange[:, 0]),
                              np.minimum(model.jnt_range[:30, 1], model.actuator_ctrlrange[:, 1])])
    bounds[8:, 0] += args.joint_margin
    bounds[8:, 1] -= args.joint_margin
    weights = np.array([.15] + [.25, .7, 1., 5.] * 5)[:, None]
    env.sim.forward()
    limited = model.tendon_limited.astype(bool)
    tendon_jac = data.ten_J[limited, :30].copy()
    tendon_offset = data.ten_length[limited] - tendon_jac @ data.qpos[:30]
    tendon_lower = model.tendon_range[limited, 0] - tendon_offset + .00015
    tendon_upper = model.tendon_range[limited, 1] - tendon_offset - .00015
    tendon_constraint = LinearConstraint(tendon_jac, tendon_lower, tendon_upper)
    previous = old['qpos'][0].copy()
    qpos, reports = [], []
    for k, frame in enumerate(indices):
        data.qpos[30:33] = poses[k, :3, 3]
        data.qpos[33:37] = transforms3d.quaternions.mat2quat(poses[k, :3, :3])
        data.qvel[:] = 0.
        seed = old['qpos'][np.argmin(abs(old['source_frames']-frame))]
        directions = finger_directions(joints[k])
        def objective(q):
            data.qpos[:30] = q
            env.sim.forward()
            points = landmarks.read(data)
            fit = np.sum(weights * (points-joints[k])**2)
            direction = .00025 * np.sum((finger_directions(points)-directions)**2)
            collision = sum(max(0., -c['distance_m']-.0003)**2 for c in contacts(env))
            if args.scene_collisions:
                collision = 20*sum(max(0., depth-.0001)**2 for depth in scene_depths())
            return fit + direction + 500*collision + 1e-5*np.sum((q-seed)**2) + 3e-5*np.sum((q-previous)**2)
        solved = minimize(objective, np.clip(previous, bounds[:, 0], bounds[:, 1]), method='SLSQP',
                          bounds=bounds, constraints=[tendon_constraint],
                          options=dict(maxiter=args.iterations, ftol=1e-9, eps=1e-5))
        if not solved.success:
            solved = minimize(objective, solved.x, method='SLSQP', bounds=bounds,
                              constraints=[tendon_constraint],
                              options=dict(maxiter=3*args.iterations, ftol=1e-9, eps=1e-5))
        if not np.isfinite(solved.x).all():
            raise RuntimeError('Nonfinite retarget at source frame %d' % frame)
        objective(solved.x)
        metrics = fidelity_metrics(landmarks.read(data), poses[k], joints[k], poses[k])
        metrics['max_hand_scene_penetration_m'] = max(scene_depths() or [0.])
        metrics.update(source_frame=int(frame), converged=bool(solved.success),
                       max_tendon_violation_m=float(max(0., np.max(tendon_lower-tendon_jac @ solved.x),
                                                       np.max(tendon_jac @ solved.x-tendon_upper))),
                       max_penetration_m=max([max(0., -c['distance_m']) for c in contacts(env)] or [0.]))
        qpos.append(solved.x.copy())
        previous = solved.x.copy()
        reports.append(metrics)
        print(json.dumps(metrics), flush=True)
    # Nearest mesh-vertex distances are geometry proxies, not measured contact forces.
    import trimesh
    mesh = trimesh.load('/media/smgbro/shared/DexYCB/dataset/models/025_mug/textured_simple.obj', process=False)
    tree = cKDTree(np.asarray(mesh.vertices)*scale)
    distances = np.stack([tree.query(object_relative(j[TIP_INDICES], p))[0] for j, p in zip(joints, poses)])
    np.savez(args.output/'geometry.npz', qpos=np.asarray(qpos), object_poses=poses, human_joints=joints,
             source_frames=indices, fps=30., scale=scale, source_tip_surface_distance_m=distances)
    baseline_paths = [old_path, ROOT/'data/demonstrations/relocate-mug-physics-verified-v1.pkl',
                      ROOT/'data/processed/seq_dexycb_001/physical_grasp_verified_v1/admission.json']
    manifest = dict(baseline_tag='physical-grasp-verified-v1', baseline_commit='5dea040b134a73507481ee71095d621707babaa7',
                    scene_collision_constraints=args.scene_collisions,
                    protected_files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline_paths},
                    source_sequence=json.loads((sequence/'meta.json').read_text())['source_sequence'],
                    source_frames=indices.tolist(), fingers=list(FINGER_NAMES),
                    source_grasp='Thumb opposed to four fingers around cup body; checked in synchronized camera views',
                    camera_views_reviewed=['840412060917', '836212060125', '839512060362', '841412060263', '932122060861'],
                    contact_proxy='Nearest YCB surface vertex to labeled fingertip; no human force labels',
                    mean_tip_error_m=float(np.mean([r['mean_tip_error_m'] for r in reports])),
                    max_penetration_m=max(r['max_penetration_m'] for r in reports),
                    max_tendon_violation_m=max(r['max_tendon_violation_m'] for r in reports),
                    tendon_constraints='Original fixed tendons, with 0.15mm internal margin',
                    finger_joint_control_margin_rad=args.joint_margin,
                    failed_optimizer_frames=[r['source_frame'] for r in reports if not r['converged']],
                    training_ready=False)
    (args.output/'frames.json').write_text(json.dumps(reports, indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)
    env.close()


if __name__ == '__main__':
    main()
