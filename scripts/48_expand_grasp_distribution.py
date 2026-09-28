#!/usr/bin/env python3
"""Validate joint scene/goal transforms and admit only executable synthetic demos."""
import argparse
import hashlib
import json
import pickle
from importlib import import_module
from pathlib import Path

import numpy as np
import transforms3d

finger = import_module('37_optimize_finger_reference')
from fromrealhand.policy_learning import StudentActions, lift_success
from check_codex_budget import latest_usage


def transform_geometry(exp, offset, yaw_degrees, goal_offset):
    g = {key: value.copy() for key, value in exp.geometry.items()}
    m, d = exp.model, exp.env.sim.data
    body = m.body_name2id('forearm')
    base_rotation = transforms3d.quaternions.quat2mat(m.body_quat[body])
    origin = g['object_poses'][0, :3, 3].copy()
    rotation = transforms3d.euler.euler2mat(0., 0., np.deg2rad(yaw_degrees))
    phase = np.clip((g['source_frames']-40.5)/(g['source_frames'][-1]-40.5), 0., 1.)
    blend = phase**3*(10.-15.*phase+6.*phase**2)
    maximum_error = 0.
    for i, q in enumerate(g['qpos'].copy()):
        delta = np.asarray(offset)+blend[i]*np.asarray(goal_offset)
        d.qpos[:30] = q
        exp.env.sim.forward()
        old_points = exp.landmarks.read(d).copy()
        root_position = m.body_pos[body]+base_rotation @ q[:3]
        root_rotation = base_rotation @ transforms3d.euler.euler2mat(*q[3:6], axes='rxyz')
        new_position = rotation @ (root_position-origin)+origin+delta
        g['qpos'][i, :3] = base_rotation.T @ (new_position-m.body_pos[body])
        g['qpos'][i, 3:6] = transforms3d.euler.mat2euler(base_rotation.T @ rotation @ root_rotation, axes='rxyz')
        g['human_joints'][i] = (g['human_joints'][i]-origin) @ rotation.T+origin+delta
        g['object_poses'][i, :3, 3] = rotation @ (g['object_poses'][i, :3, 3]-origin)+origin+delta
        g['object_poses'][i, :3, :3] = rotation @ g['object_poses'][i, :3, :3]
        d.qpos[:30] = g['qpos'][i]
        exp.env.sim.forward()
        maximum_error = max(maximum_error, float(np.max(np.abs(exp.landmarks.read(d)-((old_points-origin) @ rotation.T+origin+delta)))))
    violation = max(0., float(np.max(m.jnt_range[:30, 0]-g['qpos'])), float(np.max(g['qpos']-m.jnt_range[:30, 1])))
    return g, maximum_error, violation


def configure(exp, b):
    exp.correction = np.asarray(b['joint_correction'])
    exp.closure_lead = b.get('closure_lead', 0.)
    exp.feedback_weights = b.get('feedback_weights')
    exp.approach_gain = b.get('approach_gain', 0.)
    exp.approach_root_gain = b.get('approach_root_gain')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy', type=Path)
    p.add_argument('--case-set', choices=['train', 'heldout', 'orientation', 'test-v5', 'test-v6'], default='train')
    p.add_argument('--quota-stop', type=float, default=85.)
    p.add_argument('--skip-expert-replay', action='store_true', help='Evaluation sets only: compare the two policies without generating expert demonstrations')
    args = p.parse_args()
    if args.skip_expert_replay and args.case_set not in ('heldout', 'test-v5', 'test-v6'):
        p.error('--skip-expert-replay is only valid for evaluation sets')
    root = finger.surface.ROOT
    source = json.loads((root/'data/processed/seq_dexycb_001/scene_fidelity_v3/approach_feedback/admission.json').read_text())
    source_admission = json.loads((root/'data/processed/seq_dexycb_001/scene_fidelity_v3/verified_export/admission.json').read_text())
    raw = (root/'data/demonstrations/relocate-mug-video-faithful-v3.pkl').read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    if not source_admission.get('training_ready') or source_hash != source_admission['demo_sha256']:
        raise ValueError('Original demonstration admission or hash failed')
    original = pickle.loads(raw)
    demos = dict(original)
    nominal = original['video_seed_0']['actions']
    checkpoint = None
    if args.policy:
        with args.policy.open('rb') as stream:
            checkpoint = pickle.load(stream)
        if checkpoint.get('action_reference') is not None:
            nominal = np.asarray(checkpoint['action_reference'])
    args.output.mkdir(parents=True, exist_ok=False)
    base = finger.CorrectedExperiment(source['geometry'])
    b = source['best']
    kwargs = dict(scale=b['time_scale'], close=b['close'], gain=b['cartesian_gain'])
    cases = [('nominal', [0, 0, 0], 0, [0, 0, 0])]
    for axis in range(2):
        for sign in (-1, 1):
            offset = np.zeros(3); offset[axis] = sign*.005
            goal = np.zeros(3); goal[axis] = sign*.01
            cases.append(('position_%d_%d'%(axis, sign), offset, 0., [0, 0, 0]))
            cases.append(('goal_%d_%d'%(axis, sign), [0, 0, 0], 0., goal))
    cases += [('yaw_minus_2', [0, 0, 0], -2., [0, 0, 0]), ('yaw_plus_2', [0, 0, 0], 2., [0, 0, 0])]
    if args.case_set == 'orientation':
        cases = [('cup_yaw_%d'%angle, [0, 0, 0], 0., [0, 0, 0]) for angle in (-10, -5, 5, 10)]
    elif args.case_set == 'heldout':
        cases = []
        for axis in range(2):
            for delta in (-.007, -.003):
                offset = np.zeros(3); offset[axis] = delta
                cases.append(('held_position_%d_%g'%(axis, delta), offset, 0., [0, 0, 0]))
            for delta in (-.015, .015):
                goal = np.zeros(3); goal[axis] = delta
                cases.append(('held_goal_%d_%g'%(axis, delta), [0, 0, 0], 0., goal))
        cases += [('cup_yaw_-3', [0, 0, 0], 0., [0, 0, 0]), ('cup_yaw_3', [0, 0, 0], 0., [0, 0, 0])]
    elif args.case_set == 'test-v5':
        cases = []
        for axis in range(2):
            for delta in (-.006, -.004):
                offset = np.zeros(3); offset[axis] = delta
                cases.append(('test_position_%d_%g'%(axis, delta), offset, 0., [0, 0, 0]))
            for delta in (-.018, .018):
                goal = np.zeros(3); goal[axis] = delta
                cases.append(('test_goal_%d_%g'%(axis, delta), [0, 0, 0], 0., goal))
        cases += [('cup_yaw_-7', [0, 0, 0], 0., [0, 0, 0]), ('cup_yaw_7', [0, 0, 0], 0., [0, 0, 0])]
    elif args.case_set == 'test-v6':
        cases = []
        for axis in range(2):
            for delta in (-.008, -.002):
                offset = np.zeros(3); offset[axis] = delta
                cases.append(('test_position_%d_%g'%(axis, delta), offset, 0., [0, 0, 0]))
            for delta in (-.022, .022):
                goal = np.zeros(3); goal[axis] = delta
                cases.append(('test_goal_%d_%g'%(axis, delta), [0, 0, 0], 0., goal))
        cases += [('cup_yaw_-8', [0, 0, 0], 0., [0, 0, 0]), ('cup_yaw_8', [0, 0, 0], 0., [0, 0, 0])]
    results = []
    try:
        for name, offset, yaw, goal in cases:
            quota = latest_usage()
            if quota and quota['used_percent'] >= args.quota_stop:
                break
            folder = args.output/name
            folder.mkdir()
            geometry, fk_error, violation = transform_geometry(base, offset, yaw, goal)
            object_yaw = float(name.split('_')[-1]) if name.startswith('cup_yaw_') else 0.
            if object_yaw:
                cup_rotation = transforms3d.euler.euler2mat(0., 0., np.deg2rad(object_yaw))
                geometry['object_poses'][:, :3, :3] = cup_rotation[None] @ geometry['object_poses'][:, :3, :3]
            result = dict(name=name, offset_m=list(offset), yaw_degrees=yaw, goal_offset_m=list(goal),
                          fk_error_m=fk_error, joint_violation=violation, source_type='synthetic transform of seq_dexycb_001')
            result['object_yaw_degrees'] = object_yaw
            if fk_error > 1e-8 or violation > 1e-8:
                result['rejected_geometry'] = True
                results.append(result)
                continue
            path = folder/'geometry.npz'
            np.savez(path, **geometry)
            exp = finger.CorrectedExperiment(path)
            try:
                configure(exp, b)
                expert = replay = error = admitted = None
                if not args.skip_expert_replay:
                    expert, actions = exp.run_surface(**kwargs, output=folder/'expert')
                    demo = exp.last_demo
                    replay, _ = exp.run_surface(**kwargs, saved_actions=actions)
                    error = float(np.max(np.abs(demo['observations']-exp.last_demo['observations'])))
                    admitted = bool(expert['surface_physics_passed'] and expert['fidelity_passed']
                                    and replay['surface_physics_passed'] and replay['fidelity_passed'] and error < 1e-8)
                zero, _ = exp.run_surface(**kwargs, saved_actions=nominal)
                result.update(expert=expert, replay=replay, replay_error=error, admitted=admitted, nominal_actions=zero)
                if checkpoint:
                    student, _ = exp.run_surface(**kwargs, saved_actions=StudentActions(checkpoint, exp, len(nominal)), output=folder/'student')
                    result['student'] = student
                if admitted and name != 'nominal' and args.case_set not in ('heldout', 'test-v5', 'test-v6'):
                    demos['synthetic_'+name] = demo
                results.append(result)
                (args.output/'progress.json').write_text(json.dumps(results, indent=2)+'\n')
                print(json.dumps(dict(name=name, admitted=admitted, nominal_lift=lift_success(zero),
                    student_full=bool(result.get('student', {}).get('surface_physics_passed', False) and result.get('student', {}).get('fidelity_passed', False)))), flush=True)
            finally:
                exp.env.close()
        if args.case_set in ('heldout', 'test-v5', 'test-v6'):
            (args.output/'evaluation.json').write_text(json.dumps(dict(reports=results,
                policy=str(args.policy), policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest() if args.policy else None,
                original_demo_sha256=source_hash, case_set=args.case_set,
                nominal_source='checkpoint action_reference' if checkpoint and checkpoint.get('action_reference') is not None else 'original demonstration',
                expert_replay_evaluated=not args.skip_expert_replay,
                evaluation_only=True, planned_cases=len(cases), completed_cases=len(results)), indent=2)+'\n')
            return
        demo_path = args.output/'expanded_demonstrations.pkl'
        with demo_path.open('xb') as stream:
            pickle.dump(demos, stream)
        manifest = dict(training_ready=True, demo_sha256=hashlib.sha256(demo_path.read_bytes()).hexdigest(),
                        demo=str(demo_path.resolve()), trajectory_count=len(demos), original_count=len(original),
                        independent_real_sequences=1, synthetic_count=len(demos)-len(original), reports=results,
                        original_demo_sha256=source_hash, planned_cases=len(cases), completed_cases=len(results),
                        scope='Original admitted demonstrations plus individually replay-validated synthetic scene/goal transforms')
        (args.output/'admission.json').write_text(json.dumps(manifest, indent=2)+'\n')
    finally:
        base.env.close()


if __name__ == '__main__':
    main()
