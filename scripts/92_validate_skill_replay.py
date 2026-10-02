#!/usr/bin/env python3
"""Validate isolated skill initialization and continuous, no-reset action replay."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from stage4_common import ROOT, digest, experiment, measure, read_config, restore_once, save_json
from fromrealhand.skills import SegmentedReference, SkillContract, execute_reference


def replay(entry, pieces, config, stitched=False):
    first = pieces[0]
    exp = experiment(entry['geometry'], first)
    rows, errors = [], []
    expected = np.concatenate([p['expected_post_states'] for p in pieces])
    audit_index = [0]
    try:
        restore_once(exp.env, first['initial_snapshot'])

        def read():
            row = measure(exp, audit_index[0])
            audit_index[0] = len(exp.audit)
            return row

        def check(step, row):
            actual = np.r_[exp.env.sim.data.qpos, exp.env.sim.data.qvel]
            errors.append(float(np.max(np.abs(actual-expected[step]))))
            rows.append(row)

        if stitched:
            actions = np.concatenate([p['actions'] for p in pieces])
            provider = SegmentedReference(actions, entry['segments'], config['action_dim'])
            contracts = execute_reference(exp.env, provider, config, read, check)
        else:
            provider = first['actions']
            contract = SkillContract(first['metadata']['skill'], config, read())
            if contract.status != 'failed':
                for step, action in enumerate(provider):
                    exp.env.step(action)
                    row = read()
                    contract.update(row)
                    check(step, row)
                    if contract.status in ('failed', 'timeout'):
                        break
            contracts = [dict(skill=contract.skill, status=contract.status, reason=contract.reason,
                              steps=contract.steps)]
        max_error = max(errors or [float('inf')])
        complete = len(rows) == len(provider)
        passed = complete and max_error <= config['replay_state_tolerance'] and all(
            c['status'] == 'success' for c in contracts)
        return dict(trajectory=entry['trajectory'], mode='stitched' if stitched else first['metadata']['skill'],
            passed=bool(passed), complete=complete, steps=len(rows), expected_steps=len(provider),
            initialization_count=1, state_writes_during_execution=0,
            max_qpos_qvel_replay_error=max_error if errors else None, contracts=contracts,
            max_hand_scene_penetration_m=max([r['scene_penetration_m'] for r in rows] or [0.]),
            final_bottom_m=rows[-1]['bottom_m'] if rows else None,
            final_goal_distance_m=rows[-1]['target_distance_m'] if rows else None)
    finally:
        exp.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/processed/stage4_skills_v1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.dataset/'validation.json'
    if output.exists():
        raise FileExistsError('Refusing to overwrite validation evidence')
    manifest = read_config(args.dataset/'manifest.json')
    if not manifest.get('completed'):
        raise ValueError('Skill export is incomplete')
    results = []
    for entry in manifest['trajectories']:
        if digest(entry['geometry']) != entry['geometry_sha256']:
            raise ValueError('Geometry changed')
        pieces = []
        for segment in entry['segments']:
            path = args.dataset/segment['artifact']
            if digest(path) != segment['sha256']:
                raise ValueError('Segment artifact changed')
            piece = pickle.loads(path.read_bytes())
            pieces.append(piece)
            result = replay(entry, [piece], manifest['config'])
            results.append(result)
            print(json.dumps(result), flush=True)
        result = replay(entry, pieces, manifest['config'], stitched=True)
        results.append(result)
        print(json.dumps(result), flush=True)
    report = dict(schema_version=1, manifest_sha256=digest(args.dataset/'manifest.json'),
        passed=all(r['passed'] for r in results), results=results,
        independent_skill_policies_trained=False, human_boundary_review_complete=False,
        scope='Recorded start-state replay only; no unseen-state or hardware claim')
    save_json(output, report)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
