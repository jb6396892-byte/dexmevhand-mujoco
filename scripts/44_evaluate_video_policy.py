#!/usr/bin/env python3
"""Audit a learned policy without applying the demonstration controller."""
import argparse
import hashlib
import json
import pickle
from importlib import import_module
from pathlib import Path

import numpy as np

surface = import_module('33_optimize_surface_grasp')
from fromrealhand.policy_learning import StudentActions, lift_success


class PolicyActions:
    def __init__(self, policy, env, horizon):
        self.policy, self.env, self.horizon = policy, env, horizon

    def __len__(self):
        return self.horizon

    def __getitem__(self, index):
        _, info = self.policy.get_action(self.env._get_observations())
        action = np.asarray(info['evaluation'])
        if action.shape != (30,) or not np.isfinite(action).all():
            raise ValueError('Policy returned an invalid action at step %d' % index)
        return action


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14])
    args = p.parse_args()
    source = json.loads(args.candidate.read_text())
    if not source.get('training_ready'):
        raise ValueError('Expected an admitted video reference')
    args.output.mkdir(parents=True, exist_ok=False)
    with args.policy.open('rb') as stream:
        policy = pickle.load(stream)
    exp = surface.SurfaceExperiment(source['geometry'])
    b = source['best']
    reports = []
    try:
        horizon = int(np.ceil((.5+exp.duration*b['time_scale']+2.)/exp.env.control_timestep))
        actions = (StudentActions(policy, exp, horizon) if isinstance(policy, dict)
                   else PolicyActions(policy, exp.env, horizon))
        for seed in args.seeds:
            report, _ = exp.run_surface(b['time_scale'], 0., 0., seed=seed, saved_actions=actions,
                                         output=args.output/('seed_%d'%seed))
            reports.append(report)
            (args.output/'progress.json').write_text(json.dumps(reports, indent=2)+'\n')
            print(json.dumps({k: report[k] for k in ('seed', 'hold_s', 'max_bottom_m', 'final_distance_m',
                  'surface_physics_passed', 'fidelity_passed', 'max_hand_scene_penetration_m')}), flush=True)
        result = dict(policy=str(args.policy.resolve()), policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                      execution_mode=('Fixed nominal action trajectory plus learned residual; no online expert'
                                      if isinstance(policy, dict) and 'action_reference' in policy else
                                      'Deterministic learned policy actions through env.step; no expert control'),
                      source_candidate=str(args.candidate.resolve()), geometry=source['geometry'], reports=reports,
                      physical_pass_count=sum(r['surface_physics_passed'] for r in reports),
                      lift_and_hold_count=sum(lift_success(r) for r in reports),
                      fidelity_pass_count=sum(r['fidelity_passed'] for r in reports), episode_count=len(reports),
                      all_passed=all(r['surface_physics_passed'] and r['fidelity_passed'] for r in reports))
        (args.output/'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
        print('COMPLETE', json.dumps({k: result[k] for k in ('lift_and_hold_count', 'physical_pass_count', 'fidelity_pass_count', 'episode_count')}), flush=True)
    finally:
        exp.env.close()


if __name__ == '__main__':
    main()
