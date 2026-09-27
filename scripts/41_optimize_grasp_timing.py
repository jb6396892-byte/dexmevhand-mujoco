#!/usr/bin/env python3
"""Compare closure timing and finger feedback using free-object physics."""
import argparse
import hashlib
import itertools
import json
from importlib import import_module
from pathlib import Path

import numpy as np

finger = import_module('37_optimize_finger_reference')
from check_codex_budget import latest_usage


class TimingExperiment(finger.CorrectedExperiment):
    def __init__(self, geometry):
        super().__init__(geometry)
        self.closure_lead = 0.
        self.feedback_weights = [1.] * 5

    def run_video(self, *args, **kwargs):
        kwargs.update(closure_lead=self.closure_lead, feedback_weights=self.feedback_weights)
        return super().run_video(*args, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--leads', type=float, nargs='+', default=[0., .1, .2, -.1])
    p.add_argument('--thumb-weights', type=float, nargs='+', default=[1., 2., 4.])
    p.add_argument('--approach-gains', type=float, nargs='+', default=[0.])
    p.add_argument('--approach-root-gains', type=float, nargs='+', default=[None])
    p.add_argument('--closures', type=float, nargs='+', default=[None])
    p.add_argument('--search-seeds', type=int, nargs='+', default=[0, 2, 5])
    p.add_argument('--quota-stop', type=float, default=85.)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    admission = json.loads(args.candidate.read_text())
    b = admission['best']
    exp = TimingExperiment(admission['geometry'])
    exp.correction = np.asarray(b['joint_correction'])
    kwargs = dict(scale=b['time_scale'], close=b['close'], gain=b['cartesian_gain'])
    search_seeds = sorted(set([0] + args.search_seeds))
    results = []
    stopped = False
    try:
        for lead, weight, approach_gain, root_gain, close in itertools.product(
                args.leads, args.thumb_weights, args.approach_gains, args.approach_root_gains, args.closures):
            quota = latest_usage()
            if results and quota and quota['used_percent'] >= args.quota_stop:
                stopped = True
                break
            exp.closure_lead = lead
            exp.feedback_weights = [weight, 1., 1., 1., 1.]
            exp.approach_gain = approach_gain
            exp.approach_root_gain = root_gain
            kwargs['close'] = b['close'] if close is None else close
            reports = [exp.run_surface(**kwargs, seed=seed)[0] for seed in search_seeds]
            report = reports[0]
            report['search_score'] = max(finger.score(r) for r in reports)
            report['search_seed_metrics'] = [{k: r[k] for k in (
                'seed', 'surface_physics_passed', 'fidelity_passed', 'tail_mean_tip_error_m',
                'max_hand_scene_penetration_m')} for r in reports]
            results.append(report)
            (args.output/'search.json').write_text(json.dumps(results, indent=2)+'\n')
            print(json.dumps({k: report[k] for k in ('close', 'closure_lead', 'feedback_weights', 'approach_gain', 'approach_root_gain', 'search_score',
                  'tail_finger_tip_error_m', 'search_seed_metrics')}), flush=True)
        best = min(results, key=lambda r: r['search_score'])
        exp.closure_lead = best['closure_lead']
        exp.feedback_weights = best['feedback_weights']
        exp.approach_gain = best['approach_gain']
        exp.approach_root_gain = best['approach_root_gain']
        kwargs['close'] = best['close']
        best, actions = exp.run_surface(**kwargs, output=args.output/'best')
        original = exp.last_demo['observations'].copy()
        replay, _ = exp.run_surface(**kwargs, saved_actions=actions, output=args.output/'replay')
        error = float(np.max(np.abs(original-exp.last_demo['observations'])))
        probes = [exp.run_surface(**kwargs, seed=seed, output=args.output/('seed_%d'%seed))[0]
                  for seed in range(1, 10)]
        manifest = json.loads((Path(admission['geometry']).parent/'manifest.json').read_text())
        unchanged = all(hashlib.sha256((finger.surface.ROOT/name).read_bytes()).hexdigest() == digest
                        for name, digest in manifest['protected_files'].items())
        physical = unchanged and error < 1e-8 and all(r['surface_physics_passed'] for r in [best, replay]+probes)
        ready = physical and not manifest['failed_optimizer_frames'] and all(r['fidelity_passed'] for r in [best, replay]+probes)
        result = dict(best=best, replay=replay, holdouts=probes, search_seeds=search_seeds,
                      independent_seeds=[s for s in range(6, 10) if s not in search_seeds],
                      geometry=admission['geometry'], source_candidate=str(args.candidate.resolve()),
                      surface_physics_passed=bool(physical), training_ready=bool(ready),
                      baseline_unchanged=unchanged, replay_observation_error=error,
                      stop_reason='quota_reserve' if stopped else 'search_complete', quota=latest_usage())
        (args.output/'admission.json').write_text(json.dumps(result, indent=2)+'\n')
        print('COMPLETE', json.dumps({k: result[k] for k in ('surface_physics_passed', 'training_ready', 'quota')}), flush=True)
    finally:
        exp.env.close()


if __name__ == '__main__':
    main()
