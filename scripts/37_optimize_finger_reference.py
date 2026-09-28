#!/usr/bin/env python3
"""Search bounded per-finger reference corrections with physics-first admission."""
import argparse
import hashlib
import json
from importlib import import_module
from pathlib import Path
import numpy as np

surface = import_module('33_optimize_surface_grasp')
from check_codex_budget import latest_usage


class CorrectedExperiment(surface.SurfaceExperiment):
    def __init__(self, geometry):
        super().__init__(geometry)
        self.correction = np.zeros(30)
        self.closure_lead = 0.
        self.feedback_weights = None
        self.approach_gain = 0.
        self.approach_root_gain = None

    def run_video(self, *args, **kwargs):
        kwargs['joint_correction'] = self.correction
        kwargs.setdefault('closure_lead', self.closure_lead)
        kwargs.setdefault('feedback_weights', self.feedback_weights)
        kwargs.setdefault('approach_gain', self.approach_gain)
        kwargs.setdefault('approach_root_gain', self.approach_root_gain)
        return super().run_video(*args, **kwargs)


def score(r):
    fractions = r['tail_finger_contact_fraction']
    missing = max(0., 4-sum(min(1., v/.8) for v in fractions.values()))
    error = 1000*(r['tail_mean_tip_error_m']+.4*r['mean_tip_error_m'])
    thumb = 1000*max(0., r['tail_finger_tip_error_m']['thumb']-.025)
    penalty = 0. if r['surface_physics_passed'] else 100.
    return penalty + error + .5*thumb + 8*missing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidate', type=Path, default=surface.ROOT/'data/processed/seq_dexycb_001/surface_grasp_v2/admission.json')
    p.add_argument('--max-trials', type=int, default=65)
    p.add_argument('--quota-stop', type=float, default=85.)
    p.add_argument('--initial-correction', type=Path)
    p.add_argument('--steps', type=float, nargs='+', default=[.12, .06, .03])
    p.add_argument('--search-seeds', type=int, nargs='+', default=[0])
    p.add_argument('--groups', nargs='+', help='Joint names or comma-separated coupled joint names; defaults retain the original search')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    admission = json.loads(args.candidate.read_text())
    exp = CorrectedExperiment(admission['geometry'])
    b = admission['best']
    exp.closure_lead = b.get('closure_lead', 0.)
    exp.feedback_weights = b.get('feedback_weights')
    exp.approach_gain = b.get('approach_gain', 0.)
    exp.approach_root_gain = b.get('approach_root_gain')
    kwargs = dict(scale=b['time_scale'], close=b['close'], gain=b['cartesian_gain'])
    groups = [('THJ4',), ('THJ3',), ('THJ2',), ('THJ1',), ('THJ0',),
              ('MFJ2',), ('MFJ1', 'MFJ0'), ('FFJ2',), ('RFJ2',), ('WRJ1',), ('WRJ0',)]
    if args.groups:
        groups = [tuple(group.split(',')) for group in args.groups]
    ids = [[int(exp.model.jnt_qposadr[exp.model.joint_name2id(name)]) for name in group] for group in groups]
    results = []
    best_q = np.zeros(30)
    if args.initial_correction:
        best_q = np.asarray(json.loads(args.initial_correction.read_text())['joint_correction'])
    best_score = float('inf')
    stop_reason = 'max_trials'

    def evaluate(q):
        nonlocal best_q, best_score
        exp.correction = q.copy()
        report, _ = exp.run_surface(**kwargs)
        report['trial'] = len(results)
        probes = [report] + [exp.run_surface(**kwargs, seed=seed)[0]
                             for seed in args.search_seeds if seed != 0]
        report['score'] = max(score(r) for r in probes)
        report['search_seed_scores'] = {str(r['seed']): score(r) for r in probes}
        report['quota'] = latest_usage()
        results.append(report)
        (args.output/'search.json').write_text(json.dumps(results, indent=2)+'\n')
        if report['score'] < best_score:
            best_score, best_q = report['score'], q.copy()
            (args.output/'best_parameters.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({key:report[key] for key in ('trial','score','tail_mean_tip_error_m','tail_finger_tip_error_m','tail_finger_contact_fraction','surface_physics_passed','quota')}), flush=True)

    try:
        evaluate(best_q)
        for step in args.steps:
            for group in ids:
                center = best_q.copy()
                for direction in (1., -1.):
                    quota = latest_usage()
                    if quota and quota['used_percent'] >= args.quota_stop:
                        stop_reason = 'quota_reserve'
                        break
                    if len(results) >= args.max_trials:
                        break
                    q = center.copy()
                    q[group] = np.clip(q[group]+direction*step, -.3, .3)
                    evaluate(q)
                if stop_reason == 'quota_reserve' or len(results) >= args.max_trials:
                    break
            if stop_reason == 'quota_reserve' or len(results) >= args.max_trials:
                break
        else:
            stop_reason = 'coordinate_passes_complete'
        exp.correction = best_q.copy()
        best, actions = exp.run_surface(**kwargs, output=args.output/'best')
        original = exp.last_demo['observations'].copy()
        replay, _ = exp.run_surface(**kwargs, saved_actions=actions, output=args.output/'replay')
        error = float(np.max(np.abs(original-exp.last_demo['observations'])))
        holdouts = [exp.run_surface(**kwargs, seed=seed, output=args.output/('seed_%d'%seed))[0] for seed in range(1,6)]
        manifest = json.loads((Path(admission['geometry']).parent/'manifest.json').read_text())
        unchanged = all(hashlib.sha256((surface.ROOT/name).read_bytes()).hexdigest() == digest for name,digest in manifest['protected_files'].items())
        physical = unchanged and error < 1e-8 and all(r['surface_physics_passed'] for r in [best,replay]+holdouts)
        ready = physical and not manifest['failed_optimizer_frames'] and all(r['fidelity_passed'] for r in [best,replay]+holdouts)
        result = dict(surface_physics_passed=bool(physical), training_ready=bool(ready), best=best,replay=replay,
                      holdouts=holdouts,replay_observation_error=error,baseline_unchanged=unchanged,
                      geometry=admission['geometry'], stop_reason=stop_reason,quota=latest_usage(),
                      trials=len(results), source_candidate=str(args.candidate.resolve()))
        result['search_seeds'] = sorted(set([0]+args.search_seeds))
        result['joint_groups'] = [list(group) for group in groups]
        (args.output/'admission.json').write_text(json.dumps(result,indent=2)+'\n')
        print('COMPLETE',json.dumps({k:result[k] for k in ('surface_physics_passed','training_ready','trials','stop_reason','quota')}),flush=True)
    finally:
        exp.env.close()


if __name__ == '__main__':
    main()
