#!/usr/bin/env python3
"""Freeze a search candidate and audit held-out resets, replay and half timestep."""
import argparse
import hashlib
import json
from pathlib import Path
from importlib import import_module
import numpy as np

finger = import_module('37_optimize_finger_reference')
from check_codex_budget import latest_usage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--parameters', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 10, 11, 12, 13, 14])
    p.add_argument('--quota-stop', type=float, default=85.)
    args = p.parse_args()
    if args.seeds[0] != 0 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        p.error('Seeds must start with zero and contain unique nonnegative integers')
    raw = args.parameters.read_bytes()
    b = json.loads(raw)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'frozen_parameters.json').write_text(json.dumps(b, indent=2)+'\n')
    e = finger.CorrectedExperiment(args.geometry)
    e.correction = np.asarray(b['joint_correction'])
    e.closure_lead = b.get('closure_lead', 0.)
    e.feedback_weights = b.get('feedback_weights')
    e.approach_gain = b.get('approach_gain', 0.)
    e.approach_root_gain = b.get('approach_root_gain')
    kwargs = dict(scale=b['time_scale'], close=b['close'], gain=b['cartesian_gain'])
    reports, replay_error, replay = [], None, None
    try:
        for seed in args.seeds:
            quota = latest_usage()
            if quota and quota['used_percent'] >= args.quota_stop:
                break
            folder = args.output/('best' if seed == 0 else 'seed_%d'%seed)
            report, actions = e.run_surface(**kwargs, seed=seed, output=folder)
            reports.append(report)
            if seed == 0:
                nominal_actions = actions
                observations = e.last_demo['observations'].copy()
                replay, _ = e.run_surface(**kwargs, seed=seed, saved_actions=actions)
                replay_error = float(np.max(np.abs(observations-e.last_demo['observations'])))
            (args.output/'progress.json').write_text(json.dumps(reports, indent=2)+'\n')
            print(json.dumps({k:report[k] for k in ('seed', 'hold_s', 'final_distance_m', 'surface_physics_passed', 'fidelity_passed')}), flush=True)
        complete = len(reports) == len(args.seeds) and replay is not None
        refined = []
        quota = latest_usage()
        if complete and (quota is None or quota['used_percent'] < args.quota_stop):
            step = float(e.model.opt.timestep)
            e.model.opt.timestep = step/2
            e.env.model_timestep = step/2
            refined = [e.run_surface(**kwargs, saved_actions=nominal_actions)[0], e.run_surface(**kwargs)[0]]
        all_reports = reports+([replay] if replay else [])+refined
        manifest = json.loads((args.geometry.parent/'manifest.json').read_text())
        unchanged = all(hashlib.sha256((finger.surface.ROOT/name).read_bytes()).hexdigest() == digest
                        for name, digest in manifest['protected_files'].items())
        physics = bool(complete and len(refined) == 2 and replay_error < 1e-8 and unchanged
                       and all(r['surface_physics_passed'] for r in all_reports))
        result = dict(surface_physics_passed=physics,
                      training_ready=bool(physics and not manifest['failed_optimizer_frames'] and all(r['fidelity_passed'] for r in all_reports)),
                      best=reports[0] if reports else None, reports=reports, half_timestep=refined, replay=replay,
                      replay_observation_error=replay_error, baseline_unchanged=unchanged,
                      source_parameters_sha256=hashlib.sha256(raw).hexdigest(), geometry=str(args.geometry.resolve()),
                      completed=complete and len(refined) == 2, independent_seeds=[s for s in args.seeds if s != 0], quota=latest_usage())
        (args.output/'admission.json').write_text(json.dumps(result, indent=2)+'\n')
        print('COMPLETE', json.dumps({k:result[k] for k in ('surface_physics_passed', 'training_ready', 'completed')}), flush=True)
    finally:
        e.env.close()


if __name__ == '__main__':
    main()
