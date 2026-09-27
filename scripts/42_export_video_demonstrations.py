#!/usr/bin/env python3
"""Export full-video demonstrations only after replay and held-out physics gates."""
import argparse
import hashlib
import json
import pickle
from importlib import import_module
from pathlib import Path

import numpy as np

finger = import_module('37_optimize_finger_reference')
from check_codex_budget import latest_usage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--demo', type=Path, required=True)
    p.add_argument('--quota-stop', type=float, default=85.)
    args = p.parse_args()
    source = json.loads(args.candidate.read_text())
    step_check = json.loads((args.candidate.parent/'timestep_check.json').read_text())
    if not source.get('training_ready') or not step_check['passed'] or not step_check.get('fidelity_passed'):
        raise ValueError('Candidate must pass physics, video fidelity and timestep checks')
    if args.demo.exists():
        raise FileExistsError(args.demo)
    args.output.mkdir(parents=True, exist_ok=False)
    b = source['best']
    exp = finger.CorrectedExperiment(source['geometry'])
    exp.correction = np.asarray(b['joint_correction'])
    exp.closure_lead = b.get('closure_lead', 0.)
    exp.feedback_weights = b.get('feedback_weights')
    exp.approach_gain = b.get('approach_gain', 0.)
    exp.approach_root_gain = b.get('approach_root_gain')
    kwargs = dict(scale=b['time_scale'], close=b['close'], gain=b['cartesian_gain'])
    demos, reports, replays, holdouts = {}, [], [], []
    try:
        for seed in list(range(6)) + list(range(10, 20)):
            quota = latest_usage()
            if quota and quota['used_percent'] >= args.quota_stop:
                (args.output/'interrupted.json').write_text(json.dumps(dict(
                    training_ready=False, stop_reason='quota_reserve', quota=quota), indent=2)+'\n')
                return 1
            report, actions = exp.run_surface(**kwargs, seed=seed)
            if seed < 6:
                demo = exp.last_demo
                replay, _ = exp.run_surface(**kwargs, seed=seed, saved_actions=actions)
                replay['max_observation_replay_error'] = float(np.max(np.abs(
                    demo['observations']-exp.last_demo['observations'])))
                demos['video_seed_%d'%seed] = demo
                reports.append(report)
                replays.append(replay)
            else:
                holdouts.append(report)
            progress = dict(reports=reports, replays=replays, holdouts=holdouts)
            (args.output/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
            print(json.dumps({k: report[k] for k in ('seed', 'surface_physics_passed', 'fidelity_passed',
                  'mean_tip_error_m', 'tail_mean_tip_error_m', 'max_hand_scene_penetration_m')}), flush=True)
        manifest = json.loads((Path(source['geometry']).parent/'manifest.json').read_text())
        unchanged = all(hashlib.sha256((finger.surface.ROOT/name).read_bytes()).hexdigest() == digest
                        for name, digest in manifest['protected_files'].items())
        passed = (unchanged and not manifest['failed_optimizer_frames']
                  and all(r['surface_physics_passed'] and r['fidelity_passed'] for r in reports+replays+holdouts)
                  and all(r['max_observation_replay_error'] < 1e-8 for r in replays))
        result = dict(training_ready=bool(passed), reports=reports, replays=replays, holdouts=holdouts,
                      independent_seeds=list(range(10, 20)), timestep_check=step_check,
                      geometry=source['geometry'], demo=str(args.demo.resolve()),
                      baseline_unchanged=unchanged, source_candidate=str(args.candidate.resolve()),
                      source_candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                      horizon=len(demos['video_seed_0']['actions']), quota=latest_usage(),
                      scope='Full DexYCB video trajectory; six demo initial states; independent XY +/-2mm seeds 10-19; native collision meshes with 0.2mm contact margin')
        if passed:
            args.demo.parent.mkdir(parents=True, exist_ok=True)
            with args.demo.open('xb') as stream:
                pickle.dump(demos, stream)
            result['demo_sha256'] = hashlib.sha256(args.demo.read_bytes()).hexdigest()
        (args.output/'admission.json').write_text(json.dumps(result, indent=2)+'\n')
        print('COMPLETE', json.dumps(dict(training_ready=bool(passed), demo=str(args.demo))), flush=True)
        return 0 if passed else 1
    finally:
        exp.env.close()


if __name__ == '__main__':
    raise SystemExit(main())
