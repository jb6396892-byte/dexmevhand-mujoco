#!/usr/bin/env python3
"""Gate actual physics demonstrations on stable lift and independent action replay."""
import argparse
import hashlib
import json
import pickle
from importlib import import_module
from pathlib import Path

import numpy as np

GraspExperiment = import_module('24_search_physical_grasp').GraspExperiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', default='data/processed/seq_dexycb_001/repair_v2/geometry.npz')
    parser.add_argument('--candidate', default='data/processed/seq_dexycb_001/physical_grasp_v2/verification.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--demo', type=Path, required=True)
    args = parser.parse_args()
    if args.demo.exists():
        raise FileExistsError(args.demo)
    args.output.mkdir(parents=True, exist_ok=False)
    params = json.loads(Path(args.candidate).read_text())['best']['params']
    params.update(relocate=True, steps=1000)
    experiment = GraspExperiment(args.geometry)
    demos, reports, replays, holdouts = {}, [], [], []
    passed = True
    for seed in range(6):
        report, actions = experiment.run(params, args.output/'seed_0' if seed == 0 else None, seed=seed)
        demonstration = experiment.last_demo
        replay, _ = experiment.run(params, args.output/'replay_0' if seed == 0 else None,
                                   seed=seed, saved_actions=actions)
        error = float(np.max(np.abs(demonstration['observations']-experiment.last_demo['observations'])))
        replay['max_observation_replay_error'] = error
        passed = passed and report['passed'] and replay['passed'] and error < 1e-8
        demos['physics_seed_%d' % seed] = demonstration
        reports.append(report)
        replays.append(replay)
        print(json.dumps(dict(seed=seed, physics=report, replay=replay)), flush=True)
        (args.output/'progress.json').write_text(json.dumps(dict(reports=reports, replays=replays), indent=2)+'\n')
    for seed in range(6, 11):
        report, _ = experiment.run(params, seed=seed)
        holdouts.append(report)
        passed = passed and report['passed']
        print(json.dumps(dict(holdout=report)), flush=True)
    result = dict(training_ready=bool(passed), params=params, reports=reports, replays=replays,
                  holdouts=holdouts, scope='DexYCB-derived near-grasp curriculum; object XY perturbations +/-2 mm',
                  geometry=str(Path(args.geometry).resolve()), demo=str(args.demo.resolve()),
                  source_candidate=str(Path(args.candidate).resolve()))
    if passed:
        args.demo.parent.mkdir(parents=True, exist_ok=True)
        with args.demo.open('xb') as f:
            pickle.dump(demos, f)
        result['demo_sha256'] = hashlib.sha256(args.demo.read_bytes()).hexdigest()
    (args.output/'admission.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(training_ready=bool(passed), demo=str(args.demo)), indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
