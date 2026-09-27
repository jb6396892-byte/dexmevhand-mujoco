#!/usr/bin/env python3
"""Reaudit saved actions against hand-object, hand-table and self-contact gates."""
import argparse
import json
import pickle
from pathlib import Path
from importlib import import_module
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.result/'admission.json').read_text())
    b = report['best']
    exp = import_module('37_optimize_finger_reference').CorrectedExperiment(report['geometry'])
    exp.correction = np.asarray(b['joint_correction'])
    try:
        with (args.result/'best/diagnostic_rollout.pkl').open('rb') as f:
            demo = pickle.load(f)['video_faithful']
        audit, _ = exp.run_surface(b['time_scale'], b['close'], b['cartesian_gain'],
                                  saved_actions=demo['actions'], output=args.output)
        error = float(np.max(np.abs(exp.last_demo['observations']-demo['observations'])))
        (args.output/'replay_error.json').write_text(json.dumps(dict(max_observation_error=error))+'\n')
        print(json.dumps({k:audit[k] for k in ('surface_physics_passed','max_hand_scene_penetration_m','initial_hand_scene_penetration_m')}))
    finally:
        exp.env.close()


if __name__ == '__main__':
    main()
