#!/usr/bin/env python3
"""Check that the selected saved actions also lift at half the physics step."""
import argparse
import json
import pickle
from pathlib import Path
from importlib import import_module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    args = p.parse_args()
    output = args.result/'timestep_check.json'
    if output.exists():
        raise FileExistsError(output)
    admission = json.loads((args.result/'admission.json').read_text())
    best = admission['best']
    with (args.result/'best/diagnostic_rollout.pkl').open('rb') as f:
        actions = pickle.load(f)['video_faithful']['actions']
    exp = import_module('33_optimize_surface_grasp').SurfaceExperiment(admission['geometry'])
    try:
        original = float(exp.model.opt.timestep)
        exp.model.opt.timestep = original/2
        exp.env.model_timestep = original/2
        replay, _ = exp.run_surface(best['time_scale'], best['close'], best['cartesian_gain'], saved_actions=actions)
        closed, _ = exp.run_surface(best['time_scale'], best['close'], best['cartesian_gain'])
        result = dict(original_timestep_s=original, refined_timestep_s=original/2,
                      saved_action_replay=replay, closed_loop=closed,
                      passed=bool(replay['surface_physics_passed'] and closed['surface_physics_passed']))
        output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    finally:
        exp.env.close()


if __name__ == '__main__':
    main()
