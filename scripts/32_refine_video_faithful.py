#!/usr/bin/env python3
"""Refine contact-phase joint references using measured object-relative tip errors."""
import argparse
import hashlib
import json
from importlib import import_module
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

control = import_module('30_validate_video_faithful')
from fromrealhand.video_fidelity import FINGERS, TIP_INDICES, object_relative


def rank(report):
    fractions = report['tail_finger_contact_fraction']
    missing = max(0, 4-sum(v >= .8 for v in fractions.values()))
    return (not report['passed'], not report['physics_passed'],
            report['tail_mean_tip_error_m']+.01*missing, report['mean_tip_error_m'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    experiment = control.VideoExperiment(args.geometry)
    candidate = json.loads(args.candidate.read_text())['best']
    kwargs = dict(time_scale=candidate['time_scale'], close=candidate['close'], cartesian_gain=candidate['cartesian_gain'])
    correction = np.zeros(30)
    results = []
    for iteration in range(args.iterations+1):
        report, _ = experiment.run_video(**kwargs, joint_correction=correction)
        report['refinement_iteration'] = iteration
        results.append(report)
        (args.output/'refinement.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps(report), flush=True)
        d, m, g = experiment.env.sim.data, experiment.model, experiment.geometry
        local = object_relative(g['human_joints'][-1, TIP_INDICES], g['object_poses'][-1])
        targets = local @ d.body_xmat[experiment.env.obj_bid].reshape(3, 3).T + d.body_xpos[experiment.env.obj_bid]
        actual = experiment.landmarks.read(d)[TIP_INDICES]
        jac = np.vstack([d.get_site_jacp('S_'+finger+'tip').reshape(3, m.nv)[:, 6:30] for finger in FINGERS])
        error = (targets-actual).ravel()
        # Trust-region correction uses measured geometry; root translation stays on the video path.
        solution = lsq_linear(np.vstack([jac, .01*np.eye(24)]), np.r_[error, np.zeros(24)],
                              bounds=(-.08, .08), tol=1e-8)
        correction[6:] = np.clip(correction[6:]+.5*solution.x, -.3, .3)
    best = min(results, key=rank)
    correction = np.asarray(best['joint_correction'])
    report, actions = experiment.run_video(**kwargs, joint_correction=correction, output=args.output/'best')
    observations = experiment.last_demo['observations'].copy()
    replay, _ = experiment.run_video(**kwargs, joint_correction=correction, saved_actions=actions, output=args.output/'replay')
    replay_error = float(np.max(np.abs(observations-experiment.last_demo['observations'])))
    holdouts = [experiment.run_video(**kwargs, joint_correction=correction, seed=seed)[0] for seed in (1, 2, 3)]
    manifest = json.loads((args.geometry.parent/'manifest.json').read_text())
    unchanged = all(hashlib.sha256((control.ROOT/name).read_bytes()).hexdigest() == value
                    for name, value in manifest['protected_files'].items())
    if not unchanged:
        raise RuntimeError('Protected baseline changed')
    ready = (report['passed'] and replay['passed'] and replay_error < 1e-8
             and all(r['passed'] for r in holdouts) and not manifest['failed_optimizer_frames'])
    result = dict(training_ready=bool(ready), best=report, replay=replay, replay_observation_error=replay_error,
                  holdouts=holdouts, baseline_unchanged=unchanged, geometry=str(args.geometry.resolve()),
                  selected_iteration=best['refinement_iteration'], control=kwargs,
                  refinement='Bounded wrist/finger Jacobian correction from actual rollout; original root translation reference retained',
                  fidelity_gate='whole mean <20mm; tail mean <15mm; every finger <25mm; thumb/index and >=4 fingers contact >=80% of tail')
    (args.output/'admission.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    experiment.env.close()


if __name__ == '__main__':
    main()
