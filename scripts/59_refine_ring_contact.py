#!/usr/bin/env python3
"""Search ring reference corrections without changing video targets or physics."""
import argparse
import json
from importlib import import_module
from pathlib import Path

import numpy as np

transport = import_module('55_optimize_transport')
from check_codex_budget import latest_usage


def rank(report):
    fractions = report['tail_finger_contact_fraction']
    penalty = 100. if not report['surface_physics_passed'] else 0.
    penalty += 1000 * max(0., report['final_distance_m'] - .02)
    penalty += 2000 * max(0., report['tail_mean_tip_error_m'] - .015)
    penalty += 2000 * max(0., max(report['tail_finger_tip_error_m'].values()) - .025)
    distance = report.get('final_ring_surface_distance_m', report['tail_finger_tip_error_m']['ring'])
    return penalty + 20 * max(0., .8-fractions['ring']) + 1000*distance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=float, nargs='+', default=[.12, .06, .03])
    p.add_argument('--groups', nargs='+', default=['RFJ3', 'RFJ2', 'RFJ1,RFJ0'])
    p.add_argument('--max-trials', type=int, default=25)
    p.add_argument('--bound', type=float, default=.3)
    p.add_argument('--proposals', type=Path, help='Static IK proposals; all require new dynamic validation')
    p.add_argument('--late-correction', action='store_true', help='Keep original grasp, transition after lifting')
    p.add_argument('--quota-stop', type=float, default=80.)
    args = p.parse_args()
    if not np.isfinite(args.bound) or args.bound <= 0 or args.max_trials < 1:
        p.error('Positive finite correction bound and positive trial count required')
    source = json.loads(args.candidate.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    b = source['best']
    if 'ring_target_fraction' in b:
        e = import_module('60_refine_contact_targets').ContactExperiment(source['geometry'])
        e.target_fraction = b['ring_target_fraction']
        e.target_depth = b['ring_target_depth_m']
        e.ring_weight = b['ring_feedback_weight']
        e.isolate_ring = b.get('isolate_ring_feedback', False)
        if args.late_correction:
            e.initial_correction = np.asarray(b['joint_correction']).copy()
        elif b.get('initial_joint_correction') is not None:
            e.initial_correction = np.asarray(b['initial_joint_correction'])
    else:
        e = transport.TransportExperiment(source['geometry'])
    transport.configure(e, b)
    e.transport_kp, e.transport_ki = b['transport_kp'], b['transport_ki']
    kwargs = dict(scale=b['time_scale'], close=b['close'], gain=b['cartesian_gain'])
    groups = [[int(e.model.jnt_qposadr[e.model.joint_name2id(n)]) for n in group.split(',')]
              for group in args.groups]
    best_q, best_score, results = e.correction.copy(), float('inf'), []

    def evaluate(q):
        nonlocal best_q, best_score
        e.correction = q.copy()
        r, _ = e.run_surface(**kwargs)
        r.update(search_score=rank(r), trial=len(results), quota=latest_usage())
        results.append(r)
        (args.output/'search.json').write_text(json.dumps(results, indent=2)+'\n')
        if r['search_score'] < best_score:
            best_q, best_score = q.copy(), r['search_score']
            # This is a search candidate, never an admission certificate.
            (args.output/'candidate.json').write_text(json.dumps(dict(
                geometry=source['geometry'], best=r, training_ready=False,
                source_candidate=str(args.candidate.resolve())), indent=2)+'\n')
        print(json.dumps({k:r[k] for k in ('trial', 'search_score', 'tail_finger_contact_fraction',
              'tail_finger_tip_error_m', 'surface_physics_passed', 'fidelity_passed', 'final_distance_m')}), flush=True)

    def stop():
        usage = latest_usage()
        return len(results) >= args.max_trials or bool(usage and usage['used_percent'] >= args.quota_stop)

    try:
        if not stop():
            evaluate(best_q)
        if args.proposals:
            center = best_q.copy()
            for proposal in json.loads(args.proposals.read_text())['results']:
                if (not proposal['success'] or proposal['max_penetration_m'] > .001+1e-8
                        or proposal['max_tendon_violation_m'] > .00002+1e-8
                        or max(proposal['tip_errors_m']) > .025+1e-8):
                    continue
                q = np.asarray(proposal['joint_correction'])
                if q.shape != (30,) or not np.isfinite(q).all():
                    raise ValueError('Invalid IK correction')
                for fraction in (.5, 1.):
                    if stop():
                        break
                    evaluate(center+fraction*(q-center))
                if stop():
                    break
        for step in ([] if args.proposals else args.steps):
            for group in groups:
                center = best_q.copy()
                for direction in (1., -1.):
                    if stop():
                        break
                    q = center.copy()
                    q[group] = np.clip(q[group]+direction*step, -args.bound, args.bound)
                    evaluate(q)
                if stop():
                    break
            if stop():
                break
        (args.output/'completion.json').write_text(json.dumps(dict(
            trials=len(results), quota=latest_usage(), training_started=False,
            validation_required='Frozen candidate: saved actions, held-out seeds and half timestep'), indent=2)+'\n')
    finally:
        e.env.close()


if __name__ == '__main__':
    main()
