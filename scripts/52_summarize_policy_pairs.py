#!/usr/bin/env python3
"""Summarize paired residual/nominal evaluations without selecting checkpoints."""
import argparse
import json
from pathlib import Path


def summarize(reports):
    result = dict(count=len(reports))
    for key in ('nominal_actions', 'student'):
        rows = [r[key] for r in reports]
        result[key] = dict(
            physical=sum(r['surface_physics_passed'] for r in rows),
            full=sum(r['surface_physics_passed'] and r['fidelity_passed'] for r in rows),
            full_and_20mm=sum(r['surface_physics_passed'] and r['fidelity_passed'] and r['final_distance_m'] <= .02 for r in rows),
            mean_distance_mm=1000*sum(r['final_distance_m'] for r in rows)/len(rows),
            worst_distance_mm=1000*max(r['final_distance_m'] for r in rows))
    deltas = [1000*(r['student']['final_distance_m']-r['nominal_actions']['final_distance_m']) for r in reports]
    result.update(mean_delta_mm=sum(deltas)/len(deltas), residual_distance_wins=sum(d < -1e-6 for d in deltas))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.evaluation.read_text())
    reports = source['reports']
    if source['completed_cases'] != source['planned_cases'] or any('student' not in r for r in reports):
        raise ValueError('Incomplete or rejected paired evaluation')
    groups = {}
    for name in ('position', 'goal', 'cup_yaw'):
        subset = [r for r in reports if name in r['name']]
        if subset:
            groups[name] = summarize(subset)
    result = dict(overall=summarize(reports), groups=groups, evaluation=str(args.evaluation.resolve()),
                  scope='Paired deterministic synthetic scenarios, not independent real videos or statistical proof of generalization')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
