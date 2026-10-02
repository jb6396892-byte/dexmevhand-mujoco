#!/usr/bin/env python3
"""Publish a small, hash-linked review snapshot without copying training data."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,default=ROOT/'data/processed/stage4_boundary_review_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/presentation/stage4/evidence/boundary-review')
    args = parser.parse_args()
    report = read(args.review/'review.json')
    freeze = read(args.review/'freeze.json')
    if not report.get('completed'):
        raise ValueError('Boundary review is incomplete')
    for name, expected in freeze['protected_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Review inputs changed: '+name)
    args.output.mkdir(parents=True,exist_ok=False)
    for result in report['reports']:
        replay = result['physical_replay']
        image = ROOT/replay['image']
        if digest(image) != replay['image_sha256']:
            raise ValueError('Review image changed')
        shutil.copyfile(str(image),str(args.output/image.name))
        replay['published_image'] = image.name
    report['generated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['raw_review_sha256'] = digest(args.review/'review.json')
    report['freeze_sha256'] = digest(args.review/'freeze.json')
    write(args.output/'review.json',report)
    write(args.output/'freeze.json',freeze)
    run = ROOT/'data/processed/dual_video_v14c'
    status = read(run/'supervision_200/status.json')
    # The immutable trainer writes non-atomically; obtain one complete log snapshot.
    rows = read(run/'long_training/iterations.json')[:status['iterations']]
    failures = []
    for row in rows:
        for case in row['reports']:
            if case['task_pass'] and case['strict_pass']:
                continue
            result = case['report']
            failures.append(dict(iteration=row['iteration'],video=case['video'],case=case['case'],
                task_pass=case['task_pass'],strict_pass=case['strict_pass'],lift_pass=case['lift_pass'],
                labels=case['failures_strict'],max_penetration_m=result['max_hand_scene_penetration_m'],
                goal_distance_m=result['final_distance_m'],peak_contact=result['scene_contact_peaks'][0]))
    write(args.output/'training-progress.json',dict(status=status,failures=failures,
        scope='Timestamped training-sample snapshot, not independent evaluation'))
    print(json.dumps(dict(iteration=status['iterations'],failures=len(failures),
                          handoff_warnings=sum(r['handoff_warning_count'] for r in report['reports']))))


if __name__ == '__main__': main()
