#!/usr/bin/env python3
"""Merge hash-verified admitted datasets without counting shared originals twice."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np


def same_data(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_data(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(same_data(a, b) for a, b in zip(left, right))
    return np.array_equal(left, right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admissions', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    demos, sources = {}, []
    for path in args.admissions:
        manifest = json.loads(path.read_text())
        raw = Path(manifest['demo']).read_bytes()
        if not manifest.get('training_ready') or hashlib.sha256(raw).hexdigest() != manifest['demo_sha256']:
            raise ValueError('Admission/hash mismatch: '+str(path))
        for key, demo in pickle.loads(raw).items():
            if key in demos:
                if not same_data(demos[key], demo):
                    raise ValueError('Conflicting duplicate trajectory: '+key)
            else:
                demos[key] = demo
        sources.append(dict(admission=str(path.resolve()), demo_sha256=manifest['demo_sha256']))
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output/'demonstrations.pkl'
    with target.open('xb') as stream:
        pickle.dump(demos, stream)
    result = dict(training_ready=True, sources=sources, trajectory_count=len(demos),
                  demo=str(target.resolve()), demo_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  scope='Union of admitted datasets; source manifests define real-video diversity')
    (args.output/'admission.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
