#!/usr/bin/env python3
"""Freeze a development-selected policy before any reserved test execution."""
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from hierarchy_common import ROOT, read, write

SOURCES=[
    'scripts/153_test_contact_tracking.py', 'scripts/162_evaluate_tabletop_policy.py',
    'src/fromrealhand/tabletop/learned_control.py', 'src/fromrealhand/tabletop/training_inputs.py',
    'src/fromrealhand/tabletop/structured_control.py',
    'src/fromrealhand/tabletop/contact_control.py', 'src/fromrealhand/tabletop/control.py',
    'src/fromrealhand/tabletop/control_scene.py', 'src/fromrealhand/tabletop/scene.py',
    'src/fromrealhand/tabletop/functional_reference.py', 'src/fromrealhand/tabletop/vision_client.py',
    'src/fromrealhand/perception/tracking.py', 'src/fromrealhand/perception/association.py',
    'scripts/142_tabletop_vision_worker.py', 'configs/tabletop-dual-v3-profiles.json',
    'configs/tabletop-control-candidate.json', 'configs/tabletop-dual-v3-protocol.json']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--development',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fresh-protocol',type=Path)
    a=p.parse_args(); dev=read(a.development)
    if a.output.exists(): raise ValueError('Never overwrite a frozen selection')
    if (dev['split']!='development' or not dev['complete'] or dev['passed']!=dev['total']
            or {r['video'] for r in dev['records']}!={'first','second'}
            or Path(dev['checkpoint']).resolve()!=a.checkpoint.resolve()):
        raise ValueError('Candidate must pass both videos in closed-loop development')
    protocol=read(ROOT/'configs/tabletop-dual-v3-protocol.json')
    seeds=protocol['reserved_final_test_seeds']
    if a.fresh_protocol:
        fresh=read(a.fresh_protocol); seeds=fresh['test_seeds']
        if set(seeds)&set(protocol['reserved_final_test_seeds']+protocol['development_train_seeds']+protocol['development_validation_seeds']+fresh.get('previous_test_seeds_now_regression_only',[])):
            raise ValueError('Fresh test must be disjoint from previously exposed seeds')
    write(a.output,dict(created_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        development=str(a.development),development_sha256=hashlib.sha256(a.development.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},
        test_seeds=seeds,videos=protocol['videos'],limits=protocol['limits'],
        fresh_protocol=str(a.fresh_protocol) if a.fresh_protocol else None,
        acceptance='All requested skills, unchanged physics gates and actual post-stop goal error <=20mm',
        no_tuning_on_test=True,scope='Local in-distribution layout generalization; same camera, mug and reference family'))
    print(a.output)


if __name__=='__main__': main()
