#!/usr/bin/env python3
"""Publish compact measured evidence; never turn a failed gate into a success."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from hierarchy_common import ROOT,write

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs',type=Path,default=Path('/media/smgbro/shared/visual_grasp/adroit-navigation-v1'))
p.add_argument('--output',type=Path,default=ROOT/'docs/presentation/adroit_navigation')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
read=lambda path:json.loads(path.read_text())
def compact(value):
    if isinstance(value,dict):
        return {k:compact(v) for k,v in value.items() if k not in ('trace','hand_envelope_relative_m')}
    if isinstance(value,list):return [compact(v) for v in value]
    return value

result=dict(scope='development components, not whole-table independent grasp evaluation',
    experiment_root=str(a.runs),training_started=False,full_pipeline_accepted=False,
    navigation=compact(read(a.runs/'navigation-final/summary.json')),
    local={},carry={},handoff={},legacy_regression=compact(read(a.runs/'legacy-regression/report.json')))
for video,run in [('first','local-first-b'),('second','local-second-a')]:
    result['local'][video]=compact(read(a.runs/run/'local/report.json'))
for video in ('first','second'):
    result['carry'][video]=compact(read(a.runs/('delivery-'+video)/'carry.json'))
for video,run in [('first','transit-first-c'),('second','transit-second-a')]:
    result['handoff'][video]=read(a.runs/run/'adapter.json')
for src,name in [('navigation-final/detour.png','empty-hand-detour.png'),
                 ('delivery-first/carry.png','first-carry.png'),
                 ('delivery-second/carry.png','second-carry-not-accepted.png')]:
    shutil.copyfile(a.runs/src,a.output/name)
    (a.output/name).chmod(0o644)
paths=list((ROOT/'src/fromrealhand/whole_table').glob('*.py'))
paths += [ROOT/'src/fromrealhand/tabletop/random_task.py',ROOT/'configs/adroit-navigation-v1.json']
paths += [ROOT/'scripts'/name for name in ('178_check_adroit_navigation.py','179_view_adroit_navigation.py',
    '180_launch_adroit_navigation.sh','181_check_free_hand_local.py','182_publish_adroit_navigation.py')]
result['source_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
write(a.output/'results.json',result)
print(json.dumps(dict(navigation=result['navigation']['passed'],
    local={k:v['passed'] for k,v in result['local'].items()},
    carry={k:v['passed'] for k,v in result['carry'].items()},
    full_pipeline_accepted=False),indent=2))
