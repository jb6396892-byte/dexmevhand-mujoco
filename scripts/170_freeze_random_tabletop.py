#!/usr/bin/env python3
"""Register the exact random-task controller and untouched test schedule."""
import argparse
import hashlib
from datetime import datetime,timezone
from pathlib import Path
from hierarchy_common import ROOT,read,write

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True); p.add_argument('--development',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,default=Path('/media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt'))
p.add_argument('--protocol',type=Path,default=ROOT/'configs/tabletop-random-v5.json')
a=p.parse_args(); cfg=read(a.protocol); dev=read(a.development)
if a.output.exists(): raise ValueError('Refusing to overwrite freeze')
if not dev['complete'] or dev['split']!='development': raise ValueError('Incomplete development')
for video in ('first','second'):
    rows=[r for r in dev['records'] if r['video']==video]
    if {r['seed'] for r in rows}!=set(cfg['development_seeds']): raise ValueError('Incomplete registered development set')
    if sum(r['passed'] for r in rows)/len(rows)<=cfg['minimum_success_rate']: raise ValueError('Development rate not above 80%')
files=list((ROOT/'src/fromrealhand/tabletop').glob('*.py'))
files.append(a.protocol.resolve())
files.extend(ROOT/name for name in ('scripts/167_evaluate_random_tabletop.py','scripts/168_stream_random_tabletop.py',
    'src/fromrealhand/desktop/random_window.py','configs/tabletop-random-v5.json',
    'configs/tabletop-control-candidate.json','configs/tabletop-dual-v3-profiles.json'))
write(a.output,dict(created_utc=datetime.now(timezone.utc).isoformat(),
    sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
    development=str(a.development),development_sha256=hashlib.sha256(a.development.read_bytes()).hexdigest(),
    videos=['first','second'],seeds=cfg['heldout_seeds'],success_rate_must_exceed=.8,
    protocol=cfg,failures_counted=True,no_test_selection_or_retries=True))
print(a.output)
