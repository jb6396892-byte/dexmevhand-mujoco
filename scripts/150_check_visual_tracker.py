#!/usr/bin/env python3
"""Cached automatic detection replay for tracking regression, not held-out evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
from fromrealhand.perception.tracking import CupTracker

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); tracker=CupTracker(a.run/'mug_model.npz'); rows=[]
for directory in sorted((a.run/'observations').iterdir()):
    old=json.loads((directory/'estimate.json').read_text())
    calib=json.loads((directory/'camera.json').read_text())
    depth=np.load(str(directory/'depth.npy'))
    k,twc=np.asarray(calib['K']),np.asarray(calib['T_world_camera'])
    box=old['boxes'][0]['box_xyxy']
    try:
        _,result=tracker.estimate(depth,k,twc,box)
        rows.append(dict(frame=directory.name,old_accepted=old['accepted'],accepted=result['accepted'],
                         fitness=result['fitness'],raw_fitness=result.get('raw_fitness'),
                         rmse_m=result['rmse_m'],retained_fraction=result.get('retained_fraction')))
    except ValueError as error:
        rows.append(dict(frame=directory.name,old_accepted=old['accepted'],accepted=False,reason=str(error)))
negative=[]
for name,instance,badbox in [('zero_initial_depth',CupTracker(a.run/'mug_model.npz'),box),
                           ('zero_tracking_depth',tracker,box),
                           ('empty_roi',tracker,[-100,-100,-90,-90])]:
    try:
        _,result=instance.estimate(np.zeros_like(depth) if name!='empty_roi' else depth,k,twc,badbox)
        negative.append(dict(name=name,rejected=not result['accepted']))
    except ValueError as error: negative.append(dict(name=name,rejected=True,reason=str(error)))
report=dict(development_replay=True,not_independent_test=True,frames=rows,negative=negative,
            all_accepted=all(r['accepted'] for r in rows),all_negative_rejected=all(n['rejected'] for n in negative),
            boxes_source='cached actual Grounding DINO outputs; not redrawn boxes')
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(frames=len(rows),all_accepted=report['all_accepted'],negative=negative)))
raise SystemExit(0 if report['all_accepted'] and report['all_negative_rejected'] else 1)
