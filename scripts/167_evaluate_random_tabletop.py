#!/usr/bin/env python3
"""All sampled episodes, including failures, count in the random-tabletop rate."""
import argparse
import hashlib
import json
from pathlib import Path
from hierarchy_common import ROOT, read, write
from fromrealhand.tabletop.random_task import RandomTask

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,default=Path('/media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt'))
p.add_argument('--videos',nargs='+',choices=['first','second'],default=['first','second'])
p.add_argument('--seeds',type=int,nargs='+')
p.add_argument('--split',choices=['development','heldout'],default='development')
p.add_argument('--freeze',type=Path)
p.add_argument('--screenshots',action='store_true')
p.add_argument('--target-world',nargs=3,type=float)
p.add_argument('--count',type=int)
p.add_argument('--cup-xy',type=float,nargs=2)
p.add_argument('--protocol',type=Path,default=ROOT/'configs/tabletop-random-v5.json')
a=p.parse_args(); cfg=read(a.protocol)
seeds=a.seeds if a.seeds is not None else cfg[a.split+'_seeds']
if a.split=='heldout':
    if a.target_world is not None or a.count is not None or a.cup_xy is not None: p.error('Heldout must use the registered random distribution')
    if not a.freeze: p.error('Heldout requires a pre-execution freeze')
    frozen=read(a.freeze)
    if cfg!=frozen['protocol']: p.error('Protocol does not match freeze')
    if seeds!=frozen['seeds'] or a.videos!=frozen['videos']: p.error('Changed heldout schedule')
    for name,digest in frozen['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest: p.error('Frozen source changed: '+name)
    if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=frozen['checkpoint_sha256']: p.error('Checkpoint changed')
elif set(seeds)&set(cfg['heldout_seeds']): p.error('Reserved test seed requested as development')
a.output.mkdir(parents=True,exist_ok=False)
rows=[]
for video in a.videos:
    task=RandomTask(ROOT,video,a.checkpoint,a.protocol)
    try:
        for seed in seeds:
            folder=a.output/(video+'-seed-'+str(seed))
            result=task.run(seed,folder,goal=a.target_world,count=a.count,screenshots=a.screenshots,cup_xy=a.cup_xy)
            rows.append(dict(video=video,seed=seed,passed=result['passed'],reason=result['reason'],
                steps=result['steps'],phase=result['phase'],max_penetration_m=result['max_penetration_m'],
                final=result['final'],layout=result['layout'],output=str(folder)))
            write(a.output/'evaluation.json',dict(split=a.split,complete=False,records=rows))
            print(json.dumps({k:rows[-1][k] for k in ('video','seed','passed','reason','steps')},ensure_ascii=False),flush=True)
    finally: task.close()
write(a.output/'evaluation.json',dict(split=a.split,complete=True,records=rows,passed=sum(r['passed'] for r in rows),
    total=len(rows),checkpoint=str(a.checkpoint),seeds=seeds))
