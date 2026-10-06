#!/usr/bin/env python3
"""Physical corner regressions, separate from the unseen random test denominator."""
import argparse
from itertools import product
from pathlib import Path
from hierarchy_common import ROOT,read,write
from fromrealhand.tabletop.random_task import RandomTask

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--protocol',type=Path,default=ROOT/'configs/tabletop-random-v6.json')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); cfg=read(a.protocol); a.output.mkdir(parents=True,exist_ok=False)
checkpoint=Path('/media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt')
rows=[]
for video in ('first','second'):
    task=RandomTask(ROOT,video,checkpoint,a.protocol)
    try:
        for n,((x,y),which) in enumerate(product(product(*zip(cfg['cup_xy_min_m'],cfg['cup_xy_max_m'])),('min','max'))):
            target=cfg['goal_'+which+'_m']; folder=a.output/(video+'-corner-'+str(n))
            r=task.run(30,folder,cup_xy=[x,y],goal=target,count=4)
            rows.append(dict(video=video,seed=30,corner=n,passed=r['passed'],reason=r['reason'],
                final=r['final'],max_penetration_m=r['max_penetration_m'],layout=r['layout'],output=str(folder)))
            write(a.output/'evaluation.json',dict(split='boundary',complete=False,records=rows))
            print(video,n,r['passed'],r['reason'],flush=True)
    finally: task.close()
write(a.output/'evaluation.json',dict(split='boundary',complete=True,records=rows,
    passed=sum(r['passed'] for r in rows),total=len(rows)))
