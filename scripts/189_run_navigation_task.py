#!/usr/bin/env python3
"""Run candidate planning and one continuous full-table execution."""
import argparse
import json
from pathlib import Path
from hierarchy_common import ROOT
from fromrealhand.whole_table.task import NavigationTask,make_layout,brief

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed',type=int,default=4301)
p.add_argument('--layout',type=Path)
p.add_argument('--preferred',choices=['first','second'],default='first')
p.add_argument('--mode',choices=['auto','fixed'],default='auto')
p.add_argument('--goal',nargs=3,type=float)
p.add_argument('--cup-xy',nargs=2,type=float)
p.add_argument('--count',type=int)
p.add_argument('--config',type=Path,default=ROOT/'configs/tabletop-navigation-v5.json')
p.add_argument('--checkpoint',type=Path,default=Path('/media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt'))
a=p.parse_args();task=NavigationTask(ROOT,a.checkpoint,a.config)
try:
    layout=json.loads(a.layout.read_text()) if a.layout else make_layout(a.seed,task.config,a.goal,a.count,a.cup_xy)
    if a.goal:layout['goal_world_m']=a.goal
    result=task.run(layout,a.output,preferred=a.preferred,mode=a.mode)
    print(json.dumps({k:result.get(k) for k in ('passed','reason','selected_video','planning_attempts','actual_executions','total_wall_s')},indent=2),flush=True)
finally:task.close()
