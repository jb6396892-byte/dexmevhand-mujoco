#!/usr/bin/env python3
"""Continuous navigation/grasp/carry/place/return development and execution."""
import argparse
import json
from pathlib import Path
from hierarchy_common import ROOT
from fromrealhand.whole_table.task import NavigationTask, make_layout, brief

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--config', type=Path, default=ROOT/'configs/tabletop-placement-v1.json')
p.add_argument('--seed', type=int, default=6101)
p.add_argument('--count', type=int, default=0)
p.add_argument('--video', choices=['first','second'], default='first')
p.add_argument('--goal', nargs=3, type=float, default=[.15,.05,.20])
p.add_argument('--cup-xy', nargs=2, type=float, default=[0.,0.])
p.add_argument('--direct', action='store_true', help='Development only: one candidate, no preview selection')
p.add_argument('--entry', type=int, default=0)
p.add_argument('--yaw', type=float, default=0)
a = p.parse_args()
t = NavigationTask(ROOT, '/media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt', a.config)
try:
    layout = make_layout(a.seed,t.config,a.goal,a.count,a.cup_xy)
    if a.direct:
        result = t._candidate(a.video,layout,a.output,'place',t.config,entry_frame=a.entry,yaw_deg=a.yaw)
    else:
        result = t.run(layout,a.output,preferred=a.video,stop_skill='place')
    print(json.dumps(brief({k:result.get(k) for k in ('passed','reason','placement','planning_attempts')}),indent=2))
finally:
    t.close()
