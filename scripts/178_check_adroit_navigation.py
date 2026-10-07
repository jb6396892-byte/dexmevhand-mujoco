#!/usr/bin/env python3
"""Physical free-hand navigation acceptance, separate from learned grasping."""
import argparse
import json
import numpy as np
from hierarchy_common import ROOT,write
from fromrealhand.whole_table.layout import sample
from fromrealhand.whole_table.scene import object_catalog
from fromrealhand.whole_table.hand_scene import HandScene
from fromrealhand.whole_table.navigation import NavigationRejected,plan
from fromrealhand.whole_table.navigation_runner import navigate,capture

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=__import__('pathlib').Path,required=True)
p.add_argument('--seeds',type=int,nargs='+',default=list(range(6)))
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
cfg=json.loads((ROOT/'configs/adroit-navigation-v1.json').read_text());catalog=object_catalog()
records=[]
for seed in a.seeds:
    scene=HandScene(sample(seed,catalog,cfg,seed%5),cfg)
    row=navigate(scene,[.32,.28,.34]);row['seed']=seed;records.append(row)
    write(a.output/'random-navigation.json',records)
    print('seed',seed,row['passed'],row['endpoint_error_m'],flush=True)
write(a.output/'calibration.json',scene.calibration)
wall=dict(name='test_wall',pos=[0,0,.18],size=[.035,.8,.18])
scene=HandScene(sample(101,catalog,cfg,0),cfg,[wall])
capture(scene,a.output/'start.png')
detour=navigate(scene,[.32,.28,.34]);write(a.output/'detour.json',detour)
capture(scene,a.output/'detour.png')
assert not detour['direct']
print('detour',detour['passed'],len(detour['waypoints']),flush=True)
blocked=HandScene(sample(101,catalog,cfg,0),cfg,[dict(wall,pos=[0,0,.50],size=[.035,.8,.50])])
before=blocked.sim.data.time
try: navigate(blocked,[.32,.28,.34])
except NavigationRejected as error: blocked_result=dict(rejected=True,reason=str(error),no_step=blocked.sim.data.time==before)
else: blocked_result=dict(rejected=False,no_step=False)
invalid=[]
for goal in ([1,0,.3],[0,0,float('nan')],[0,0,.2]):
    before=blocked.sim.data.time
    try: navigate(blocked,goal)
    except NavigationRejected: invalid.append(blocked.sim.data.time==before)
    else: invalid.append(False)
result=dict(scope='actuated free-hand navigation; no grasp',passed=all(r['passed'] for r in records)
    and detour['passed'] and blocked_result['rejected'] and blocked_result['no_step'] and all(invalid),
    random_passed=sum(r['passed'] for r in records),random_total=len(records),
    detour_passed=detour['passed'],detour_waypoints=detour['waypoints'],blocked=blocked_result,
    invalid_rejected=sum(invalid),calibration=scene.calibration,training_started=False)
write(a.output/'summary.json',result);print(json.dumps(result,indent=2),flush=True)
if not result['passed']:raise SystemExit(1)
