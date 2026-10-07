#!/usr/bin/env python3
"""Seeded YCB obstacle navigation, forward and reverse without resetting physics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from hierarchy_common import ROOT,write
from fromrealhand.whole_table.layout import sample,rotated_bounds
from fromrealhand.whole_table.scene import object_catalog
from fromrealhand.whole_table.hand_scene import HandScene
from fromrealhand.whole_table.navigation_runner import navigate,capture

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seeds',type=int,nargs='+',default=[4101,4102,4103,4104,4105,4106])
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
base=json.loads((ROOT/'configs/adroit-navigation-v1.json').read_text())
base.update(collision_geometry_only=True,workspace_min_m=[-.65,-.75,.06],workspace_max_m=[.65,.50,.65])
catalog=object_catalog();cases=[]
for index,seed in enumerate(a.seeds):
    layout=sample(seed,catalog,base,2+index%3)
    blocker=max(layout['objects'],key=lambda obj:np.ptp(catalog[obj['name']][:,2]))
    lo,hi=rotated_bounds(catalog[blocker['name']],blocker['yaw_deg'])
    y=float(blocker['xy'][1]+(lo[1]+hi[1])/2)
    z=float(np.clip(hi[2]-lo[2]-.01,.10,.18))
    start=[-.60,y,z];goal=[.60,y,z]
    cases.append(dict(seed=seed,layout=layout,config=dict(base,home_m=start),
        start_m=start,goal_m=goal,goals=[goal,start],selected_blocker=blocker['name'],
        selection='highest mesh extent; crossing its projected center; no outcome-based resampling'))
sources=[Path(__file__).resolve(),ROOT/'configs/adroit-navigation-v1.json']
sources+=list((ROOT/'src/fromrealhand/whole_table').glob('*.py'))
write(a.output/'manifest.json',dict(cases=cases,created_before_physics=True,
    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    acceptance='original empty-hand limits; no relaxed acceleration gate',training_started=False))
records=[]
for index,case in enumerate(cases):
    directory=a.output/('seed-'+str(case['seed']));directory.mkdir()
    write(directory/'case.json',case)
    scene=None;failed=False;initialization_error=None
    try:
        scene=HandScene(case['layout'],case['config'])
        write(directory/'initial_obstacles.json',scene.obstacles())
        initial={name:scene.sim.data.body_xpos[scene.sim.model.body_name2id(name+'_0')].copy()
                 for name in scene.object_vertices}
        if index in (0,len(cases)-1):capture(scene,directory/'start.png')
    except Exception as error:
        failed=True;initial={};initialization_error=type(error).__name__+': '+str(error)
    for direction,goal in zip(('forward','reverse'),case['goals']):
        result=dict(seed=case['seed'],direction=direction,passed=False,goal_m=goal,
                    selected_blocker=case['selected_blocker'],objects=len(case['layout']['objects']))
        if failed:
            result['reason']=initialization_error or 'previous_direction_failed'
        else:
            captured=[False]
            blocker_x=next(x['xy'][0] for x in case['layout']['objects'] if x['name']==case['selected_blocker'])
            def frame(scene,report,row):
                if index in (0,len(cases)-1) and direction=='forward' and not captured[0] and abs(row['center_m'][0]-blocker_x)<.015:
                    capture(scene,directory/'detour.png');captured[0]=True
            try:
                scene.last_navigation={}
                scene.hand_envelope=scene.hand_shapes()-scene.position()
                result.update(navigate(scene,goal,frame))
                if index in (0,len(cases)-1) and direction=='forward':capture(scene,directory/'arrived.png')
            except Exception as error:
                result.update(getattr(scene,'last_navigation',{}))
                result.update(passed=False,reason=type(error).__name__+': '+str(error))
            result['object_displacement_m']={name:float(np.linalg.norm(scene.sim.data.body_xpos[
                scene.sim.model.body_name2id(name+'_0')]-position)) for name,position in initial.items()}
            failed=not result['passed']
        write(directory/(direction+'.json'),result)
        records.append({k:v for k,v in result.items() if k not in ('trace','hand_envelope_relative_m','obstacles')})
        print(json.dumps(records[-1]),flush=True)
    write(a.output/'partial.json',records)
summary=dict(scope='empty-hand seeded point-to-point development test',records=records,
    passed=sum(r['passed'] for r in records),total=len(records),seeds=a.seeds,
    planned_detours=sum(r.get('direct') is False for r in records),
    failed=[dict(seed=r['seed'],direction=r['direction'],reason=r.get('reason','acceptance_failed')) for r in records if not r['passed']],
    endpoint_max_m=max((r.get('endpoint_error_m',0) for r in records),default=0),
    tracking_max_m=max((r.get('max_tracking_error_m',0) for r in records),default=0),
    hand_environment_contacts=sum(r.get('hand_environment_contacts',r.get('stop_audit',{}).get('hand_environment_contacts',0)) for r in records),
    reset_between_directions=False,training_started=False,grasp_executed=False)
write(a.output/'summary.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2),flush=True)
if summary['passed']!=summary['total']:raise SystemExit(1)
