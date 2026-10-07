#!/usr/bin/env python3
"""Predeclare complete full-table cases, freeze source hashes, then evaluate all."""
import argparse
import hashlib
import json
from pathlib import Path
from hierarchy_common import ROOT,write
from fromrealhand.whole_table.task import NavigationTask,make_layout
from fromrealhand.whole_table.scene import object_catalog

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--start-seed',type=int,default=5401)
p.add_argument('--count',type=int,default=20)
p.add_argument('--config',type=Path,default=ROOT/'configs/tabletop-navigation-v5.json')
p.add_argument('--checkpoint',type=Path,default=Path('/media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt'))
p.add_argument('--label',default='heldout')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
task=NavigationTask(ROOT,a.checkpoint,a.config);catalog=object_catalog()
cases=[dict(seed=s,preferred='first' if i%2==0 else 'second',
            layout=make_layout(s,task.config,count=2+i%3,catalog=catalog))
       for i,s in enumerate(range(a.start_seed,a.start_seed+a.count))]
sources=list((ROOT/'src/fromrealhand/whole_table').glob('*.py'))
sources+=list((ROOT/'src/fromrealhand/tabletop').glob('*.py'))
sources+=list((ROOT/'configs').glob('adroit-navigation-*.json'))+[a.config.resolve(),Path(__file__).resolve()]
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
write(a.output/'manifest.json',dict(cases=cases,label=a.label,created_before_execution=True,source_sha256=hashes,
    config=task.config,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
    denominator=a.count,acceptance='at least 16 of 20 complete physical tasks; no failure removal',
    selection='isolated candidate previews followed by at most one actual execution'))
records=[]
try:
    for case in cases:
        print('START '+str(case['seed']),flush=True)
        try:
            result=task.run(case['layout'],a.output/('seed-'+str(case['seed'])),preferred=case['preferred'])
            c=result.get('carry',{});adapter=result.get('adapter',{})
            row=dict(seed=case['seed'],preferred=case['preferred'],selected=result['selected_video'],
                selected_entry_frame=result['selected_entry_frame'],selected_yaw_deg=result['selected_yaw_deg'],
                passed=bool(result['passed'] and result['actual_executions']==1 and c.get('passed') and c.get('hold_supported_fraction')==1.),
                reason=result['reason'],attempts=result['planning_attempts'],actual_executions=result['actual_executions'],
                cup_error_m=c.get('cup_error_m'),carry_penetration_m=c.get('max_grasp_penetration_m'),
                local_penetration_m=result.get('max_penetration_m'),hold_supported_fraction=c.get('hold_supported_fraction'),
                execution_resets=result['execution_resets'],planning_seconds=result['planning_seconds'],
                total_wall_s=result['total_wall_s'],carry_detour=c.get('direct') is False,
                egress=bool(c.get('egress')),strict_passed=all(r.get('strict_passed',False) for r in
                    [c,adapter.get('navigation',{}),adapter.get('approach',{})]),
                cup_xy=case['layout']['objects'][0]['xy'],goal=case['layout']['goal_world_m'])
        except Exception as error:
            row=dict(seed=case['seed'],passed=False,reason=type(error).__name__+': '+str(error))
        records.append(row);write(a.output/'partial.json',records);print(json.dumps(row),flush=True)
    unchanged=all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
    summary=dict(cases=records,passed=sum(r['passed'] for r in records),total=len(cases),label=a.label,
        success_rate=sum(r['passed'] for r in records)/len(cases),source_unchanged=unchanged,
        reaches_80_percent=sum(r['passed'] for r in records)>=.8*len(cases),
        claim='system with model-based candidate previews; not independent raw-policy trials')
    write(a.output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2),flush=True)
finally:task.close()
