#!/usr/bin/env python3
"""Actual Qt interaction checks using the same navigation task backend as evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--cases',nargs='+',default=['first','second','clutter','stop','locked','rejected'])
p.add_argument('--protocol',type=Path,default=ROOT/'configs/tabletop-navigation-v5.json')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
gui=Path('/media/smgbro/shared/lora/language/gui-runtime');env=dict(os.environ)
for key in ('LD_PRELOAD','PYTHONHOME','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH'):env.pop(key,None)
env.update(PYTHONPATH=str(gui/'packages')+':'+str(ROOT/'src'),
    LD_LIBRARY_PATH=str(gui/'packages/PySide6/Qt/lib')+':'+str(gui),
    QT_QPA_PLATFORM='xcb',QT_IM_MODULE='ibus',PYTHONNOUSERSITE='1')
first=['--scene','first','--seed','4304','--count','2','--cup-xy','.16','.04','--target-world','-.2','.08','.2']
cases=dict(first=first,
    clutter=['--scene','first','--seed','5203','--count','4'],
    second=['--scene','second','--seed','4301','--count','2','--cup-xy','-.2','-.04',
            '--target-world','.2','.08','.2','--speed','.8'],
    stop=first+['--cancel-step','50'],locked=['--locked'],rejected=['--instruction','将杯子放到我手上'])
results={}
for name in a.cases:
    command=[str(ROOT/'data/runtime/stage6-study-venv/bin/python'),str(ROOT/'scripts/148_check_tabletop_qt.py'),
        '--navigation-mode','--protocol',str(a.protocol),'--checkpoint','/media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt',
        '--output',str(a.output/name)]+cases[name]
    print('START '+name,flush=True)
    with (a.output/(name+'.log')).open('w') as log:
        code=subprocess.run(command,env=env,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,timeout=1200).returncode
    path=a.output/name/'summary.json'
    if not path.exists():raise RuntimeError('Qt case failed before report: '+name+' exit='+str(code))
    result=json.loads(path.read_text());results[name]=result
    print(json.dumps(dict(case=name,status=result['report']['status'],reason=result['report']['reason'],
                         frames=result['unique_frames'],gui_responsive=result['gui_responsive'])),flush=True)
quality={}
for name,result in results.items():
    if name in ('first','second','clutter'):
        quality[name]=all(result.get(k,False) for k in ('task_passed','target_matches_ui','count_matches_ui','cup_matches_ui','speed_matches_ui','clearance_matches_ui')) and result['unique_frames']>20
    elif name=='stop':quality[name]=result['report']['reason']=='user_stop' and result['cancellation_latency_s']<3
    elif name=='locked':quality[name]=result['report']['status']=='locked'
    elif name=='rejected':quality[name]=result['report']['status']=='rejected'
quality['gui']=all(r['gui_responsive'] and not r['surviving_workers'] and r['stop_button_visible'] and r['stop_button_in_window'] for r in results.values())
(a.output/'quality.json').write_text(json.dumps(quality,indent=2)+'\n')
print(json.dumps(quality),flush=True)
raise SystemExit(0 if all(quality.values()) else 1)
