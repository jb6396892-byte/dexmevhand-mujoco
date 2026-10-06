#!/usr/bin/env python3
"""Actual Qt clicks, edited target coordinates, model execution and safety cases."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
gui=Path('/media/smgbro/shared/lora/language/gui-runtime'); env=dict(os.environ)
for key in ('LD_PRELOAD','PYTHONHOME','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH'): env.pop(key,None)
env.update(PYTHONPATH=str(gui/'packages')+':'+str(ROOT/'src'),
    LD_LIBRARY_PATH=str(gui/'packages/PySide6/Qt/lib')+':'+str(gui),
    QT_QPA_PLATFORM='xcb',QT_IM_MODULE='ibus',PYTHONNOUSERSITE='1')
cases={
    'first-manual':['--scene','first','--seed','10','--target-world','-.02','-.06','.16','--count','4'],
    'second-manual':['--scene','second','--seed','11','--target-world','.03','-.04','.18','--count','2'],
    'empty-lift':['--scene','first','--seed','12','--count','0','--goal','lift'],
    'stop':['--scene','second','--seed','13','--cancel-step','100'],
    'locked':['--locked'],
    'instruction-rejected':['--instruction','将杯子放到我手上']}
results={}
for name,extra in cases.items():
    command=[str(ROOT/'data/runtime/stage6-study-venv/bin/python'),str(ROOT/'scripts/148_check_tabletop_qt.py'),
        '--random-mode','--checkpoint','/media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt',
        '--output',str(a.output/name)]+extra
    with (a.output/(name+'.log')).open('w') as log:
        subprocess.run(command,env=env,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    row=json.loads((a.output/name/'summary.json').read_text()); results[name]=row
    print(json.dumps(dict(case=name,status=row['report']['status'],reason=row['report']['reason'],
        unique_frames=row['unique_frames'],gui_responsive=row['gui_responsive'])),flush=True)
quality={name:results[name]['task_passed'] and results[name].get('target_matches_ui',False)
    and results[name].get('count_matches_ui',False) and results[name]['unique_frames']>20
    for name in ('first-manual','second-manual','empty-lift')}
quality.update(stop=results['stop']['report']['reason']=='user_stop' and results['stop']['cancellation_latency_s']<3,
    locked=results['locked']['report']['status']=='locked',
    instruction_rejected=results['instruction-rejected']['report']['status']=='rejected',
    gui=all(r['gui_responsive'] and not r['surviving_workers'] for r in results.values()))
(a.output/'quality.json').write_text(json.dumps(quality,indent=2)+'\n')
raise SystemExit(0 if all(quality.values()) else 1)
