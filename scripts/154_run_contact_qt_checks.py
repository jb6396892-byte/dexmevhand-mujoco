#!/usr/bin/env python3
"""Sequential actual desktop checks using the existing isolated Qt runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
gui=Path('/media/smgbro/shared/lora/language/gui-runtime')
env=dict(os.environ)
for key in ('LD_PRELOAD','PYTHONHOME','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH'): env.pop(key,None)
env.update(PYTHONPATH=str(gui/'packages')+':'+str(ROOT/'src'),
    LD_LIBRARY_PATH=str(gui/'packages/PySide6/Qt/lib')+':'+str(gui),
    QT_QPA_PLATFORM='xcb',QT_IM_MODULE='ibus',PYTHONNOUSERSITE='1')
cases={
    'transport':['--scene','first','--goal','transport'],
    'stop':['--scene','first','--cancel-step','100'],
    'locked':['--locked'],
    'second-rejected':['--scene','second'],
    'instruction-rejected':['--instruction','将杯子放到我手上']}
results={}
for name,args in cases.items():
    command=[str(ROOT/'data/runtime/stage6-study-venv/bin/python'),str(ROOT/'scripts/148_check_tabletop_qt.py'),
        '--output',str(a.output/name)]+args
    with (a.output/(name+'.log')).open('w') as log:
        subprocess.run(command,env=env,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1100)
    result=json.loads((a.output/name/'summary.json').read_text()); results[name]=result
    print(json.dumps(dict(case=name,status=result['report']['status'],reason=result['report']['reason'],
        steps=result['report']['steps'],gui_responsive=result['gui_responsive']),ensure_ascii=False),flush=True)
quality=dict(transport=results['transport']['task_passed'],
    stop=results['stop']['report']['reason']=='user_stop' and results['stop']['cancellation_latency_s']<3,
    locked=results['locked']['report']['status']=='locked',
    second_rejected=results['second-rejected']['report']['steps']==0 and not results['second-rejected']['task_passed'],
    instruction_rejected=results['instruction-rejected']['report']['status']=='rejected',
    gui=all(r['gui_responsive'] and not r['surviving_workers'] for r in results.values()))
(a.output/'quality.json').write_text(json.dumps(quality,indent=2)+'\n')
raise SystemExit(0 if all(quality.values()) else 1)
