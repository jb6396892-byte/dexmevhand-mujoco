#!/usr/bin/env python3
"""Re-execute a saved batch case in a live native MuJoCo window."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from hierarchy_common import ROOT

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--case',type=Path,required=True,help='Case directory containing local/report.json')
p.add_argument('--seconds',type=float,default=0)
p.add_argument('--close-after-run',action='store_true')
a=p.parse_args();case_dir=a.case.resolve()
manifest=json.loads((case_dir.parent/'manifest.json').read_text())
case=next(c for c in manifest['cases'] if c['name']==case_dir.name)
output=Path(tempfile.mkdtemp(prefix='adroit-navigation-grasp-'))/'run'
command=[sys.executable,str(ROOT/'scripts/181_check_free_hand_local.py'),
    '--config',manifest.get('config',str(ROOT/'configs/adroit-navigation-v2.json')),
    '--video',case['video'],'--seed',str(case['seed']),
    '--layout',str(case_dir.parent/(case['name']+'-layout.json')),
    '--shift']+list(map(str,case['shift']))+['--transit','--carry-goal']+list(map(str,case['goal']))+[
    '--output',str(output),'--viewer','--viewer-seconds',str(a.seconds)]
if a.close_after_run:command+=['--close-after-run']
print('Viewer run output: '+str(output),flush=True)
os.execv(sys.executable,command)
