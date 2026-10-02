#!/usr/bin/env python3
"""Compatibility CLI with a verified GPU environment for native MuJoCo windows."""
import argparse
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fromrealhand.desktop.planning import new_run,plan_instruction,validated_plan
from fromrealhand.desktop.runtime import ROOT,LEGACY_PYTHON,physics_environment
from fromrealhand.language_planner.contracts import write

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('instruction')
parser.add_argument('--scene',choices=['first','second'],default='first')
parser.add_argument('--root',type=Path,default=Path('/media/smgbro/shared/lora'))
parser.add_argument('--execute',action='store_true')
parser.add_argument('--render',action='store_true')
parser.add_argument('--verify-render',action='store_true',help=argparse.SUPPRESS)
args = parser.parse_args()
if args.render and not args.execute: parser.error('--render requires --execute')
output = new_run(args.root)
if not plan_instruction(args.root,args.instruction,args.scene,output): raise SystemExit(2)
if args.execute:
    validated_plan(args.root,output)
    command = [LEGACY_PYTHON,str(ROOT/'scripts/15_run_skill_plan.py'),'--plan',str(output/'plan.json'),
               '--output',str(output/'physics')]
    if args.render: command.append('--render')
    if args.verify_render:
        command = [LEGACY_PYTHON,str(ROOT/'scripts/131_verify_native_window.py'),'--plan',str(output/'plan.json'),
                   '--output',str(output/'physics')]
    result = subprocess.run(command,cwd=str(ROOT),env=physics_environment(),check=False)
    write(output/'execution.json',dict(returncode=result.returncode,backend='verified_reference',not_dapg_policy=True))
    raise SystemExit(result.returncode)
