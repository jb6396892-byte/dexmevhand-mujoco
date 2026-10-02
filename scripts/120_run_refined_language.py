#!/usr/bin/env python3
"""Guarded inference; execution requires a matching independent acceptance receipt."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fromrealhand.language_planner.contracts import read, write
from fromrealhand.language_planner.sft import generate, load_model, verify_model
from fromrealhand.language_planner.instruction_guard import instruction_contract, semantic_guard
from fromrealhand.language_planner.refinement import deployment_matches, verify_refinement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('instruction')
    parser.add_argument('--scene',choices=['first','second'],default='first')
    parser.add_argument('--root',type=Path,default=Path('/media/smgbro/shared/lora'))
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--render',action='store_true')
    args = parser.parse_args()
    if args.render and not args.execute:
        parser.error('--render requires --execute')
    study = args.root.resolve()/'language/study_v4'
    cfg = verify_refinement(ROOT,study)
    model_path = study.parent/'model'
    if args.execute:
        receipt = study/'acceptance/summary.json'
        if not receipt.exists() or not deployment_matches(read(receipt),study,model_path):
            parser.error('Candidate has no matching language/physics acceptance; execution is locked')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = study/'instructions'/stamp
    output.mkdir(parents=True,exist_ok=False)
    contract = instruction_contract(args.instruction)
    if not contract['allowed']:
        result = dict(accepted=False,reason='instruction_not_admitted',instruction_contract=contract,
                      model_called=False,simulation_created=False,steps=0)
        write(output/'generation.json',dict(instruction=args.instruction,guard=result,executed=False))
        print(json.dumps(result,ensure_ascii=False,indent=2))
        raise SystemExit(2)
    verify_model(model_path)
    model,tokenizer = load_model(model_path,study/'candidate/adapter')
    raw = generate(model,tokenizer,args.instruction,args.scene,
        (ROOT/'configs/stage6-system-prompt.txt').read_text(encoding='utf-8'),cfg['max_new_tokens'])
    guard = semantic_guard(raw,args.instruction,args.scene,read(ROOT/'configs/skill_plan.schema.json'),
                           read(study/'dataset/feasibility.json'),cfg['feasibility_threshold'])
    write(output/'generation.json',dict(instruction=args.instruction,raw=raw,guard=guard,executed=False,model_called=True))
    print(json.dumps(guard,ensure_ascii=False,indent=2),flush=True)
    if not guard['accepted']:
        raise SystemExit(2)
    write(output/'plan.json',guard['response']['plan'])
    if args.execute:
        env = os.environ.copy(); env.pop('PYTHONPATH',None)
        env['LD_LIBRARY_PATH'] = '/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu'
        env['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
        env['__NV_PRIME_RENDER_OFFLOAD'] = '1'; env['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
        command = ['/home/smgbro/miniconda3/envs/dexmv/bin/python',str(ROOT/'scripts/15_run_skill_plan.py'),
                   '--plan',str(output/'plan.json'),'--output',str(output/'physics')]
        if args.render:
            command.append('--render')
        result = subprocess.run(command,cwd=str(ROOT),env=env,check=False)
        write(output/'execution.json',dict(attempted=True,returncode=result.returncode,
            report=str(output/'physics/report.json'),backend='verified_reference',not_dapg_policy=True))
        raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
