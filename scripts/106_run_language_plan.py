#!/usr/bin/env python3
"""Generate a guarded plan. Default is inspection only, not simulation execution."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.language_planner.contracts import digest, guarded_response, read, write
from fromrealhand.language_planner.sft import generate, load_model, verify_model, verify_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('instruction')
    parser.add_argument('--scene', choices=['first', 'second'], default='first')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--acceptance-report', type=Path)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    dataset = ROOT/'data/processed/stage6_language_v1/dataset'
    verify_protocol(ROOT, dataset)
    verify_model(args.model)
    if args.render and not args.execute:
        raise ValueError('--render requires --execute')
    if args.execute:
        if args.acceptance_report is None:
            raise ValueError('Execution requires the completed model/physics acceptance report from script 107')
        acceptance = read(args.acceptance_report)
        actual_adapter = {p.name: digest(p) for p in args.adapter.iterdir() if p.is_file()}
        if not (acceptance.get('model_acceptance_passed') and acceptance.get('adapter_sha256') == actual_adapter
                and acceptance.get('model_source_sha256') == digest(args.model/'source.json')
                and acceptance.get('dataset_protocol_sha256') == digest(dataset/'protocol.json')):
            raise ValueError('The adapter does not have matching language and nominal-physics acceptance')
    cfg = read(ROOT/'configs/stage6-lora.json')
    model, tokenizer = load_model(args.model, args.adapter)
    raw = generate(model, tokenizer, args.instruction, args.scene,
        (ROOT/'configs/stage6-system-prompt.txt').read_text(encoding='utf-8'), cfg['max_new_tokens'])
    guard = guarded_response(raw, args.scene, read(ROOT/'configs/skill_plan.schema.json'),
                             read(dataset/'feasibility.json'), cfg['feasibility_threshold'])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or ROOT/'data/processed/stage6_language_v1/instructions'/stamp
    output.mkdir(parents=True, exist_ok=False)
    write(output/'generation.json', dict(instruction=args.instruction, raw=raw, guard=guard, executed=False))
    print(json.dumps(guard, ensure_ascii=False, indent=2))
    if not guard['accepted']:
        raise SystemExit(2)
    write(output/'plan.json', guard['response']['plan'])
    if args.execute:
        # Drop the language environment before invoking the legacy physics process.
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env['LD_LIBRARY_PATH'] = '/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu'
        env['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
        env['__NV_PRIME_RENDER_OFFLOAD'] = '1'
        env['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
        command = ['/home/smgbro/miniconda3/envs/dexmv/bin/python', str(ROOT/'scripts/15_run_skill_plan.py'),
                   '--plan', str(output/'plan.json'), '--output', str(output/'physics')]
        if args.render:
            command.append('--render')
        result = subprocess.run(command, cwd=str(ROOT), env=env, check=False)
        write(output/'execution.json', dict(attempted=True, returncode=result.returncode,
            report=str(output/'physics/report.json'), backend='verified_reference', not_dapg_policy=True))
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
