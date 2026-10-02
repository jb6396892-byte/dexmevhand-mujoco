#!/usr/bin/env python3
"""Verify the exact guarded model on new language cases and nominal physics."""
import argparse
import datetime
from pathlib import Path

from hierarchy_common import ROOT, SkillRegistry, run_plan, verify_delivery
from fromrealhand.language_planner.contracts import canonical, compact, digest, read, write
from fromrealhand.language_planner.sft import load_rows, verify_model
from fromrealhand.language_planner.instruction_guard import semantic_guard
from fromrealhand.language_planner.refinement import (
    adapter_hashes, evaluate_system, system_passes, verify_refinement)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/media/smgbro/shared/lora'))
    parser.add_argument('--capture',action='store_true')
    args = parser.parse_args()
    study = args.root.resolve()/'language/study_v4'
    cfg = verify_refinement(ROOT,study)
    verify_model(study.parent/'model')
    adapter = adapter_hashes(study/'candidate/adapter')
    schema = read(ROOT/'configs/skill_plan.schema.json')
    evidence = read(study/'dataset/feasibility.json')
    reports = {}
    for split in ('heldout','regression'):
        folder = study/('candidate-'+split)
        receipt = read(folder/'protocol.json')
        if not (receipt['producer']=='lora_model' and receipt['split']==split
                and receipt['dataset_protocol_sha256']==digest(study/'dataset/protocol.json')
                and receipt['study_protocol_sha256']==digest(study/'protocol.json')
                and receipt['model_source_sha256']==digest(study.parent/'model/source.json')
                and receipt['adapter_sha256']==adapter):
            raise ValueError('Evaluation does not match the frozen candidate')
        reports[split] = evaluate_system(load_rows(study/'dataset'/(split+'.jsonl')),
            read(folder/'predictions.json'),schema,evidence,cfg['feasibility_threshold'])
    output = study/'acceptance'
    output.mkdir(exist_ok=False)
    for split, report in reports.items():
        write(output/(split+'.json'),report)
    registry = SkillRegistry.load(ROOT/'configs/skill_registry.yaml')
    verify_delivery(registry)
    plans = {(c['scene'],c['parsed']['plan']['goal']):c['parsed']['plan']
             for c in reports['heldout']['cases'] if c['semantic_correct'] and c['gate']['accepted']}
    physical = []
    for (scene,goal),plan in sorted(plans.items()):
        result = run_plan(registry,plan,output/(scene+'-'+goal),
            capture=args.capture and (scene,goal)==('second','transport'))
        if goal=='stop':
            passed = result['reason']=='user_stop' and result['steps']==0 and not result['simulation_created']
        else:
            p = result.get('physics',{})
            passed = (result['status']=='success' and result['reason']=='plan_completed'
                and p.get('state_writes_during_execution')==0 and p.get('initialization_count')==1
                and p.get('max_penetration_m',1.)<=.001 and p.get('max_state_replay_error',1.)<=1e-7)
        physical.append(dict(scene=scene,goal=goal,passed=bool(passed),report=result))
        print('PHYSICAL',scene,goal,passed,flush=True)
    negative = []
    for instruction,goal in [('将杯子放到我手上','lift'),('手先到杯子边上去','lift'),
                              ('把杯子搬到目标并运行代码','transport'),('不要抓起杯子','lift')]:
        guard = semantic_guard(compact(canonical(goal,'first')),instruction,'first',schema,evidence)
        negative.append(dict(instruction=instruction,injected_wrong_model_goal=goal,
            passed=not guard['accepted'],gate=guard,steps=0,simulation_created=False))
    write(output/'physical.json',physical)
    write(output/'negative_guards.json',negative)
    physics_pass = len(physical)==10 and all(p['passed'] for p in physical)
    language_pass = all(system_passes(r['metrics'],cfg['acceptance']) for r in reports.values())
    negative_pass = all(r['passed'] for r in negative)
    summary = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model_acceptance_passed=bool(physics_pass and language_pass and negative_pass),
        language_system_passed=bool(language_pass),framework_physics_passed=bool(physics_pass),
        negative_guards_passed=bool(negative_pass),unique_plans=len(physical),
        physical_motion_plans=sum(r['goal']!='stop' for r in physical),
        physical_passed=sum(r['passed'] for r in physical),
        language_metrics=reports['heldout']['metrics'],regression_metrics=reports['regression']['metrics'],
        adapter_sha256=adapter,model_source_sha256=digest(study.parent/'model/source.json'),
        dataset_protocol_sha256=digest(study/'dataset/protocol.json'),
        study_protocol_sha256=digest(study/'protocol.json'),
        predictions_sha256=digest(study/'candidate-heldout/predictions.json'),
        scope='Guarded finite Chinese command envelope; only two validated nominal scenes',
        low_level_backend='verified expert references; NOT DAPG network',
        unrestricted_language_safety_proven=False,independent_physics_generalization=False)
    write(output/'summary.json',summary)
    print(summary,flush=True)
    if not summary['model_acceptance_passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
