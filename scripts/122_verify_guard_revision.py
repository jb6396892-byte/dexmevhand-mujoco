#!/usr/bin/env python3
"""Freeze a guard-only revision, replay old outputs, test fresh language, verify physics."""
import argparse
import datetime
import importlib
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.language_planner.contracts import canonical, compact, digest, read, write
from fromrealhand.language_planner.sft import load_rows, verify_model
from fromrealhand.language_planner.refinement import adapter_hashes, system_passes, verify_refinement
from fromrealhand.language_planner.guard_revision import evaluate_system, semantic_guard, verify_revision

SOURCES = ('candidate-validation', 'candidate-heldout', 'candidate-regression')


def prepare(study):
    source = study.parent/'study_v4'
    cfg = verify_refinement(ROOT, source)
    phrases = read(ROOT/'configs/stage6-guard2-heldout.json')
    rows = []
    for goal, texts in phrases.items():
        for index, text in enumerate(texts):
            for scene in ('first', 'second'):
                rows.append(dict(id='guard2-%s-%02d-%s' % (goal, index, scene), instruction=text,
                    scene=scene, response=canonical(goal, scene), provenance='new synthetic language, not new physics'))
    previous = set()
    for path in (source/'dataset').glob('*.jsonl'):
        previous.update(r['instruction'] for r in load_rows(path))
    if previous.intersection(r['instruction'] for r in rows):
        raise ValueError('Fresh test duplicates a previous sentence')
    if len({(r['instruction'], r['scene']) for r in rows}) != len(rows):
        raise ValueError('Duplicate fresh test input')
    study.mkdir(exist_ok=False)
    dataset = study/'dataset'; dataset.mkdir()
    regression, predictions = [], []
    for name in SOURCES:
        report = read(source/name/'report.json')
        for c in report['cases']:
            regression.append(dict(id=c['id'], instruction=c['instruction'], scene=c['scene'], response=c['expected']))
        predictions += read(source/name/'predictions.json')
    for split, values in (('heldout', rows), ('regression', regression)):
        (dataset/(split+'.jsonl')).write_text(''.join(compact(r)+'\n' for r in values), encoding='utf-8')
    write(dataset/'regression-predictions.json', predictions)
    shutil.copyfile(str(source/'dataset/feasibility.json'), str(dataset/'feasibility.json'))
    shutil.copytree(str(source/'candidate/adapter'), str(study/'candidate/adapter'))
    paths = [ROOT/'configs/stage6-guard2-heldout.json', Path(__file__).resolve(),
             ROOT/'scripts/123_run_guarded_language.py',
             ROOT/'src/fromrealhand/language_planner/guard_revision.py']
    write(dataset/'protocol.json', dict(sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},
        data_sha256={p.name:digest(p) for p in dataset.iterdir() if p.is_file()},
        counts=dict(heldout=len(rows), regression=len(regression)), heldout_for_selection=False,
        acceptance=cfg['acceptance'], scope='Frozen weights; lexical repair; fresh synthetic language in two known scenes'))
    write(study/'protocol.json', dict(config=cfg, created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        dataset_protocol_sha256=digest(dataset/'protocol.json'), source_study=str(source),
        source_predictions_sha256={name+'/predictions.json':digest(source/name/'predictions.json') for name in SOURCES},
        change='Guard-only grammar repair. Retired v4 test is regression; no new gradient updates.'))
    print('PREPARED', len(rows), 'new test;', len(regression), 'regression', flush=True)


def regression(study, cfg):
    output = study/'candidate-regression'; output.mkdir(exist_ok=False)
    predictions = read(study/'dataset/regression-predictions.json')
    result = evaluate_system(load_rows(study/'dataset/regression.jsonl'), predictions,
        read(ROOT/'configs/skill_plan.schema.json'), read(study/'dataset/feasibility.json'))
    write(output/'report.json', result)
    write(output/'predictions.json', predictions)
    write(output/'protocol.json', dict(producer='frozen_model_prediction_replay', adapter_sha256=adapter_hashes(study/'candidate/adapter'),
        dataset_protocol_sha256=digest(study/'dataset/protocol.json'), study_protocol_sha256=digest(study/'protocol.json')))
    print('REGRESSION', result['metrics'], flush=True)
    if not system_passes(result['metrics'], cfg['acceptance']):
        raise ValueError('Regression gate failed; do not open fresh test')


def accept(study, cfg, capture):
    from hierarchy_common import SkillRegistry, run_plan, verify_delivery
    verify_model(study.parent/'model')
    reports = {}
    for split in ('heldout', 'regression'):
        folder = study/('candidate-'+split)
        receipt = read(folder/'protocol.json')
        if (receipt['adapter_sha256'] != adapter_hashes(study/'candidate/adapter')
                or receipt['dataset_protocol_sha256'] != digest(study/'dataset/protocol.json')
                or receipt['study_protocol_sha256'] != digest(study/'protocol.json')):
            raise ValueError('Prediction receipt mismatch')
        if split=='heldout' and (receipt['producer']!='lora_model'
                or receipt['model_source_sha256']!=digest(study.parent/'model/source.json')):
            raise ValueError('Fresh test must use actual model generation')
        reports[split] = evaluate_system(load_rows(study/'dataset'/(split+'.jsonl')),
            read(folder/'predictions.json'), read(ROOT/'configs/skill_plan.schema.json'),
            read(study/'dataset/feasibility.json'))
    output = study/'acceptance'; output.mkdir(exist_ok=False)
    registry = SkillRegistry.load(ROOT/'configs/skill_registry.yaml'); verify_delivery(registry)
    plans = {(c['scene'],c['parsed']['plan']['goal']):c['parsed']['plan']
             for c in reports['heldout']['cases'] if c['semantic_correct'] and c['gate']['accepted']}
    physical = []
    for (scene,goal), plan in sorted(plans.items()):
        result = run_plan(registry, plan, output/(scene+'-'+goal), capture=capture and (scene,goal)==('second','transport'))
        p = result.get('physics', {})
        passed = (result['reason']=='user_stop' and result['steps']==0 and not result['simulation_created']) if goal=='stop' else (
            result['status']=='success' and result['reason']=='plan_completed' and p.get('state_writes_during_execution')==0
            and p.get('initialization_count')==1 and p.get('max_penetration_m',1.)<=.001
            and p.get('max_state_replay_error',1.)<=1e-7)
        physical.append(dict(scene=scene,goal=goal,passed=bool(passed),report=result))
        print('PHYSICAL',scene,goal,passed,flush=True)
    negative = []
    for instruction, goal in [('将杯子放到我手上','lift'),('手先到杯子边上去','lift'),
                              ('将杯子握在我的手中','grasp'),('不要抓起杯子','lift')]:
        guard = semantic_guard(compact(canonical(goal,'first')),instruction,'first',
            read(ROOT/'configs/skill_plan.schema.json'),read(study/'dataset/feasibility.json'))
        negative.append(dict(instruction=instruction,passed=not guard['accepted'],gate=guard,steps=0,simulation_created=False))
    for name, report in reports.items(): write(output/(name+'.json'), report)
    write(output/'physical.json', physical); write(output/'negative_guards.json', negative)
    language = all(system_passes(r['metrics'],cfg['acceptance']) for r in reports.values())
    physics = len(physical)==10 and all(p['passed'] for p in physical)
    negatives = all(p['passed'] for p in negative)
    summary = dict(model_acceptance_passed=bool(language and physics and negatives), language_system_passed=language,
        framework_physics_passed=physics,negative_guards_passed=negatives,physical_passed=sum(p['passed'] for p in physical),
        unique_plans=len(physical),physical_motion_plans=sum(p['goal']!='stop' for p in physical),
        language_metrics=reports['heldout']['metrics'],regression_metrics=reports['regression']['metrics'],
        adapter_sha256=adapter_hashes(study/'candidate/adapter'),model_source_sha256=digest(study.parent/'model/source.json'),
        dataset_protocol_sha256=digest(study/'dataset/protocol.json'),study_protocol_sha256=digest(study/'protocol.json'),
        predictions_sha256=digest(study/'candidate-heldout/predictions.json'),
        scope='Guarded finite Chinese commands, two nominal scenes, explicit --execute only',
        low_level_backend='verified expert references; NOT DAPG network',
        unrestricted_language_safety_proven=False,independent_physics_generalization=False)
    write(output/'summary.json',summary); print('ACCEPTANCE',summary,flush=True)
    if not summary['model_acceptance_passed']: raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','regression','heldout','accept','verify'])
    parser.add_argument('--root',type=Path,default=Path('/media/smgbro/shared/lora'))
    parser.add_argument('--capture',action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if Path('/media/smgbro/shared') not in root.parents: raise ValueError('Use shared storage')
    study = root/'language/study_v4_guard2'
    if args.command=='prepare': prepare(study); return
    cfg = verify_revision(ROOT,study)
    if args.command=='regression': regression(study,cfg)
    elif args.command=='heldout':
        if not system_passes(read(study/'candidate-regression/report.json')['metrics'],cfg['acceptance']):
            raise ValueError('Regression not passed')
        base = importlib.import_module('111_language_study')
        base.DATASET = study/'dataset'; base.evaluate = evaluate_system
        for adapter,name in ((study.parent/'study_v3/formal/adapter','previous-heldout'),(study/'candidate/adapter','candidate-heldout')):
            base.score(study,cfg,'heldout',adapter,name)
    elif args.command=='accept': accept(study,cfg,args.capture)
    else: print('Guard revision verified')


if __name__=='__main__': main()
