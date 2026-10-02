#!/usr/bin/env python3
"""Freeze paraphrase-family splits and link labels to rule and physical evidence."""
import argparse
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.hierarchy import SkillRegistry, RulePlanner
from fromrealhand.language_planner.contracts import canonical, compact, digest, read, write, validate_response


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    registry = SkillRegistry.load(ROOT/'configs/skill_registry.yaml')
    planner = RulePlanner(registry)
    phrases_path = ROOT/'configs/stage6-language-phrases.json'
    schema_path = ROOT/'configs/skill_plan.schema.json'
    stage5_path = ROOT/'data/processed/stage5_hierarchy_v1/cases.json'
    stage4_path = ROOT/'data/processed/stage4_pipeline_v2/entries/cases.json'
    phrases, schema, cases, entry_cases = map(read, (phrases_path, schema_path, stage5_path, stage4_path))
    for path, expected in read(stage5_path.parent/'protocol.json')['code_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Frozen hierarchy code changed: '+path)
    for case in cases:
        if digest(stage5_path.parent/case['case']/'report.json') != case['report_sha256']:
            raise ValueError('Source execution report changed')
    success = {scene: next(c for c in cases if c['case'] == scene+'-transport') for scene in ('first', 'second')}
    evidence = dict(scope='Frozen nominal references only; score is a margin, not a success probability',
                    penetration_limit_m=.001, scenes={})
    for scene, trajectory in registry.config['scenes'].items():
        evidence['scenes'][scene] = {}
        for skill in registry.config['skills']:
            row = next(c for c in entry_cases if c['trajectory'] == trajectory
                       and c['skills'] == [skill] and c['offset_m'] == [0., 0.])
            if not row['passed'] or not success[scene]['test_passed']:
                raise ValueError('Reference evidence failed')
            if skill not in success[scene]['report']['completed']:
                raise ValueError('Missing successful skill in execution log')
            evidence['scenes'][scene][skill] = dict(skill=skill, passed=True,
                max_penetration_m=row['max_penetration_m'], trials=1,
                margin_score=max(0., 1.-row['max_penetration_m']/.001),
                source_case=scene+'-transport', source_trajectory=trajectory)
    write(output/'feasibility.json', evidence)
    seen, counts = set(), {}
    for split in ('train', 'validation', 'heldout'):
        rows = []
        for goal, bases in phrases[split].items():
            for index, text in enumerate(bases):
                if text in seen:
                    raise ValueError('Paraphrase leakage: '+text)
                seen.add(text)
                wrappers = phrases['training_wrappers'] if split == 'train' else ['{}']
                for scene in ('first', 'second'):
                    response = canonical(goal, scene)
                    if goal != 'reject':
                        label = planner.plan(registry.config['instructions'][goal][0], scene)
                        if response['plan'] != label:
                            raise ValueError('Rule and SFT label disagree')
                    validate_response(response, scene, schema)
                    for variation, wrapper in enumerate(wrappers):
                        rows.append(dict(id='%s-%s-%02d-%s-%d' % (split, goal, index, scene, variation),
                            family='%s-%s-%02d' % (split, goal, index), scene=scene,
                            instruction=wrapper.format(text), response=response,
                            provenance=dict(language='manually authored synthetic paraphrase',
                                label='registered rule plan' if goal != 'reject' else 'unsupported instruction contract',
                                physical_case=scene+'-transport' if goal not in ('reject', 'stop') else None)))
        (output/(split+'.jsonl')).write_text(''.join(compact(row)+'\n' for row in rows), encoding='utf-8')
        counts[split] = dict(rows=len(rows), families=len({r['family'] for r in rows}))
    write(output/'execution_audit.json', [dict(case=c['case'], status=c['report']['status'],
        reason=c['report']['reason'], fault_injection=c['report']['fault_injection'],
        source_sha256=c['report_sha256'], use='success labels' if c['case'].endswith('-transport')
        else 'audit only; injected failures are not new language labels') for c in cases])
    paths = [phrases_path, schema_path, ROOT/'configs/stage6-lora.json',
             ROOT/'configs/stage6-system-prompt.txt', ROOT/'requirements-stage6.txt', stage5_path, stage4_path,
             ROOT/'configs/skill_registry.yaml', Path(__file__).resolve()]
    paths += list((ROOT/'src/fromrealhand/language_planner').glob('*.py'))
    paths += [ROOT/'scripts'/name for name in ('104_train_language_lora.py', '105_evaluate_language.py',
        '106_run_language_plan.py', '107_verify_language_pipeline.py', '103_setup_language_env.sh', 'stage6_python.sh')]
    paths += [output/(split+'.jsonl') for split in ('train', 'validation', 'heldout')]
    paths += [output/'feasibility.json', output/'execution_audit.json']
    protocol = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), counts=counts,
        splits='Base sentences disjoint. Wrapper and scene variants stay together; not a syntactic OOD split.',
        source_videos=2, independent_physics_scenes=False,
        heldout_policy='No training, checkpoint selection, or threshold tuning on heldout',
        acceptance=dict(semantic_accuracy_min=.9, unsupported_false_execution_max=0,
            registered_order_violations_max=0, nominal_physical_success_required=True),
        sha256={str(p.relative_to(ROOT)): digest(p) for p in paths})
    write(output/'protocol.json', protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/processed/stage6_language_v1/dataset')
    build(parser.parse_args().output)
