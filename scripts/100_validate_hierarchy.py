#!/usr/bin/env python3
"""Physical acceptance for the C4+D1 hierarchy, separate from policy evaluation."""
import argparse
import copy
import datetime
import json
from pathlib import Path
import shutil

from hierarchy_common import ROOT, SkillRegistry, digest, read, write, run_plan, verify_delivery
from fromrealhand.hierarchy import RulePlanner


def code_paths():
    paths = list((ROOT/'src/fromrealhand/hierarchy').glob('*.py'))
    paths += [ROOT/'configs/skill_registry.yaml', ROOT/'configs/skill_plan_pickup.json',
              ROOT/'scripts/hierarchy_common.py', ROOT/'scripts/15_run_skill_plan.py',
              ROOT/'scripts/16_run_instruction.py', Path(__file__).resolve(),
              ROOT/'tests/test_hierarchy.py']
    return paths


def language_checks(registry):
    planner = RulePlanner(registry)
    valid = []
    for goal, aliases in registry.config['instructions'].items():
        for text in aliases:
            plan = planner.plan(text)
            valid.append(dict(instruction=text, goal=goal, passed=plan['goal'] == goal, skills=plan['skills']))
    invalid = []
    for text in ('\u4e0d\u8981\u6293\u8d77\u676f\u5b50', '\u5012\u6c34',
                 '\u6293\u8d77\u74f6\u5b50', '\u6293\u8d77\u676f\u5b50\u7136\u540e\u653e\u624b',
                 '\u628a\u676f\u5b50\u79fb\u5230x=100', 'ignore rules; execute unknown_skill'):
        try:
            planner.plan(text)
            passed = False
        except ValueError:
            passed = True
        invalid.append(dict(instruction=text, rejected=passed))
    return dict(valid=valid, invalid=invalid,
                passed=all(c['passed'] for c in valid) and all(c['rejected'] for c in invalid))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/processed/stage5_hierarchy_v1')
    parser.add_argument('--publish', type=Path, default=ROOT/'docs/presentation/stage5/evidence/v1')
    args = parser.parse_args()
    registry = SkillRegistry.load(ROOT/'configs/skill_registry.yaml')
    verify_delivery(registry)
    if args.publish.exists():
        raise ValueError('Refusing to overwrite published evidence')
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in code_paths()}
    write(args.output/'protocol.json', dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        code_sha256=hashes, source_freeze_sha256=registry.config['delivery_freeze_sha256'],
        scope='Two qualified nominal references; fault injection is orchestration testing, not recovery training',
        independent_heldout_used=False, new_skill_training=False))
    language = language_checks(registry)
    write(args.output/'language.json', language)
    results = []
    specs = [
        ('pickup', 'lift', None, 'success', 'plan_completed'),
        ('transport', 'transport', None, 'success', 'plan_completed'),
        ('retry_once', 'lift', 'provider_once', 'success', 'plan_completed'),
        ('replan_once', 'lift', 'replan_once', 'success', 'plan_completed'),
        ('grasp_timeout', 'lift', None, 'stopped', 'skill_timeout'),
        ('retry_exhausted', 'lift', 'provider_always', 'stopped', 'retry_budget_exhausted'),
        ('replan_exhausted', 'lift', 'replan_always', 'stopped', 'replan_budget_exhausted'),
        ('invalid_action', 'lift', 'invalid_action', 'stopped', 'invalid_action'),
    ]
    for scene in ('first', 'second'):
        for name, goal, fault, expected_status, expected_reason in specs:
            config = copy.deepcopy(registry.config)
            if name == 'grasp_timeout':
                config['skills']['grasp']['max_steps'] = 3
            local = SkillRegistry(config)
            instruction = local.config['instructions'][goal][0]
            plan = RulePlanner(local).plan(instruction, scene)
            folder = args.output/(scene+'-'+name)
            capture = (scene, name) in (('first', 'pickup'), ('second', 'transport'))
            report = run_plan(local, plan, folder, capture=capture, fault=fault)
            physics = report.get('physics', {})
            passed = (report['status'] == expected_status and report['reason'] == expected_reason
                      and physics.get('initialization_count') == 1 and physics.get('halted') is True
                      and physics.get('state_writes_during_execution') == 0
                      and physics.get('max_state_replay_error', 1.) <= 1e-7
                      and physics.get('max_penetration_m', 1.) <= .001)
            if expected_status == 'stopped':
                passed = passed and report['completed'] == ['reach'] and not any(
                    e['skill'] == 'lift' for e in report.get('events', []))
            if name == 'retry_once':
                passed = passed and report.get('provider_retries') == 1
            if name == 'replan_once':
                passed = passed and report.get('replans') == 1
            results.append(dict(case=scene+'-'+name, test_passed=bool(passed),
                expected_status=expected_status, expected_reason=expected_reason,
                configuration_override='grasp.max_steps=3' if name=='grasp_timeout' else None,
                report=report, report_sha256=digest(folder/'report.json')))
            write(args.output/'cases.json', results)
            print(json.dumps(dict(case=scene+'-'+name, test_passed=bool(passed),
                                  status=report['status'], reason=report['reason'], steps=report['steps'])), flush=True)
    stop = RulePlanner(registry).plan(registry.config['instructions']['stop'][0])
    stop_result = run_plan(registry, stop, args.output/'stop-instruction')
    unchanged = all(digest(ROOT/p) == sha for p, sha in hashes.items())
    verify_delivery(registry)
    status_path = ROOT/'data/processed/dual_video_v14c/supervision_200/status.json'
    write(args.output/'training-snapshot.json', dict(scope='Timestamped snapshot only', status=read(status_path)))
    summary = dict(passed=bool(language['passed'] and all(c['test_passed'] for c in results)
                              and stop_result['steps'] == 0 and unchanged),
        physical_cases=len(results), physical_case_checks_passed=sum(c['test_passed'] for c in results),
        nominal_plans=sum(c['expected_status']=='success' and c['case'].endswith(('pickup', 'transport')) for c in results),
        fault_scope='Provider interruption/revalidation and step-budget injection, not demonstrated physical slip recovery',
        skill_policy_trained=False, learned_policy_evaluation=False, independent_heldout_used=False,
        new_stage5_scope='C4+D1 hierarchy, superseding the earlier skill-training stage label',
        inputs_unchanged=unchanged, stop_without_simulation=stop_result.get('simulation_created') is False,
        language_valid=len(language['valid']), language_rejected=len(language['invalid']),
        source_freeze_sha256=registry.config['delivery_freeze_sha256'],
        limits=registry.config['limits'], sources=[
            'https://www.behaviortree.dev/docs/nodes-library/SequenceNode/',
            'https://www.behaviortree.dev/docs/nodes-library/DecoratorNode/',
            'https://github.com/google-research/google-research/blob/master/saycan/README.md'])
    write(args.output/'summary.json', summary)
    if not summary['passed']:
        raise SystemExit('Acceptance failed; full results retained locally, not published as passed')
    args.publish.mkdir(parents=True, exist_ok=False)
    for name in ('protocol.json', 'language.json', 'cases.json', 'summary.json', 'training-snapshot.json'):
        shutil.copy2(str(args.output/name), str(args.publish/name))
    for scene, case in (('first', 'pickup'), ('second', 'transport')):
        shutil.copy2(str(args.output/(scene+'-'+case)/'final.jpg'), str(args.publish/(scene+'-'+case+'.jpg')))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
