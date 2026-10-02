#!/usr/bin/env python3
"""Use dexmv: test JSON-to-physics separately from actual learned language quality."""
import argparse
import datetime
from pathlib import Path

from hierarchy_common import ROOT, SkillRegistry, run_plan, verify_delivery
from fromrealhand.language_planner.contracts import canonical, compact, digest, guarded_response, read, write
from fromrealhand.language_planner.evaluation import evaluate
from fromrealhand.language_planner.sft import load_rows, verify_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/processed/stage6_language_v1/dataset')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-evaluation', type=Path, help='Directory from script 105; absent means fixtures only')
    parser.add_argument('--capture', action='store_true', help='At most one final screenshot')
    args = parser.parse_args()
    verify_protocol(ROOT, args.dataset)
    registry = SkillRegistry.load(ROOT/'configs/skill_registry.yaml')
    verify_delivery(registry)
    schema, evidence = read(ROOT/'configs/skill_plan.schema.json'), read(args.dataset/'feasibility.json')
    threshold = read(ROOT/'configs/stage6-lora.json')['feasibility_threshold']
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {}
    if args.model_evaluation:
        receipt = read(args.model_evaluation/'protocol.json')
        if receipt['dataset_protocol_sha256'] != digest(args.dataset/'protocol.json'):
            raise ValueError('Evaluation used different data')
        if receipt['split'] != 'heldout' or receipt['producer'] not in ('lora_model', 'pretrained_model'):
            raise ValueError('Final acceptance requires a real-model heldout evaluation')
        language = evaluate(load_rows(args.dataset/'heldout.jsonl'),
            read(args.model_evaluation/'predictions.json'), schema, evidence, threshold)
        producer = receipt['producer']
    else:
        rows = load_rows(args.dataset/'validation.jsonl')
        predictions = [dict(id=r['id'], raw=compact(r['response'])) for r in rows]
        language = evaluate(rows, predictions, schema, evidence, threshold)
        producer = 'ground_truth_fixtures_NOT_model_predictions'
    write(args.output/'language.json', dict(language, producer=producer))
    unique_plans = {}
    for case in language['cases']:
        if case['gate']['accepted'] and case['semantic_correct']:
            plan = case['parsed']['plan']
            unique_plans[(plan['scene'], plan['goal'])] = plan
    results = []
    for (scene, goal), plan in sorted(unique_plans.items()):
        report = run_plan(registry, plan, args.output/(scene+'-'+goal),
            capture=args.capture and (scene, goal) == ('second', 'transport'))
        if goal == 'stop':
            passed = report['reason'] == 'user_stop' and report['steps'] == 0 and not report['simulation_created']
        else:
            physics = report.get('physics', {})
            passed = (report['status'] == 'success' and report['reason'] == 'plan_completed'
                and physics.get('state_writes_during_execution') == 0 and physics.get('initialization_count') == 1
                and physics.get('max_penetration_m', 1.) <= .001
                and physics.get('max_state_replay_error', 1.) <= 1e-7)
        results.append(dict(scene=scene, goal=goal, passed=bool(passed), report=report))
    # Explicitly exercise the new low-level margin gate without starting simulation.
    negatives = []
    plan = canonical('transport', 'second')
    raw = compact(plan)
    low = guarded_response(raw, 'second', schema, evidence, threshold=.2)
    unknown = guarded_response(raw, 'second', schema, evidence, nominal=False)
    skipped = canonical('lift', 'second')
    skipped['plan']['skills'] = ['lift']
    illegal = guarded_response(compact(skipped), 'second', schema, evidence)
    for name, result in [('insufficient_margin', low), ('non_nominal_scene', unknown), ('skipped_prerequisite', illegal)]:
        negatives.append(dict(case=name, passed=not result['accepted'], result=result,
                              simulation_created=False, steps=0))
    write(args.output/'physical.json', results)
    write(args.output/'negative_guards.json', negatives)
    complete = len(results) == 10 and all(r['passed'] for r in results) and all(c['passed'] for c in negatives)
    model_passed = bool(args.model_evaluation and producer == 'lora_model'
                        and language['language_acceptance_passed'] and complete)
    summary = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        producer=producer, framework_physics_passed=complete, model_acceptance_passed=model_passed,
        unique_plans_tested=len(results), unique_plans_passed=sum(r['passed'] for r in results),
        physical_motion_plans=sum(r['goal'] != 'stop' for r in results),
        language_metrics=language['metrics'] if args.model_evaluation else None,
        fixture_contract_cases=len(language['cases']) if not args.model_evaluation else None,
        model_trained=producer == 'lora_model', independent_physics_scenes=False,
        low_level_backend='verified_reference; NOT the DAPG checkpoint',
        dataset_protocol_sha256=digest(args.dataset/'protocol.json'),
        adapter_sha256=receipt.get('adapter_sha256'), model_source_sha256=receipt.get('model_source_sha256'),
        source_evaluation_sha256=digest(args.model_evaluation/'protocol.json') if args.model_evaluation else None,
        no_language_training_reason=None if args.model_evaluation else 'User deferred until shared disk is writable',
        code_sha256={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__).resolve(),
            ROOT/'src/fromrealhand/language_planner/contracts.py', ROOT/'src/fromrealhand/language_planner/evaluation.py']})
    write(args.output/'summary.json', summary)
    write(args.output/'training-snapshot.json', read(ROOT/'data/processed/dual_video_v14c/supervision_200/status.json'))
    print(summary)
    if not complete or (args.model_evaluation and not model_passed):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
