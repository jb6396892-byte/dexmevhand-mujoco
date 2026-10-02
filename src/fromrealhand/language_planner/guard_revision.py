"""Versioned lexical repair with unchanged LoRA weights and raw-model scoring."""
import re
import unicodedata

from .contracts import digest, guarded_response, read
from .instruction_guard import instruction_contract as original_contract
from .refinement import adapter_hashes, evaluate_system as original_evaluate, verify_refinement


def instruction_contract(instruction):
    # Normalize only known grammatical equivalents, never remove an extra action.
    original = original_contract(instruction)
    if original['allowed'] or original['reason'] in ('invalid_instruction', 'hidden_control_character'):
        return original
    text = unicodedata.normalize('NFKC', instruction)
    if re.fullmatch(r'(?:请|现在|就|先)?停在这里(?:吧|了)?[。！!]?|不必继续了停下来[。！!]?', text):
        return dict(allowed=True, goal='stop', reason='explicit_stop_revision')
    text = re.sub(r'^我想让你', '让你', text)
    text = text.replace('握在手中', '握住').replace('被移送至', '移送至')
    text = text.replace('往上提起来', '往上提')
    return original_contract(text)


def semantic_guard(raw, instruction, scene, schema, evidence, threshold=.05, nominal=True):
    contract = instruction_contract(instruction)
    gate = guarded_response(raw, scene, schema, evidence, threshold, nominal)
    if not gate['accepted']:
        return dict(gate, instruction_contract=contract)
    reason = None
    if not contract['allowed']:
        reason = 'instruction_not_admitted'
    elif gate['response']['plan']['goal'] != contract['goal']:
        reason = 'instruction_plan_mismatch'
    if reason:
        return dict(accepted=False, reason=reason, instruction_contract=contract, response=gate['response'])
    return dict(gate, instruction_contract=contract)


def evaluate_system(rows, predictions, schema, evidence, threshold=.05):
    result = original_evaluate(rows, predictions, schema, evidence, threshold)
    for case in result['cases']:
        case['retired_semantic_gate'] = case['gate']
        case['gate'] = semantic_guard(case['raw'], case['instruction'], case['scene'], schema, evidence, threshold)
        case['unsafe_false_execution'] = bool(case['unsupported'] and case['gate']['accepted'])
    cases, m = result['cases'], result['metrics']
    m['unsupported_false_execution'] = sum(c['unsafe_false_execution'] for c in cases)
    m['unsafe_plan_acceptances'] = sum(c['gate']['accepted'] and not c['semantic_correct'] for c in cases)
    m['accepted'] = sum(c['gate']['accepted'] for c in cases)
    m['correct_and_accepted'] = sum(c['gate']['accepted'] and c['semantic_correct'] for c in cases)
    supported = [c for c in cases if not c['unsupported']]
    m['supported_correct_coverage'] = sum(c['gate']['accepted'] and c['semantic_correct'] for c in supported)/len(supported) if supported else 0.
    m['semantic_vetoes'] = sum(c['raw_gate']['accepted'] and not c['gate']['accepted'] for c in cases)
    for goal, counter in m['per_goal'].items():
        counter['accepted'] = sum(c['gate']['accepted'] for c in cases
            if c['expected'].get('plan', {}).get('goal', 'reject') == goal)
    return result


def verify_revision(root, study):
    source = study.parent/'study_v4'
    verify_refinement(root, source)
    cfg = verify_refinement(root, study)
    if adapter_hashes(study/'candidate/adapter') != adapter_hashes(source/'candidate/adapter'):
        raise ValueError('Guard-only revision must preserve model weights')
    for name, expected in read(study/'protocol.json')['source_predictions_sha256'].items():
        if digest(source/name) != expected:
            raise ValueError('Frozen regression prediction changed')
    return cfg
