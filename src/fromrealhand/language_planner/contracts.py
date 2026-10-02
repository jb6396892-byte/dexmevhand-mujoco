"""Fail-closed JSON boundary and evidence-based nominal feasibility margins."""
import hashlib
import json
import math
from pathlib import Path

import jsonschema

GOALS = ('reach', 'grasp', 'lift', 'transport', 'stop')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def strict_json(text):
    if not isinstance(text, str) or len(text) > 4096:
        raise ValueError('Invalid model response size')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)


def canonical(goal, scene):
    if goal == 'reject':
        return {'decision': 'reject'}
    if goal not in GOALS or scene not in ('first', 'second'):
        raise ValueError('Unknown goal or scene')
    skills = [] if goal == 'stop' else list(GOALS[:GOALS.index(goal)+1])
    return dict(decision='execute', plan=dict(schema_version=1, scene=scene,
                object='mug', goal=goal, skills=skills))


def validate_response(value, scene, plan_schema):
    envelope = {'oneOf': [
        {'type': 'object', 'required': ['decision'], 'additionalProperties': False,
         'properties': {'decision': {'const': 'reject'}}},
        {'type': 'object', 'required': ['decision', 'plan'], 'additionalProperties': False,
         'properties': {'decision': {'const': 'execute'}, 'plan': plan_schema}}]}
    try:
        jsonschema.Draft7Validator(envelope).validate(value)
    except jsonschema.ValidationError as error:
        raise ValueError('Invalid plan response: '+error.message) from error
    if value['decision'] == 'execute' and value['plan']['scene'] != scene:
        raise ValueError('Model changed the selected scene')
    return value


def feasibility(plan, evidence, threshold=.05, nominal=True):
    """A normalized penetration margin, NOT a calibrated success probability."""
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('Invalid feasibility threshold')
    if plan['goal'] == 'stop':
        return dict(accepted=True, score=1., reason='user_stop', skills=[])
    scene = evidence.get('scenes', {}).get(plan['scene'])
    if not nominal or not scene:
        return dict(accepted=False, score=0., reason='unvalidated_scene', skills=[])
    rows = [scene.get(skill, {}) for skill in plan['skills']]
    scores = [row.get('margin_score', 0.) if row.get('passed') else 0. for row in rows]
    if any(not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1
           for score in scores):
        return dict(accepted=False, score=0., reason='invalid_evidence', skills=rows)
    score = min(scores) if scores else 0.
    if not math.isfinite(score) or not 0 <= score <= 1:
        return dict(accepted=False, score=0., reason='invalid_evidence', skills=rows)
    return dict(accepted=score >= threshold, score=score,
                reason='nominal_margin' if score >= threshold else 'insufficient_margin', skills=rows)


def guarded_response(raw, scene, schema, evidence, threshold=.05, nominal=True):
    try:
        value = validate_response(strict_json(raw), scene, schema)
    except (ValueError, TypeError) as error:
        return dict(accepted=False, reason='invalid_response', detail=str(error))
    if value['decision'] == 'reject':
        return dict(accepted=False, reason='model_rejected', response=value)
    result = feasibility(value['plan'], evidence, threshold, nominal)
    return dict(result, response=value)
