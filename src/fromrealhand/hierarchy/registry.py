"""Strict, data-only registry and plan validation using existing parsers."""
import copy
import math
from pathlib import Path
import unicodedata

import jsonschema
import yaml
from ..skills import SKILLS


SEMANTICS = {
    'reach': ('safe_state', 'sustained_contact'),
    'grasp': ('contact', 'sustained_thumb_and_three_finger_support'),
    'lift': ('thumb_and_three_finger_support', 'supported_bottom_above_50mm'),
    'transport': ('supported_bottom_above_50mm', 'supported_goal_hold'),
}


class UniqueSafeLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in result:
            raise ValueError('Duplicate registry key: %s' % key)
        result[key] = loader.construct_object(value_node, deep=True)
    return result


UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def normalize_instruction(text):
    if not isinstance(text, str) or not text or len(text) > 120:
        raise ValueError('Instruction must contain 1 to 120 characters')
    # Exact aliases only: negation, new objects and coordinates are never discarded.
    return ''.join(unicodedata.normalize('NFKC', text).split()).rstrip('.!\u3002')


def object_schema(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def integer(low, high):
    return dict(type='integer', minimum=low, maximum=high)


def validate_schema(schema, value):
    try:
        jsonschema.Draft7Validator(schema).validate(value)
    except jsonschema.ValidationError as error:
        raise ValueError('Schema rejected: '+error.message) from error


class SkillRegistry:
    def __init__(self, config):
        skill = object_schema(dict(precondition={'type': 'string'}, success={'type': 'string'},
                                   max_steps=integer(1, 1600)))
        schema = object_schema(dict(
            schema_version={'const': 1}, backend={'const': 'verified_reference'},
            dataset={'type': 'string', 'minLength': 1},
            delivery_freeze_sha256={'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
            scenes=object_schema(dict(first={'type': 'string'}, second={'type': 'string'})),
            limits=object_schema(dict(max_total_steps=integer(1, 3200),
                wall_timeout_s={'type': 'number', 'minimum': .01, 'maximum': 600},
                max_provider_retries=integer(0, 3), max_replans=integer(0, 2))),
            skills=object_schema({name: skill for name in SKILLS}),
            instructions=object_schema({name: dict(type='array', minItems=1,
                items={'type': 'string', 'minLength': 1, 'maxLength': 120}, uniqueItems=True)
                for name in SKILLS + ('stop',)})))
        validate_schema(schema, config)
        if not math.isfinite(config['limits']['wall_timeout_s']):
            raise ValueError('Finite wall-clock budget required')
        dataset = Path(config['dataset'])
        if dataset.is_absolute() or '..' in dataset.parts:
            raise ValueError('Dataset must stay inside the project')
        self.config = copy.deepcopy(config)
        self.aliases = {}
        for name in SKILLS:
            spec = config['skills'][name]
            if (spec['precondition'], spec['success']) != SEMANTICS[name]:
                raise ValueError('Unsupported contract semantics: ' + name)
        for goal, aliases in config['instructions'].items():
            for text in aliases:
                key = normalize_instruction(text)
                if not key or key in self.aliases:
                    raise ValueError('Ambiguous or empty normalized instruction')
                self.aliases[key] = goal

    @classmethod
    def load(cls, path):
        try:
            config = yaml.load(Path(path).read_text(), Loader=UniqueSafeLoader)
        except (yaml.YAMLError, TypeError) as error:
            raise ValueError('Invalid data-only registry') from error
        return cls(config)

    def validate_plan(self, plan):
        schema = object_schema(dict(schema_version={'const': 1},
            scene={'enum': sorted(self.config['scenes'])}, object={'const': 'mug'},
            goal={'enum': list(SKILLS) + ['stop']},
            skills=dict(type='array', maxItems=4, uniqueItems=True, items={'enum': list(SKILLS)})))
        validate_schema(schema, plan)
        expected = [] if plan['goal'] == 'stop' else list(SKILLS[:SKILLS.index(plan['goal'])+1])
        if plan['skills'] != expected:
            raise ValueError('This backend supports only ordered nominal prefixes')
        return copy.deepcopy(plan)
