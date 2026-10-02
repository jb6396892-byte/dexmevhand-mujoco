import copy
import json
from pathlib import Path
import tempfile
import unittest

from fromrealhand.hierarchy import SkillRegistry
from fromrealhand.language_planner.contracts import (
    canonical, compact, digest, feasibility, guarded_response, read, strict_json, validate_response, write)
from fromrealhand.language_planner.evaluation import evaluate
from fromrealhand.language_planner.sft import collate, encode, messages, verify_protocol

ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    """Only tests label masking and padding; never counts as pretrained-model inference."""
    def apply_chat_template(self, rows, tokenize=True, add_generation_prompt=False):
        value = ''.join('<%s>%s' % (r['role'], r['content']) for r in rows)
        value += '<assistant>' if add_generation_prompt else '<end>'
        return [ord(c) for c in value]


class LanguagePlannerTests(unittest.TestCase):
    def setUp(self):
        self.schema = read(ROOT/'configs/skill_plan.schema.json')
        self.registry = SkillRegistry.load(ROOT/'configs/skill_registry.yaml')
        self.evidence = dict(scenes={s: {k: dict(passed=True, margin_score=.2)
            for k in ('reach', 'grasp', 'lift', 'transport')} for s in ('first', 'second')})

    def guard(self, value, scene='first', **kwargs):
        return guarded_response(compact(value), scene, self.schema, self.evidence, **kwargs)

    def test_schema_matches_registry_for_all_supported_goals(self):
        for scene in ('first', 'second'):
            for goal in ('reach', 'grasp', 'lift', 'transport', 'stop'):
                response = canonical(goal, scene)
                self.assertEqual(validate_response(response, scene, self.schema), response)
                self.assertEqual(self.registry.validate_plan(response['plan']), response['plan'])

    def test_cannot_skip_reorder_or_repeat_prerequisites(self):
        for skills in (['lift'], ['reach', 'lift'], ['grasp', 'reach', 'lift'], ['reach', 'grasp', 'grasp', 'lift']):
            response = canonical('lift', 'first')
            response['plan']['skills'] = skills
            self.assertFalse(self.guard(response)['accepted'])

    def test_no_unknown_skills_fields_objects_or_scene_changes(self):
        for patch in (dict(skills=['pour']), dict(target=[1, 2]), dict(object='bottle'),
                      dict(scene='second'), dict(scene='unknown'), dict(goal='tilt')):
            response = canonical('lift', 'first')
            response['plan'].update(patch)
            self.assertFalse(self.guard(response)['accepted'])

    def test_duplicate_keys_nonfinite_or_code_are_rejected(self):
        for text in ('{"decision":"reject","decision":"execute"}', '{"x":NaN}',
                     '{"x":Infinity}', '```json\n{}\n```', 'import os', 'x'*4097):
            result = guarded_response(text, 'first', self.schema, self.evidence)
            self.assertFalse(result['accepted'])

    def test_reject_is_not_stop(self):
        result = self.guard(canonical('reject', 'first'))
        self.assertEqual(result['reason'], 'model_rejected')
        self.assertFalse(result['accepted'])
        stop = self.guard(canonical('stop', 'first'))
        self.assertTrue(stop['accepted'])
        self.assertEqual(stop['response']['plan']['skills'], [])

    def test_feasibility_margin_and_unknown_scene_fail_closed(self):
        plan = canonical('lift', 'first')
        self.assertTrue(self.guard(plan)['accepted'])
        self.assertFalse(self.guard(plan, nominal=False)['accepted'])
        self.evidence['scenes']['first']['grasp']['margin_score'] = .01
        self.assertFalse(self.guard(plan)['accepted'])
        self.assertAlmostEqual(self.guard(plan)['score'], .01)

    def test_nan_evidence_anywhere_is_rejected(self):
        for skill in ('reach', 'grasp', 'lift'):
            evidence = copy.deepcopy(self.evidence)
            evidence['scenes']['first'][skill]['margin_score'] = float('nan')
            self.assertFalse(feasibility(canonical('lift', 'first')['plan'], evidence)['accepted'])

    def test_valid_but_wrong_intent_is_not_semantic_success(self):
        row = dict(id='a', instruction='fixture', scene='first', response=canonical('lift', 'first'))
        result = evaluate([row], [dict(id='a', raw=compact(canonical('transport', 'first')))],
                          self.schema, self.evidence)
        self.assertEqual(result['metrics']['schema_legal'], 1)
        self.assertEqual(result['metrics']['accepted'], 1)
        self.assertEqual(result['metrics']['semantic_correct'], 0)
        self.assertFalse(result['language_acceptance_passed'])

    def test_unsupported_false_execution_is_a_separate_failure(self):
        row = dict(id='a', instruction='fixture', scene='first', response=canonical('reject', 'first'))
        result = evaluate([row], [dict(id='a', raw=compact(canonical('lift', 'first')))],
                          self.schema, self.evidence)
        self.assertEqual(result['metrics']['unsupported_false_execution'], 1)
        self.assertFalse(result['language_acceptance_passed'])

    def test_missing_extra_duplicate_prediction_ids_rejected(self):
        row = dict(id='a', instruction='fixture', scene='first', response=canonical('reject', 'first'))
        prediction = dict(id='a', raw='{"decision":"reject"}')
        for values in ([], [prediction, prediction], [dict(prediction, id='unknown')]):
            with self.assertRaises(ValueError):
                evaluate([row], values, self.schema, self.evidence)

    def test_assistant_only_loss_and_padding_masks(self):
        row = dict(instruction='fixture', scene='first', response=canonical('lift', 'first'))
        tokenizer = FakeTokenizer()
        value = encode(row, tokenizer, 'system', 2048)
        prefix = tokenizer.apply_chat_template(messages('fixture', 'first', 'system'), add_generation_prompt=True)
        self.assertEqual(value['labels'][:len(prefix)], [-100]*len(prefix))
        self.assertEqual(value['labels'][len(prefix):], value['input_ids'][len(prefix):])
        short = encode(dict(row, response=canonical('reject', 'first')), tokenizer, 'system', 2048)
        batch = collate([value, short], 0)
        self.assertTrue((batch['labels'][1, len(short['labels']):] == -100).all())
        self.assertTrue((batch['attention_mask'][1, len(short['labels']):] == 0).all())

    def test_overlong_sequence_never_silently_truncated(self):
        with self.assertRaises(ValueError):
            encode(dict(instruction='fixture', scene='first', response=canonical('lift', 'first')),
                   FakeTokenizer(), 'system', 5)

    def test_bad_instruction_and_scene_rejected(self):
        for text, scene in (('', 'first'), ('x'*121, 'first'), ('text', 'unknown')):
            with self.assertRaises(ValueError):
                messages(text, scene, 'system')

    def test_frozen_input_change_and_path_escape_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root/'input.json', dict(a=1))
            write(root/'protocol.json', dict(sha256={'input.json': digest(root/'input.json')}))
            verify_protocol(root, root)
            write(root/'input.json', dict(a=2))
            with self.assertRaises(ValueError):
                verify_protocol(root, root)
            write(root/'protocol.json', dict(sha256={'../escape.json': '0'*64}))
            with self.assertRaises(ValueError):
                verify_protocol(root, root)

    def test_base_phrases_do_not_cross_splits(self):
        phrases = read(ROOT/'configs/stage6-language-phrases.json')
        groups = [set(p for values in phrases[s].values() for p in values)
                  for s in ('train', 'validation', 'heldout')]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


if __name__ == '__main__':
    unittest.main()
