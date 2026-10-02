import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fromrealhand.language_planner.contracts import canonical, compact, digest, write
from fromrealhand.language_planner.instruction_guard import instruction_contract, semantic_guard
from fromrealhand.language_planner.refinement import adapter_hashes, deployment_matches, evaluate_system, system_passes


class InstructionGuardTests(unittest.TestCase):
    def test_development_positive_contracts(self):
        phrases = json.loads((ROOT/'configs/stage6-refinement-validation.json').read_text())
        for goal, texts in phrases.items():
            if goal=='reject': continue
            for text in texts:
                with self.subTest(text=text):
                    self.assertEqual(instruction_contract(text).get('goal'),goal)

    def test_development_negatives_fail_closed(self):
        phrases = json.loads((ROOT/'configs/stage6-refinement-validation.json').read_text())
        for text in phrases['reject']:
            self.assertFalse(instruction_contract(text)['allowed'],text)

    def test_known_boundaries_and_hidden_suffixes(self):
        for text in ('将杯子放到我手上','把杯子递给我','把杯子拿起来\u200b',
                     '把杯子拿起来然后做未知动作','把杯子搬到目标后爆炸',
                     '把手移动到杯子旁边的目标位置','抓起杯子并停止'):
            self.assertFalse(instruction_contract(text)['allowed'],text)
        self.assertEqual(instruction_contract('手先到杯子边上去')['goal'],'reach')
        self.assertEqual(instruction_contract('请停止当前任务')['goal'],'stop')

    def setUp(self):
        self.schema = json.loads((ROOT/'configs/skill_plan.schema.json').read_text())
        self.evidence = dict(scenes={'first':{s:dict(passed=True,margin_score=.5)
                             for s in ('reach','grasp','lift','transport')}})

    def test_guard_vetoes_without_rewriting(self):
        raw = compact(canonical('lift','first'))
        for text in ('手先到杯子边上去','将杯子放到我手上'):
            result = semantic_guard(raw,text,'first',self.schema,self.evidence)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['response']['plan']['goal'],'lift')
        self.assertTrue(semantic_guard(raw,'抓起杯子','first',self.schema,self.evidence)['accepted'])

    def test_raw_errors_remain_visible(self):
        row = dict(id='a',instruction='将杯子放到我手上',scene='first',response=canonical('reject','first'))
        result = evaluate_system([row],[dict(id='a',raw=compact(canonical('lift','first')))],self.schema,self.evidence)
        m = result['metrics']
        self.assertEqual(m['semantic_correct'],0)
        self.assertEqual(m['raw_unsupported_false_execution'],1)
        self.assertEqual(m['unsupported_false_execution'],0)
        self.assertEqual(m['semantic_vetoes'],1)

    def test_reject_everything_cannot_pass(self):
        gate = dict(raw_semantic_min=.9,schema_fraction_min=.95,supported_correct_coverage_min=.9,unsafe_acceptances_max=0)
        m = dict(total=100,semantic_accuracy=1.,schema_legal=100,supported_correct_coverage=0.,unsafe_plan_acceptances=0)
        self.assertFalse(system_passes(m,gate))

    def test_acceptance_is_bound_to_exact_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            study,model = Path(folder)/'study',Path(folder)/'model'
            for path in (study/'candidate/adapter/weights.json',study/'dataset/protocol.json',
                         study/'protocol.json',study/'candidate-heldout/predictions.json',model/'source.json'):
                write(path,dict(version=1))
            receipt = dict(model_acceptance_passed=True,
                adapter_sha256=adapter_hashes(study/'candidate/adapter'),
                model_source_sha256=digest(model/'source.json'),
                dataset_protocol_sha256=digest(study/'dataset/protocol.json'),
                study_protocol_sha256=digest(study/'protocol.json'),
                predictions_sha256=digest(study/'candidate-heldout/predictions.json'))
            self.assertTrue(deployment_matches(receipt,study,model))
            self.assertFalse(deployment_matches(dict(receipt,model_acceptance_passed=False),study,model))
            for path in (study/'candidate/adapter/weights.json',study/'dataset/protocol.json',
                         study/'protocol.json',study/'candidate-heldout/predictions.json',model/'source.json'):
                write(path,dict(version=2))
                self.assertFalse(deployment_matches(receipt,study,model),str(path))
                write(path,dict(version=1))

    def test_wrong_goal_cannot_pass_on_structural_validity(self):
        for request,goal in (('接近杯子','reach'),('握住杯子','grasp'),('抓起杯子','lift'),
                             ('把杯子搬到目标位置','transport'),('停止当前动作','stop')):
            for proposed in ('reach','grasp','lift','transport','stop'):
                accepted = semantic_guard(compact(canonical(proposed,'first')),request,'first',
                                          self.schema,self.evidence)['accepted']
                self.assertEqual(accepted,proposed==goal,(request,proposed))


if __name__=='__main__':
    unittest.main()
