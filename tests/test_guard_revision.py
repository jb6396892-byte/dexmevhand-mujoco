import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fromrealhand.language_planner.contracts import canonical, compact
from fromrealhand.language_planner.guard_revision import instruction_contract, semantic_guard, evaluate_system


class GuardRevisionTests(unittest.TestCase):
    def test_retired_regression_grammar(self):
        for text,goal in [('将桌上水杯握在手中','grasp'),('我想让你举起眼前的杯子','lift'),
            ('我想让水杯被移送至目标处','transport'),('就停在这里吧','stop'),
            ('不必继续了停下来','stop'),('把杯子轻轻往上提起来','lift')]:
            self.assertEqual(instruction_contract(text).get('goal'),goal,text)

    def test_revision_does_not_remove_extra_actions(self):
        for text in ('将杯子握在我的手中','我想让你举起水壶','就停在这里吧然后抓杯子',
                     '把杯子往上提起来并张开手指','将杯子握在手中再递给我',
                     '不要将杯子握在手中','让杯子被移送至我手里','我想让你抓起杯子\u200b'):
            self.assertFalse(instruction_contract(text)['allowed'],text)

    def test_revision_vetoes_wrong_goal_without_repair(self):
        schema = json.loads((ROOT/'configs/skill_plan.schema.json').read_text())
        evidence = dict(scenes={'first':{s:dict(passed=True,margin_score=.5) for s in ('reach','grasp','lift','transport')}})
        for instruction in ('将杯子握在手中','将杯子握在我的手中'):
            gate = semantic_guard(compact(canonical('lift','first')),instruction,'first',schema,evidence)
            self.assertFalse(gate['accepted'])
            self.assertEqual(gate['response']['plan']['goal'],'lift')

    def test_raw_model_error_is_not_hidden(self):
        schema = json.loads((ROOT/'configs/skill_plan.schema.json').read_text())
        evidence = dict(scenes={'first':{s:dict(passed=True,margin_score=.5) for s in ('reach','grasp','lift','transport')}})
        row = dict(id='a',instruction='将杯子握在我的手中',scene='first',response=canonical('reject','first'))
        metrics = evaluate_system([row],[dict(id='a',raw=compact(canonical('grasp','first')))],schema,evidence)['metrics']
        self.assertEqual(metrics['semantic_correct'],0)
        self.assertEqual(metrics['raw_unsupported_false_execution'],1)
        self.assertEqual(metrics['unsafe_plan_acceptances'],0)


if __name__=='__main__': unittest.main()
