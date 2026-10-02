import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fromrealhand.desktop.planning import plan_instruction,validated_plan
from fromrealhand.language_planner.contracts import canonical,compact,digest,write


class DesktopPlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.storage=Path(self.temp.name); self.study=self.storage/'language/study_v4_guard2'
        self.output=self.storage/'language/desktop_runs/test'
        write(self.study/'acceptance/summary.json',dict(passed=True))
        write(self.study/'dataset/feasibility.json',dict(scenes={'first':{s:dict(passed=True,margin_score=.5)
            for s in ('reach','grasp','lift','transport')}}))
        self.check=patch('fromrealhand.desktop.planning.checked_study',return_value=(self.study,dict(feasibility_threshold=.05)))
        self.check.start(); self.addCleanup(self.check.stop)

    def generated(self,instruction='抓起杯子'):
        plan=canonical('lift','first')['plan']
        write(self.output/'generation.json',dict(producer='lora_model',instruction=instruction,scene='first',
            raw=compact(canonical('lift','first')),acceptance_sha256=digest(self.study/'acceptance/summary.json')))
        write(self.output/'plan.json',plan)
        return plan

    def test_unsupported_preflight_has_no_model_or_simulation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(plan_instruction(self.storage,'将杯子放到我手上','first',self.output))
        data=json.loads((self.output/'generation.json').read_text())
        self.assertFalse(data['model_called']); self.assertFalse(data['simulation_created']); self.assertEqual(data['steps'],0)

    def test_accepted_plan_revalidated(self):
        expected=self.generated()
        self.assertEqual(expected,validated_plan(self.storage,self.output))

    def test_edited_plan_rejected(self):
        self.generated(); write(self.output/'plan.json',canonical('transport','first')['plan'])
        with self.assertRaises(ValueError): validated_plan(self.storage,self.output)

    def test_instruction_mismatch_rejected(self):
        self.generated('手先到杯子边上去')
        with self.assertRaises(ValueError): validated_plan(self.storage,self.output)

    def test_stale_receipt_rejected(self):
        self.generated(); write(self.study/'acceptance/summary.json',dict(passed=False))
        with self.assertRaises(ValueError): validated_plan(self.storage,self.output)

    def test_wrong_storage_rejected(self):
        with self.assertRaises(ValueError): plan_instruction(self.storage,'抓起杯子','first',self.storage/'outside')


if __name__=='__main__': unittest.main()
