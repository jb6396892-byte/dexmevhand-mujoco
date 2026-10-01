import unittest
import numpy as np
import json
import importlib
import sys
from pathlib import Path
from fromrealhand.corrective_learning import phase_weights,tracking_delta


class CorrectiveTests(unittest.TestCase):
    def test_zero_error_is_zero_correction(self):
        q=np.arange(30)*.01
        np.testing.assert_array_equal(tracking_delta(q,q,q,q,np.ones(30),.5),np.zeros(30))

    def test_feedback_direction_and_single_conversion(self):
        z=np.zeros(30);r=np.ones(30)*.001
        np.testing.assert_allclose(tracking_delta(z,r,z,z,np.full(30,4.),.25),.001)

    def test_errors_are_bounded(self):
        z=np.zeros(30)
        d=tracking_delta(z,np.full(30,1e6),z,np.full(30,1e6),np.ones(30),.5)
        np.testing.assert_allclose(d[:3],.015);np.testing.assert_allclose(d[3:],.06)
        with self.assertRaises(ValueError): tracking_delta(z,z,z,z,np.full(30,np.nan),.5)

    def test_phase_mass_not_frame_count(self):
        masses=[.15,.15,.2,.3,.2];w=phase_weights(1450,masses)
        self.assertAlmostEqual(w.sum(),1.)
        self.assertAlmostEqual(w[:50].sum(),.15)
        self.assertAlmostEqual(w[50:550].sum(),.15)
        self.assertTrue(np.all(w>0))

    def test_new_test_definition_differs_from_old_test(self):
        root=Path(__file__).resolve().parents[1]
        old=json.loads((root/'configs/v10-study.json').read_text())
        new=json.loads((root/'configs/v11-study.json').read_text())
        spec=new['heldout']['axis_cases_per_video']
        self.assertTrue(set(spec['cup_yaw_deg']).isdisjoint({c['cup_yaw_deg'] for c in old['heldout_per_video']}))
        self.assertNotIn(.0035,spec['cup_xy_offsets_m'])
        self.assertNotIn(.017,spec['goal_xy_offsets_m'])
        self.assertFalse(new['heldout']['tuning_after_test'])

    def test_safety_pulse_preserves_approach_and_hold(self):
        root=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root/'scripts'))
        module=importlib.import_module('71_develop_safe_experts')
        problem=object.__new__(module.SafetyProblem)
        problem.demo={'actions':np.zeros((1450,30))}
        problem.scales=np.ones(30);problem.conversion=np.ones(30)
        parameters=np.zeros(30);parameters[29]=.03;parameters[23]=-.02
        actions=problem.action_sequence(parameters)
        np.testing.assert_array_equal(actions[:631],0.)
        np.testing.assert_array_equal(actions[1120:],0.)
        np.testing.assert_allclose(actions[800,[29,23]],[.03,-.02])
        self.assertLess(np.max(np.abs(np.diff(actions,axis=0))),.001)


if __name__=='__main__': unittest.main()
