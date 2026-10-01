import importlib
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
dynamic = importlib.import_module('64_optimize_dynamic_contact')


class DynamicContactTests(unittest.TestCase):
    def test_verification_rejects_bad_actions_before_constructing_simulator(self):
        demo = dict(actions=np.zeros((30,30)))
        for actions in (np.zeros((2,30)),np.full((30,30),np.nan),np.full((30,30),1.1)):
            with self.assertRaises(ValueError):
                dynamic.verify({},demo,actions,Path('/tmp/not-created'))

    def test_envelope_keeps_prefix_and_has_smooth_endpoints(self):
        b = dynamic.action_envelope(30,10,10)
        np.testing.assert_array_equal(b[:11],0.)
        np.testing.assert_array_equal(b[20:],1.)
        self.assertTrue(np.all(np.diff(b)>=0))

    def test_zero_correction_preserves_actions(self):
        a = np.ones((30,30))*.25
        result = dynamic.corrected_actions(a,np.zeros(30),np.ones(30),np.ones(30),10,10)
        np.testing.assert_array_equal(result,a)

    def test_correction_is_scaled_once_and_does_not_mutate_input(self):
        a = np.zeros((30,30))
        result = dynamic.corrected_actions(a,np.ones(30),np.ones(30)*.1,np.ones(30)*2,10,10)
        np.testing.assert_array_equal(a,0.)
        np.testing.assert_allclose(result[-1],.2)
        np.testing.assert_array_equal(result[:11],0.)
