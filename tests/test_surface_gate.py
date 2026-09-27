import copy
import importlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
gate = importlib.import_module('33_optimize_surface_grasp').surface_gate


class SurfaceGateTest(unittest.TestCase):
    def setUp(self):
        self.valid = dict(physics_passed=True, max_loaded_gap_m=.0002,
                          max_penetration_m=.0007, max_joint_violation_rad=0.,
                          tail_slip_m=.00002, saturation=.003, finite=True,
                          tail_finger_contact_fraction=dict(thumb=1., index=1.))

    def test_near_surface_passes(self):
        self.assertTrue(gate(self.valid))

    def test_rejects_original_floating_contact(self):
        self.valid['max_loaded_gap_m'] = .00283
        self.assertFalse(gate(self.valid))

    def test_each_safety_condition(self):
        for key, value in dict(physics_passed=False, max_penetration_m=.0011,
                               max_joint_violation_rad=.03, tail_slip_m=.006,
                               saturation=.02, finite=False).items():
            report = copy.deepcopy(self.valid)
            report[key] = value
            with self.subTest(key=key):
                self.assertFalse(gate(report))

    def test_requires_index_and_thumb(self):
        for finger in ('thumb', 'index'):
            report = copy.deepcopy(self.valid)
            report['tail_finger_contact_fraction'][finger] = .7
            self.assertFalse(gate(report))
