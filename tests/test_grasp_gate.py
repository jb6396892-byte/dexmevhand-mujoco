import unittest
from fromrealhand.grasp_gate import physical_gate


class GraspGateTest(unittest.TestCase):
    def setUp(self):
        self.report = dict(hold_s=2., tail_min_bottom_m=.05,
                           tail_min_force_n=2., tail_min_fingers=3,
                           max_penetration_m=.001, saturation=.02,
                           final_distance_m=.04)

    def test_stable_lift_passes(self):
        self.assertTrue(physical_gate(self.report))

    def test_transient_lift_does_not_pass(self):
        self.report['tail_min_bottom_m'] = .003
        self.assertFalse(physical_gate(self.report))

    def test_each_condition_is_required(self):
        failures = dict(hold_s=.5, tail_min_bottom_m=.01,
                        tail_min_force_n=0., tail_min_fingers=1,
                        max_penetration_m=.006, saturation=.2,
                        final_distance_m=.2)
        for key, value in failures.items():
            report = dict(self.report, **{key: value})
            with self.subTest(key=key):
                self.assertFalse(physical_gate(report))
