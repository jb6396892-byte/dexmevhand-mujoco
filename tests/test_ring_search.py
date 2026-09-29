import copy
import importlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
rank = importlib.import_module('59_refine_ring_contact').rank


class RingSearchTests(unittest.TestCase):
    def setUp(self):
        self.report = dict(surface_physics_passed=True, final_distance_m=.002,
                           tail_mean_tip_error_m=.014,
                           tail_finger_contact_fraction=dict(ring=0.),
                           tail_finger_tip_error_m=dict(ring=.02, thumb=.017))

    def test_rewards_loaded_ring_contact(self):
        loaded = copy.deepcopy(self.report)
        loaded['tail_finger_contact_fraction']['ring'] = 1.
        self.assertLess(rank(loaded), rank(self.report))

    def test_rejects_contact_gained_by_bad_physics(self):
        invalid = copy.deepcopy(self.report)
        invalid['tail_finger_contact_fraction']['ring'] = 1.
        invalid['surface_physics_passed'] = False
        self.assertGreater(rank(invalid), rank(self.report))

    def test_penalizes_target_and_fidelity_regression(self):
        for key, value in [('final_distance_m', .04), ('tail_mean_tip_error_m', .03)]:
            invalid = copy.deepcopy(self.report)
            invalid[key] = value
            self.assertGreater(rank(invalid), rank(self.report))
