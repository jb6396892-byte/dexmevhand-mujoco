import importlib.util
from pathlib import Path
import unittest
import numpy as np


def load_script(filename):
    spec = importlib.util.spec_from_file_location(filename, Path(__file__).resolve().parents[1]/'scripts'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContactAuditTests(unittest.TestCase):
    def test_paired_comparison_does_not_hide_failed_fidelity(self):
        summarize = load_script('52_summarize_policy_pairs.py').summarize
        rows = [dict(nominal_actions=dict(surface_physics_passed=True, fidelity_passed=True, final_distance_m=.03),
                     student=dict(surface_physics_passed=True, fidelity_passed=False, final_distance_m=.01))]
        result = summarize(rows)
        self.assertEqual(result['residual_distance_wins'], 1)
        self.assertEqual(result['student']['full_and_20mm'], 0)
        self.assertAlmostEqual(result['mean_delta_mm'], -20.)

    def test_duplicate_comparison_includes_hidden_state(self):
        same = load_script('50_merge_admitted_demos.py').same_data
        a = dict(actions=np.zeros((2, 30)), sim_data=[dict(qvel=np.zeros(36))])
        b = dict(actions=np.zeros((2, 30)), sim_data=[dict(qvel=np.ones(36))])
        self.assertTrue(same(a, a))
        self.assertFalse(same(a, b))
        self.assertFalse(same(a, dict(actions=a['actions'])))

    def test_opposition_requires_thumb_and_other_loaded_normals(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
        cosine = load_script('51_audit_opposition.py').opposing_cosine
        self.assertIsNone(cosine({'TH': [], 'FF': [np.array([1., 0., 0.])]}))
        self.assertIsNone(cosine({'TH': [np.array([1., 0., 0.])]}))
        self.assertEqual(cosine({'TH': [np.array([1., 0., 0.])], 'FF': [np.array([-1., 0., 0.])]}), -1.)
        self.assertEqual(cosine({'TH': [np.array([1., 0., 0.])], 'FF': [np.array([1., 0., 0.])]}), 1.)
