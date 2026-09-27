import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from fromrealhand.policy_learning import StudentActions, lift_success, policy_features


class PolicyLearningTests(unittest.TestCase):
    def setUp(self):
        self.obs = np.zeros(39)
        self.qpos = np.zeros(37)
        self.qpos[33] = 1.
        self.qvel = np.zeros(36)

    def features(self, mode, step=0):
        return policy_features(self.obs, self.qpos, self.qvel, step, .01, 2., 5., mode)

    def test_dimensions_and_phase_endpoints(self):
        self.assertEqual(self.features('original').shape, (39,))
        self.assertEqual(self.features('state').shape, (81,))
        self.assertEqual(self.features('phase').shape, (82,))
        self.assertEqual(self.features('phase')[-1], 0.)
        self.assertEqual(self.features('phase', 2000)[-1], 1.)

    def test_velocity_visible_and_quaternion_sign_invariant(self):
        baseline = self.features('state')
        self.qpos[33] = -1.
        np.testing.assert_array_equal(baseline, self.features('state'))
        self.qvel[0] = .25
        self.assertEqual(self.features('state')[39], .25)
        self.qpos[33:37] = [0., 0., 0., 1.]
        self.assertFalse(np.array_equal(baseline[-6:], self.features('state')[-6:]))

    def test_invalid_features_fail_closed(self):
        with self.assertRaises(ValueError):
            self.features('unknown')
        self.qpos[33] = 0.
        with self.assertRaises(ValueError):
            self.features('phase')
        self.obs[0] = np.nan
        with self.assertRaises(ValueError):
            self.features('original')

    def student(self, reference=None, dt=.01):
        model = torch.nn.Linear(82, 30)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.fill_(.02)
        checkpoint = dict(policy=SimpleNamespace(model=model), dt=dt, duration=2.,
                          time_scale=5., feature_mode='phase', action_reference=reference)
        data = SimpleNamespace(qpos=self.qpos, qvel=self.qvel)
        env = SimpleNamespace(_get_observations=lambda: self.obs,
                              sim=SimpleNamespace(data=data), control_timestep=.01)
        return StudentActions(checkpoint, SimpleNamespace(env=env, duration=2.), 5)

    def test_residual_uses_reference_once(self):
        student = self.student(np.full((5, 30), .1))
        np.testing.assert_allclose(student[0], .12, atol=1e-7)
        np.testing.assert_allclose(self.student()[0], .02, atol=1e-7)

    def test_reference_shape_and_clock_checked(self):
        with self.assertRaises(ValueError):
            self.student(np.zeros((4, 30)))
        with self.assertRaises(ValueError):
            self.student(np.full((5, 30), np.nan))
        with self.assertRaises(ValueError):
            self.student(dt=.02)

    def test_lift_is_serializable_bool(self):
        result = lift_success(dict(hold_s=np.float64(2), tail_min_bottom_m=.02,
                                   tail_min_fingers=3, tail_min_force_n=1.))
        self.assertIs(result, True)

    def test_sustained_divergence_ignores_short_spikes(self):
        path = Path(__file__).resolve().parents[1]/'scripts/45_diagnose_policy_divergence.py'
        spec = importlib.util.spec_from_file_location('diagnose', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIsNone(module.first_sustained([]))
        self.assertIsNone(module.first_sustained([True]*9))
        self.assertEqual(module.first_sustained([True]*9+[False]+[True]*10), 10)
        with self.assertRaises(ValueError):
            module.first_sustained([True], 0)


if __name__ == '__main__':
    unittest.main()
