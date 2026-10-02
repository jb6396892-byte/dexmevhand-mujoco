import unittest
import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import patch
from fromrealhand.routed_residual import phase_routes, RoutedNetwork, takeover_weight, TakeoverActions
from fromrealhand.corrective_learning import aligned_phase_indices


class RoutedResidualTests(unittest.TestCase):
    def geometry(self):
        return dict(source_frames=np.arange(1, 74), fps=30.)

    def test_routes_are_convex_and_match_hard_phase_away_from_boundaries(self):
        g = self.geometry()
        hard = phase_routes(1450, g, 5., 0)
        soft = phase_routes(1450, g, 5., 10)
        np.testing.assert_allclose(soft.sum(1), 1.)
        self.assertTrue((soft >= 0).all())
        phase = aligned_phase_indices(1450, g, 5.)
        np.testing.assert_array_equal(hard.argmax(1), phase)
        self.assertTrue((np.count_nonzero(soft, axis=1) <= 2).all())
        self.assertLess(np.max(np.abs(np.diff(soft, axis=0))), .08)
        with self.assertRaises(ValueError):
            phase_routes(1450, g, 5., -1)

    def test_heads_are_independent_and_convex_blend_is_correct(self):
        base = torch.nn.Linear(2, 1)
        with torch.no_grad():
            base.weight.fill_(0.)
            base.bias.fill_(1.)
        net = RoutedNetwork(base, 2)
        with torch.no_grad():
            net.heads[1].bias.fill_(3.)
        result = net(torch.zeros(3, 2), torch.tensor([[1., 0.], [.5, .5], [0., 1.]]))
        np.testing.assert_allclose(result.detach().numpy().ravel(), [1., 2., 3.])
        self.assertEqual(base.bias.item(), 1.)

    def test_takeover_has_unchanged_prefix_and_full_labels_only_after_transition(self):
        self.assertEqual(takeover_weight(99, 100, 20), 0.)
        self.assertEqual(takeover_weight(100, 100, 20), 0.)
        self.assertEqual(takeover_weight(110, 100, 20), .5)
        self.assertEqual(takeover_weight(120, 100, 20), 1.)
        with self.assertRaises(ValueError):
            takeover_weight(100, 100, 0)

    def test_takeover_labels_are_executed_bounded_actions_and_do_not_write_state(self):
        q = np.zeros(37); q[33] = 1.; q.setflags(write=False)
        v = np.zeros(36); v.setflags(write=False)
        env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=q, qvel=v)),
                              act_rng=np.ones(30), control_timestep=.01,
                              _get_observations=lambda:np.zeros(39))
        model = SimpleNamespace(actuator_biasprm=np.tile([0., -1., 0.], (30, 1)),
                                actuator_gainprm=np.ones((30, 3)))
        experiment = SimpleNamespace(env=env, model=model, duration=2.4)
        video = dict(id=1, horizon=10, control=dict(time_scale=5.))
        class Student:
            reference = np.full((10, 30), .2)
            limits = np.full(30, .01)
            def __getitem__(self, step):
                return np.full(30, .19)
        expert = dict(actions=np.full((10, 30), .3),
                      sim_data=[dict(qpos=q, qvel=v) for _ in range(10)])
        with patch('fromrealhand.routed_residual.student_actions', return_value=Student()):
            c = TakeoverActions({}, experiment, video, expert, 2, 2, .1, 6)
            np.testing.assert_allclose(c[1], .19)
            np.testing.assert_allclose(c[3], .20)
            self.assertEqual(len(c.labels), 0)
            action = c[4]
            np.testing.assert_allclose(action, .21)
            np.testing.assert_allclose(c.labels[0]+c.student.reference[4], action)
            self.assertEqual(c.steps, [4])
            c[6]
            self.assertEqual(c.steps, [4])
        self.assertFalse(q.flags.writeable)
        self.assertFalse(v.flags.writeable)


if __name__ == '__main__':
    unittest.main()
