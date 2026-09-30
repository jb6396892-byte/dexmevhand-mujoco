import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
TrackingActions = importlib.import_module('63_plan_contact_transition').TrackingActions
contact_push = importlib.import_module('63_plan_contact_transition').contact_push


class ContactTransitionTests(unittest.TestCase):
    def test_contact_integral_has_anti_windup(self):
        force, integral = contact_push(.01, .01, 2.99, 500., 3.)
        self.assertEqual(force, 3.)
        self.assertEqual(integral, 3.)
        force, integral = contact_push(-.001, .01, integral, 500., 3.)
        self.assertLess(integral, 3.)
        self.assertLessEqual(force, 3.)

    def build(self, delta=0., gain=2.):
        states = [dict(qpos=np.zeros(37), qvel=np.zeros(36)) for _ in range(10)]
        demo = dict(actions=np.full((10, 30), .1), sim_data=states)
        bias = np.zeros((30, 3)); bias[:, 1] = -1.
        model = SimpleNamespace(actuator_biasprm=bias, actuator_gainprm=np.ones((30, 3)))
        data = SimpleNamespace(qpos=np.zeros(37), qvel=np.zeros(36))
        env = SimpleNamespace(sim=SimpleNamespace(data=data), control_timestep=.01, act_rng=np.ones(30))
        exp = SimpleNamespace(model=model, env=env)
        plan = dict(steps=np.array([3, 6, 9]), deltas=np.full((3, 24), delta))
        return TrackingActions(exp, demo, plan, gain), demo

    def test_ring_feedback_does_not_add_other_finger_gain(self):
        actions, _ = self.build(.1, 2.)
        actions.ring_only = True
        other, _ = self.build(.1, 0.)
        result = actions[6]-other[6]
        np.testing.assert_array_equal(result[np.r_[0:16,20:30]], np.zeros(26))
        self.assertTrue((result[16:20] > 0).all())

    def test_object_relative_root_uses_actual_cup_and_bounded_integral(self):
        actions, demo = self.build(0., 0.)
        actions.object_relative_root = True
        m, e = actions.e.model, actions.e.env
        m.body_name2id = lambda name: 0
        m.body_quat = np.array([[1.,0.,0.,0.]])
        m.body_pos = np.zeros((1,3))
        m.jnt_range = np.tile([-1.,1.], (30,1))
        e.obj_bid = 0
        e.sim.data.body_xpos = np.array([[.01,0.,0.]])
        e.sim.data.body_xmat = np.eye(3).reshape(1,9)
        for s in demo['sim_data']:
            s['qpos'][33] = 1.
        self.assertAlmostEqual(actions[6][0], .10997)
        for _ in range(2000):
            actions[6]
        self.assertLessEqual(np.max(np.abs(actions.integral)), .03)

    def test_zero_plan_reproduces_nominal_action_at_nominal_state(self):
        actions, demo = self.build()
        for i in range(len(actions)):
            np.testing.assert_array_equal(actions[i], demo['actions'][i])

    def test_plan_can_explicitly_include_root_joints(self):
        old, demo = self.build(0., 0.)
        plan = dict(steps=np.array([3, 6, 9]), deltas=np.full((3,30), .01), joint_indices=np.arange(30))
        actions = TrackingActions(old.e, demo, plan, 0.)
        np.testing.assert_allclose(actions[6], .11)

    def test_fractional_root_follow_is_bounded(self):
        actions, demo = self.build(0., 0.)
        actions.root_follow = .5
        m, e = actions.e.model, actions.e.env
        m.body_name2id = lambda name: 0
        m.body_quat = np.array([[1.,0.,0.,0.]])
        m.body_pos = np.zeros((1,3))
        m.jnt_range = np.tile([-1.,1.], (30,1))
        e.obj_bid = 0
        e.sim.data.body_xpos = np.array([[1.,0.,0.]])
        e.sim.data.body_xmat = np.eye(3).reshape(1,9)
        for s in demo['sim_data']:
            s['qpos'][33] = 1.
        self.assertAlmostEqual(actions[6][0], .11)

    def test_pretransition_actions_and_inputs_unchanged(self):
        actions, demo = self.build(.1)
        before = demo['actions'].copy()
        np.testing.assert_array_equal(actions[1], before[1])
        self.assertGreater(actions[6][16], before[6,16])
        np.testing.assert_array_equal(demo['actions'], before)

    def test_output_is_normalized_and_finite(self):
        actions, _ = self.build(10., 50.)
        for i in range(len(actions)):
            self.assertTrue(np.isfinite(actions[i]).all())
            self.assertLessEqual(np.max(np.abs(actions[i])), 1.)
