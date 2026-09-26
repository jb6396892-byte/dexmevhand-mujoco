import unittest
import numpy as np
from fromrealhand.trajectory_control import smooth_blend, world_to_root_delta


class TrajectoryControlTest(unittest.TestCase):
    def test_phase_endpoints_and_clamping(self):
        np.testing.assert_allclose(smooth_blend(np.array([-1., 0., .5, 1., 2.])), [0., 0., .5, 1., 1.])

    def test_endpoints_have_zero_velocity_and_acceleration(self):
        h = 1e-4
        self.assertLess(abs(smooth_blend(h)/h), 1e-6)
        self.assertLess(abs((smooth_blend(2*h)-2*smooth_blend(h))/(h*h)), .01)
        self.assertLess(abs((1-smooth_blend(1-h))/h), 1e-6)

    def test_world_z_is_not_root_z(self):
        axes = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
        world = np.array([.01, -.02, .10])
        delta = world_to_root_delta(axes, world)
        np.testing.assert_allclose(delta, [-.01, .10, -.02])
        np.testing.assert_allclose(axes @ delta, world)

    def test_invalid_mapping_rejected(self):
        with self.assertRaises(ValueError):
            world_to_root_delta(np.eye(2), np.zeros(3))
        with self.assertRaises(ValueError):
            world_to_root_delta(np.eye(3), [0., np.nan, 0.])
