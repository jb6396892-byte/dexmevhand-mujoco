import unittest
import numpy as np
from fromrealhand.video_fidelity import fidelity_metrics, object_relative, source_clock


class VideoFidelityTests(unittest.TestCase):
    def test_shared_rigid_transform_preserves_object_relative_pose(self):
        points = np.random.RandomState(4).normal(size=(21, 3))
        pose = np.eye(4)
        pose[:3, 3] = [.4, -.2, .5]
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        moved = pose.copy()
        moved[:3, :3] = rotation
        moved[:3, 3] = rotation @ pose[:3, 3] + 2.
        np.testing.assert_allclose(object_relative(points, pose),
                                   object_relative(points @ rotation.T + 2., moved), atol=1e-12)

    def test_finger_swap_is_detected(self):
        points = np.arange(63.).reshape(21, 3) / 1000.
        swapped = points.copy()
        swapped[8], swapped[12] = points[12], points[8]
        error = fidelity_metrics(swapped, np.eye(4), points, np.eye(4))['tip_error_m']
        self.assertEqual(error[0], 0.)
        self.assertGreater(error[1], .01)
        self.assertGreater(error[2], .01)

    def test_clock_keeps_order_and_holds_endpoints(self):
        times = np.linspace(0., 10., 1001)
        clock = source_clock(times, 2., time_scale=3.)
        self.assertTrue(np.all(np.diff(clock) >= -1e-12))
        self.assertEqual(clock[0], 0.)
        self.assertEqual(clock[-1], 2.)
        self.assertAlmostEqual(source_clock(3.5, 2., time_scale=3.), 1.)


if __name__ == '__main__':
    unittest.main()
