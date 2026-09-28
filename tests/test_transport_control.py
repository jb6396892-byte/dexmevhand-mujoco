import unittest
import numpy as np
from fromrealhand.transport_control import TransportCorrection


class TransportControlTests(unittest.TestCase):
    def test_requires_settled_contact(self):
        controller = TransportCorrection(1., .5)
        for _ in range(30):
            np.testing.assert_array_equal(controller.update([1., 0., 0.], False, .01), np.zeros(3))
        for _ in range(19):
            np.testing.assert_array_equal(controller.update([1., 0., 0.], True, .01), np.zeros(3))
        self.assertGreater(controller.update([1., 0., 0.], True, .01)[0], 0.)

    def test_vector_speed_and_displacement_bounds(self):
        controller = TransportCorrection(2., 1., max_shift=.08, max_speed=.03, settle_s=0.)
        previous = np.zeros(3)
        for _ in range(1000):
            current = controller.update([1., 1., 1.], True, .01)
            self.assertLessEqual(np.linalg.norm(current-previous), .0003+1e-12)
            self.assertLessEqual(np.linalg.norm(current), .08+1e-12)
            previous = current
        np.testing.assert_allclose(np.linalg.norm(current), .08)

    def test_contact_loss_freezes_shift_and_integral(self):
        controller = TransportCorrection(1., .5, settle_s=0.)
        shift = controller.update([.04, 0., 0.], True, .01)
        integral = controller.integral.copy()
        for _ in range(100):
            np.testing.assert_array_equal(controller.update([-.2, 0., 0.], False, .01), shift)
        np.testing.assert_array_equal(controller.integral, integral)

    def test_zero_gains_and_invalid_inputs(self):
        controller = TransportCorrection(0., 0., settle_s=0.)
        np.testing.assert_array_equal(controller.update([.04, .03, .02], True, 1.), np.zeros(3))
        with self.assertRaises(ValueError):
            controller.update([np.nan, 0., 0.], True, .01)
        with self.assertRaises(ValueError):
            TransportCorrection(-1., 0.)
        with self.assertRaises(ValueError):
            controller.update([0., 0., 0.], True, 0.)
