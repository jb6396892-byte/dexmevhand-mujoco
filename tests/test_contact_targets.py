import unittest
import numpy as np
from fromrealhand.video_fidelity import shifted_tip_targets


class ContactTargetTests(unittest.TestCase):
    def test_reference_unchanged_and_only_ring_shifted(self):
        local = np.arange(15, dtype=float).reshape(5, 3)/100
        before = local.copy()
        offset = np.zeros((5, 3)); offset[3, 0] = .02
        out = shifted_tip_targets(local, offset, .5)
        np.testing.assert_array_equal(local, before)
        np.testing.assert_allclose(out[3]-local[3], [.01, 0, 0])
        np.testing.assert_array_equal(out[[0, 1, 2, 4]], local[[0, 1, 2, 4]])

    def test_inactive_and_zero_offsets_preserve_targets(self):
        local = np.ones((5, 3))
        np.testing.assert_array_equal(shifted_tip_targets(local, np.full((5, 3), .001), -1.), local)
        np.testing.assert_array_equal(shifted_tip_targets(local, np.zeros((5, 3)), 1.), local)

    def test_limits_and_nonfinite_fail_closed(self):
        local = np.zeros((5, 3))
        for offset in [np.full((5, 3), .02), np.full((5, 3), np.nan), np.zeros((3,))]:
            with self.assertRaises(ValueError):
                shifted_tip_targets(local, offset, 1.)
        with self.assertRaises(ValueError):
            shifted_tip_targets(local, local, np.nan)
