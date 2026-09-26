from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fromrealhand.transforms import align_object_origin, transform_points, transform_pose


class TaskAlignmentTests(unittest.TestCase):
    def test_object_and_hand_share_translation(self) -> None:
        camera_to_world = np.eye(4)
        camera_to_world[:3, 3] = [0.8, -0.3, 0.1]
        object_pose = np.eye(4)
        object_pose[:3, 3] = [0.2, 0.4, -0.05]
        target = np.array([0.05, 0.0, 0.04065])
        hand = np.array([0.1, 0.6, 0.02])

        aligned = align_object_origin(camera_to_world, object_pose, target)
        np.testing.assert_allclose(transform_pose(aligned, object_pose)[:3, 3], target)
        before = transform_points(camera_to_world, hand) - transform_pose(camera_to_world, object_pose)[:3, 3]
        after = transform_points(aligned, hand) - transform_pose(aligned, object_pose)[:3, 3]
        np.testing.assert_allclose(after, before)

    def test_invalid_target_rejected(self) -> None:
        with self.assertRaises(ValueError):
            align_object_origin(np.eye(4), np.eye(4), [np.nan, 0, 0])


if __name__ == "__main__":
    unittest.main()
