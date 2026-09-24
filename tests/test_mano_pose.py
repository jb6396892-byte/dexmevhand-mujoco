from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fromrealhand.mano_pose import make_mano_layer, recover_mano_frames, rotation_error_degrees


class ManoPoseTest(unittest.TestCase):
    def test_global_frame_origins_match_mano_joints(self) -> None:
        model_root = Path(os.environ.get("MANO_ROOT", "/media/smgbro/shared/DexYCB/mano/models"))
        if not (model_root / "MANO_RIGHT.pkl").is_file():
            self.skipTest("licensed MANO_RIGHT.pkl is not installed")
        layer = make_mano_layer(str(model_root))
        pose = np.zeros(51, dtype=np.float32)
        pose[:3] = [0.2, -0.1, 0.3]
        pose[3:7] = [0.1, -0.15, 0.2, 0.05]
        pose[48:51] = [0.3, 0.1, 0.7]
        frames, joints = recover_mano_frames(layer, pose, np.zeros(10, dtype=np.float32))
        selected = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
        self.assertLess(np.max(np.abs(frames[:, :3, 3] - joints[selected])), 1e-6)
        self.assertTrue(np.allclose(frames[:, :3, :3].transpose(0, 2, 1) @ frames[:, :3, :3], np.eye(3), atol=1e-5))
        self.assertLess(np.max(rotation_error_degrees(frames[:, :3, :3], frames[:, :3, :3])), 0.05)


if __name__ == "__main__":
    unittest.main()
