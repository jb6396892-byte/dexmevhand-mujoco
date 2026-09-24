from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fromrealhand.dexycb_io import (
    FINGER_CHAINS,
    joint_frames_from_positions,
    load_camera_to_world,
    project_points,
    read_yaml,
    scan_sequences,
)
from fromrealhand.pose_io import nearest_valid_indices


SERIAL = "932122060861"
SUBJECT = "20200709-subject-01"
SEQUENCE = "20200709_141754"


def synthetic_joints() -> np.ndarray:
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[0] = [0.0, 0.0, 0.8]
    finger_x = [-0.055, -0.028, 0.0, 0.028, 0.055]
    for x, chain in zip(finger_x, FINGER_CHAINS):
        for level, joint_index in enumerate(chain):
            joints[joint_index] = [x, 0.035 + level * 0.025, 0.80 - level * 0.004]
    return joints


def create_fixture(root: Path, frame_count: int = 3) -> Path:
    sequence_dir = root / SUBJECT / SEQUENCE
    camera_dir = sequence_dir / SERIAL
    camera_dir.mkdir(parents=True)
    intrinsics_dir = root / "calibration" / "intrinsics"
    intrinsics_dir.mkdir(parents=True)
    extrinsics_dir = root / "calibration" / "extrinsics_test"
    extrinsics_dir.mkdir(parents=True)

    meta = {
        "serials": [SERIAL],
        "num_frames": frame_count,
        "ycb_ids": [14],
        "ycb_grasp_ind": 0,
        "mano_sides": ["right"],
        "mano_calib": ["subject-01"],
        "extrinsics": "test",
        "fps": 30.0,
    }
    (sequence_dir / "meta.yml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    intrinsics = {"color": {"fx": 500.0, "fy": 500.0, "ppx": 320.0, "ppy": 240.0, "coeffs": [0, 0, 0, 0, 0]}}
    (intrinsics_dir / f"{SERIAL}_640x480.yml").write_text(yaml.safe_dump(intrinsics), encoding="utf-8")
    extrinsics = {
        "master": SERIAL,
        "extrinsics": {
            SERIAL: np.eye(4, dtype=float)[:3].reshape(-1).tolist(),
            "apriltag": np.eye(4, dtype=float)[:3].reshape(-1).tolist(),
        },
    }
    (extrinsics_dir / "extrinsics.yml").write_text(yaml.safe_dump(extrinsics), encoding="utf-8")

    camera_matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    for frame_index in range(frame_count):
        image = np.full((480, 640, 3), 32 + frame_index * 8, dtype=np.uint8)
        depth = np.full((480, 640), 800, dtype=np.uint16)
        joints = synthetic_joints()
        joints[:, 0] += frame_index * 0.002
        if frame_index == 1:
            joints[:] = -1
        joint_2d, _ = project_points(joints, camera_matrix)
        joint_2d[~np.isfinite(joint_2d).all(axis=1)] = -1
        object_pose = np.eye(4, dtype=np.float32)
        object_pose[:3, 3] = [0.04 + frame_index * 0.002, 0.08, 0.82]
        mano_pose = np.zeros((1, 51), dtype=np.float32)
        if frame_index != 1:
            mano_pose[0, 0] = 0.1
        cv2.imwrite(str(camera_dir / f"color_{frame_index:06d}.jpg"), image)
        cv2.imwrite(str(camera_dir / f"aligned_depth_to_color_{frame_index:06d}.png"), depth)
        np.savez(
            camera_dir / f"labels_{frame_index:06d}.npz",
            pose_y=object_pose[None, :3, :],
            pose_m=mano_pose,
            joint_3d=joints[None, :, :],
            joint_2d=joint_2d[None, :, :],
            seg=np.zeros((480, 640), dtype=np.uint8),
        )
    return sequence_dir


class DexYCBPipelineTest(unittest.TestCase):
    def test_nearest_valid_indices(self) -> None:
        indices = nearest_valid_indices(np.array([False, False, True, True, False]))
        self.assertEqual(indices.tolist(), [2, 2, 2, 3, 3])
        with self.assertRaisesRegex(ValueError, "no valid frames"):
            nearest_valid_indices(np.zeros(3, dtype=bool))

    def test_read_official_python_tuple_yaml_safely(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fromrealhand-yaml-") as temp_dir:
            path = Path(temp_dir) / "intrinsics.yml"
            path.write_text("extrinsics: !!python/tuple\n- 1.0\n- 2.0\n", encoding="utf-8")
            self.assertEqual(read_yaml(path)["extrinsics"], [1.0, 2.0])

    def test_camera_to_world_uses_apriltag_table_frame(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fromrealhand-extrinsics-") as temp_dir:
            root = Path(temp_dir)
            calibration = root / "calibration" / "extrinsics_test"
            calibration.mkdir(parents=True)
            camera_to_master = np.eye(4)
            camera_to_master[0, 3] = 0.4
            table_to_master = np.eye(4)
            table_to_master[2, 3] = 1.2
            payload = {
                "master": SERIAL,
                "extrinsics": {
                    SERIAL: camera_to_master[:3].reshape(-1).tolist(),
                    "apriltag": table_to_master[:3].reshape(-1).tolist(),
                },
            }
            (calibration / "extrinsics.yml").write_text(yaml.safe_dump(payload), encoding="utf-8")
            transform, master, _ = load_camera_to_world(root, {"extrinsics": "test"}, SERIAL)
            self.assertEqual(master, SERIAL)
            self.assertTrue(np.allclose(transform, np.linalg.inv(table_to_master) @ camera_to_master))

    def test_scan_convert_and_visualize(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fromrealhand-dexycb-") as temp_dir:
            workspace = Path(temp_dir)
            dataset_root = workspace / "dexycb"
            create_fixture(dataset_root)

            records = scan_sequences(dataset_root, object_name="025_mug", hand_side="right")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["grasped_object_id"], 14)
            self.assertTrue(records[0]["cameras"][SERIAL]["complete"])

            manifest = workspace / "mug.json"
            scan_command = [
                sys.executable,
                str(ROOT / "scripts" / "09_scan_dexycb.py"),
                "--root",
                str(dataset_root),
                "--object",
                "025_mug",
                "--hand-side",
                "right",
                "--output",
                str(manifest),
            ]
            subprocess.run(scan_command, check=True, cwd=ROOT)
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["count"], 1)

            converted = workspace / "seq_dexycb_001"
            convert_command = [
                sys.executable,
                str(ROOT / "scripts" / "10_convert_dexycb.py"),
                "--root",
                str(dataset_root),
                "--sequence",
                f"{SUBJECT}/{SEQUENCE}",
                "--camera",
                SERIAL,
                "--output",
                str(converted),
                "--transfer",
                "symlink",
            ]
            subprocess.run(convert_command, check=True, cwd=ROOT)
            self.assertEqual(np.load(converted / "valid_frames.npy").tolist(), [True, False, True])
            self.assertEqual(np.load(converted / "hand_pose" / "results_global_000000.npy").shape, (16, 4, 4))
            self.assertTrue(np.isnan(np.load(converted / "hand_pose" / "results_global_000001.npy")).all())
            self.assertEqual(np.load(converted / "object_pose" / "000000.npy").shape, (4, 4))
            self.assertTrue((converted / "rgb" / "000000.jpg").is_symlink())

            report_dir = workspace / "report"
            environment = dict(os.environ)
            environment["MPLCONFIGDIR"] = str(workspace / "matplotlib")
            visualize_command = [
                sys.executable,
                str(ROOT / "scripts" / "11_visualize_source_pose.py"),
                "--sequence-dir",
                str(converted),
                "--output",
                str(report_dir),
                "--sample-count",
                "3",
            ]
            subprocess.run(visualize_command, check=True, cwd=ROOT, env=environment)
            report = json.loads((report_dir / "trajectory_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["valid_frame_count"], 2)
            self.assertLess(report["joint_reprojection_error_px"]["max"], 1e-4)
            self.assertEqual(len(list((report_dir / "overlay").glob("*.jpg"))), 2)
            self.assertTrue((report_dir / "trajectory_3d.png").is_file())

    def test_joint_frame_shape_and_positions(self) -> None:
        joints = synthetic_joints()
        frames = joint_frames_from_positions(joints)
        self.assertEqual(frames.shape, (16, 4, 4))
        self.assertTrue(np.isfinite(frames).all())
        self.assertTrue(np.allclose(frames[0, :3, 3], joints[0]))
        for frame in frames:
            self.assertTrue(np.allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
