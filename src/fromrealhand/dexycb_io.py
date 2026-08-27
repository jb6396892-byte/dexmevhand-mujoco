from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml


YCB_CLASSES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}
YCB_NAME_TO_ID = {name: object_id for object_id, name in YCB_CLASSES.items()}

COLOR_PATTERN = "color_{:06d}.jpg"
DEPTH_PATTERN = "aligned_depth_to_color_{:06d}.png"
LABEL_PATTERN = "labels_{:06d}.npz"

# DexMV uses the wrist plus MCP/PIP/DIP frames for all five fingers.
DEXMV_FRAME_JOINTS = (0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)
FINGER_CHAINS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return data


def write_json(data: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def object_id_from_name(name: str) -> int:
    if name.isdigit():
        object_id = int(name)
        if object_id in YCB_CLASSES:
            return object_id
    if name in YCB_NAME_TO_ID:
        return YCB_NAME_TO_ID[name]
    matches = [object_id for object_id, class_name in YCB_CLASSES.items() if class_name.endswith(f"_{name}")]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"unknown YCB object {name!r}; use a class name such as '025_mug'")


def discover_sequence_meta(root: str | Path) -> list[Path]:
    dataset_root = Path(root)
    return sorted(dataset_root.glob("2020*-subject-*/*/meta.yml"))


def grasped_object_id(meta: dict[str, Any]) -> int | None:
    ycb_ids = [int(value) for value in meta.get("ycb_ids", [])]
    grasp_index = meta.get("ycb_grasp_ind")
    if not isinstance(grasp_index, int) or not 0 <= grasp_index < len(ycb_ids):
        return None
    return ycb_ids[grasp_index]


def sequence_hand_side(meta: dict[str, Any]) -> str | None:
    sides = meta.get("mano_sides", [])
    if isinstance(sides, str):
        return sides.lower()
    if isinstance(sides, list) and sides:
        return str(sides[0]).lower()
    return None


def camera_file_counts(sequence_dir: Path, serial: str, num_frames: int) -> dict[str, Any]:
    camera_dir = sequence_dir / serial
    counts = {
        "color": len(list(camera_dir.glob("color_*.jpg"))),
        "depth": len(list(camera_dir.glob("aligned_depth_to_color_*.png"))),
        "label": len(list(camera_dir.glob("labels_*.npz"))),
    }
    counts["expected"] = num_frames
    counts["complete"] = camera_dir.is_dir() and all(counts[name] == num_frames for name in ("color", "depth", "label"))
    return counts


def scan_sequences(
    root: str | Path,
    *,
    object_name: str = "025_mug",
    hand_side: str = "right",
    require_grasp_target: bool = True,
) -> list[dict[str, Any]]:
    dataset_root = Path(root).resolve()
    target_id = object_id_from_name(object_name)
    records = []
    for meta_path in discover_sequence_meta(dataset_root):
        meta = read_yaml(meta_path)
        ycb_ids = [int(value) for value in meta.get("ycb_ids", [])]
        grasp_id = grasped_object_id(meta)
        side = sequence_hand_side(meta)
        if require_grasp_target and grasp_id != target_id:
            continue
        if not require_grasp_target and target_id not in ycb_ids:
            continue
        if hand_side and side != hand_side.lower():
            continue

        sequence_dir = meta_path.parent
        serials = [str(value) for value in meta.get("serials", [])]
        num_frames = int(meta.get("num_frames", 0))
        cameras = {
            serial: camera_file_counts(sequence_dir, serial, num_frames)
            for serial in serials
        }
        records.append(
            {
                "subject": sequence_dir.parent.name,
                "sequence": sequence_dir.name,
                "relative_sequence": str(sequence_dir.relative_to(dataset_root)),
                "hand_side": side,
                "num_frames": num_frames,
                "fps": float(meta.get("fps", 30.0)),
                "ycb_ids": ycb_ids,
                "grasped_object_id": grasp_id,
                "grasped_object": YCB_CLASSES.get(grasp_id),
                "serials": serials,
                "extrinsics": meta.get("extrinsics"),
                "mano_calib": meta.get("mano_calib", []),
                "cameras": cameras,
            }
        )
    return records


def find_intrinsics_file(root: str | Path, serial: str) -> Path:
    candidates = sorted((Path(root) / "calibration" / "intrinsics").glob(f"{serial}_*.yml"))
    if not candidates:
        raise FileNotFoundError(f"no intrinsics YAML found for camera {serial}")
    preferred = [path for path in candidates if path.name.endswith("_640x480.yml")]
    return preferred[0] if preferred else candidates[0]


def load_intrinsics(root: str | Path, serial: str) -> tuple[np.ndarray, np.ndarray, Path]:
    path = find_intrinsics_file(root, serial)
    data = read_yaml(path).get("color")
    if not isinstance(data, dict):
        raise ValueError(f"missing color intrinsics in {path}")
    required = ("fx", "fy", "ppx", "ppy")
    if any(name not in data for name in required):
        raise ValueError(f"incomplete color intrinsics in {path}")
    matrix = np.array(
        [[data["fx"], 0.0, data["ppx"]], [0.0, data["fy"], data["ppy"]], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    coefficients = data.get("coeffs", data.get("distortion_coefficients", [0.0] * 5))
    distortion = np.asarray(coefficients, dtype=np.float32).reshape(-1)
    return matrix, distortion, path


def _as_homogeneous(matrix: Any, *, label: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    if value.size == 12:
        value = value.reshape(3, 4)
    if value.shape == (3, 4):
        value = np.vstack((value, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)))
    if value.shape != (4, 4):
        raise ValueError(f"{label} must contain a 3x4 or 4x4 transform, got {value.shape}")
    return value


def load_camera_to_world(root: str | Path, sequence_meta: dict[str, Any], serial: str) -> tuple[np.ndarray, str, Path]:
    extrinsics_name = sequence_meta.get("extrinsics")
    if not extrinsics_name:
        raise ValueError("sequence meta.yml has no extrinsics key")
    path = Path(root) / "calibration" / f"extrinsics_{extrinsics_name}" / "extrinsics.yml"
    data = read_yaml(path)
    transforms = data.get("extrinsics")
    if not isinstance(transforms, dict) or serial not in transforms:
        raise KeyError(f"camera {serial} is missing from {path}")
    return _as_homogeneous(transforms[serial], label=f"extrinsics[{serial}]"), str(data.get("master", "unknown")), path


def load_label(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as payload:
        required = ("pose_y", "pose_m", "joint_3d", "joint_2d")
        missing = [name for name in required if name not in payload]
        if missing:
            raise KeyError(f"label file {path} is missing keys: {', '.join(missing)}")
        return {name: np.asarray(payload[name]).copy() for name in required}


def object_pose_from_label(label: dict[str, np.ndarray], grasp_index: int) -> np.ndarray:
    poses = np.asarray(label["pose_y"], dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (3, 4):
        raise ValueError(f"pose_y must have shape (N, 3, 4), got {poses.shape}")
    if not 0 <= grasp_index < poses.shape[0]:
        raise IndexError(f"grasped object index {grasp_index} is outside pose_y with {poses.shape[0]} objects")
    return _as_homogeneous(poses[grasp_index], label="object pose")


def hand_joints_from_label(label: dict[str, np.ndarray]) -> np.ndarray:
    joints = np.asarray(label["joint_3d"], dtype=np.float32)
    if joints.shape == (1, 21, 3):
        joints = joints[0]
    if joints.shape != (21, 3):
        raise ValueError(f"joint_3d must have shape (1, 21, 3) or (21, 3), got {joints.shape}")
    return joints


def mano_pose_from_label(label: dict[str, np.ndarray]) -> np.ndarray:
    pose = np.asarray(label["pose_m"], dtype=np.float32)
    if pose.shape == (1, 51):
        pose = pose[0]
    if pose.shape != (51,):
        raise ValueError(f"pose_m must have shape (1, 51) or (51,), got {pose.shape}")
    return pose


def is_valid_label(label: dict[str, np.ndarray], grasp_index: int) -> tuple[bool, list[str]]:
    reasons = []
    try:
        joints = hand_joints_from_label(label)
        if not np.isfinite(joints).all() or np.all(joints == -1):
            reasons.append("invalid_joint_3d")
    except (ValueError, IndexError):
        reasons.append("invalid_joint_3d_shape")
    try:
        mano_pose = mano_pose_from_label(label)
        if not np.isfinite(mano_pose).all() or np.all(mano_pose == 0):
            reasons.append("invalid_mano_pose")
    except ValueError:
        reasons.append("invalid_mano_pose_shape")
    try:
        object_pose = object_pose_from_label(label, grasp_index)
        if not np.isfinite(object_pose).all() or np.all(object_pose[:3] == 0):
            reasons.append("invalid_object_pose")
    except (ValueError, IndexError):
        reasons.append("invalid_object_pose_shape")
    return not reasons, reasons


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return np.asarray(vector / norm, dtype=np.float32)


def joint_frames_from_positions(joints: np.ndarray, hand_side: str = "right") -> np.ndarray:
    """Derive DexMV-compatible joint frames from a 21-joint skeleton.

    This geometry-only fallback preserves position and flexion cues but does not
    replace MANO forward kinematics when exact joint rotations are required.
    """
    points = np.asarray(joints, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError(f"joints must have shape (21, 3), got {points.shape}")

    across_palm = points[17] - points[5]
    toward_fingers = points[9] - points[0]
    palm_normal = _normalize(np.cross(across_palm, toward_fingers), np.array([0.0, 0.0, 1.0]))
    if hand_side.lower() == "left":
        palm_normal = -palm_normal

    frames = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], len(DEXMV_FRAME_JOINTS), axis=0)
    frame_lookup = {joint_index: frame_index for frame_index, joint_index in enumerate(DEXMV_FRAME_JOINTS)}

    wrist_x = _normalize(toward_fingers, np.array([1.0, 0.0, 0.0]))
    wrist_y = _normalize(np.cross(palm_normal, wrist_x), np.array([0.0, 1.0, 0.0]))
    wrist_z = _normalize(np.cross(wrist_x, wrist_y), palm_normal)
    frames[0, :3, :3] = np.column_stack((wrist_x, wrist_y, wrist_z))
    frames[0, :3, 3] = points[0]

    for chain in FINGER_CHAINS:
        finger_normal = np.cross(points[chain[1]] - points[chain[0]], points[chain[3]] - points[chain[1]])
        finger_normal = _normalize(finger_normal, palm_normal)
        if np.dot(finger_normal, palm_normal) < 0:
            finger_normal = -finger_normal
        for position, joint_index in enumerate(chain[:-1]):
            child_index = chain[position + 1]
            x_axis = _normalize(points[child_index] - points[joint_index], wrist_x)
            y_axis = _normalize(np.cross(finger_normal, x_axis), wrist_y)
            z_axis = _normalize(np.cross(x_axis, y_axis), finger_normal)
            frame_index = frame_lookup[joint_index]
            frames[frame_index, :3, :3] = np.column_stack((x_axis, y_axis, z_axis))
            frames[frame_index, :3, 3] = points[joint_index]
    return frames


def copy_or_link(source: str | Path, destination: str | Path, mode: str = "symlink") -> None:
    src = Path(source).resolve()
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        raise ValueError(f"unsupported transfer mode {mode!r}")


def frame_paths(sequence_dir: str | Path, serial: str, frame_index: int) -> dict[str, Path]:
    camera_dir = Path(sequence_dir) / serial
    return {
        "color": camera_dir / COLOR_PATTERN.format(frame_index),
        "depth": camera_dir / DEPTH_PATTERN.format(frame_index),
        "label": camera_dir / LABEL_PATTERN.format(frame_index),
    }


def project_points(points: np.ndarray, camera_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    matrix = np.asarray(camera_matrix, dtype=np.float32)
    depth = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (depth > 1e-6)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float32)
    camera_points = (matrix @ points[valid].T).T
    pixels[valid] = camera_points[:, :2] / camera_points[:, 2:3]
    return pixels, valid


def sample_evenly(indices: Iterable[int], count: int) -> list[int]:
    values = list(indices)
    if len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[index] for index in positions]
