from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

from fromrealhand.transforms import ensure_homogeneous, transform_pose


def natural_key(path: str | Path) -> list[Any]:
    text = str(path)
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def sorted_npy_files(directory: str | Path) -> list[Path]:
    root = Path(directory)
    return sorted(root.glob("*.npy"), key=natural_key)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def dump_pickle(data: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(data, f)


def _unwrap_numpy_payload(payload: Any) -> Any:
    if isinstance(payload, np.ndarray) and payload.shape == ():
        return payload.item()
    return payload


def pose_from_payload(payload: Any, *, object_name: str = "mug", object_id: str | None = None) -> np.ndarray:
    payload = _unwrap_numpy_payload(payload)
    if isinstance(payload, dict):
        if object_id is not None and object_id in payload:
            payload = payload[object_id]
        elif object_name in payload:
            payload = payload[object_name]
        elif len(payload) == 1:
            payload = next(iter(payload.values()))
        else:
            keys = ", ".join(map(str, payload.keys()))
            raise KeyError(f"object pose dict has no key for {object_name!r} or {object_id!r}; keys: {keys}")
    return ensure_homogeneous(np.asarray(payload), name="object pose")


def load_object_pose_sequence(
    object_dir: str | Path,
    *,
    object_name: str = "mug",
    object_id: str | None = None,
    camera_to_world: np.ndarray | None = None,
    skip_frame: int = 0,
    limit: int | None = None,
) -> list[dict[str, np.ndarray]]:
    files = sorted_npy_files(object_dir)[skip_frame:]
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no .npy object pose files found in {object_dir}")

    sequence = []
    for file in files:
        payload = np.load(file, allow_pickle=True)
        pose = pose_from_payload(payload, object_name=object_name, object_id=object_id)
        if camera_to_world is not None:
            pose = transform_pose(camera_to_world, pose)
        sequence.append({object_name: pose})
    return sequence


def load_retargeting_sequence(path: str | Path, *, skip_frame: int = 0, limit: int | None = None) -> list[np.ndarray]:
    data = load_pickle(path)
    if isinstance(data, np.ndarray):
        seq = [np.asarray(row).copy() for row in data]
    else:
        seq = [np.asarray(row).copy() for row in list(data)]
    seq = seq[skip_frame:]
    if limit is not None:
        seq = seq[:limit]
    if not seq:
        raise ValueError(f"retargeting sequence is empty: {path}")
    return seq


def count_hand_pose_pairs(hand_dir: str | Path) -> tuple[int, int]:
    root = Path(hand_dir)
    global_files = [p for p in root.glob("*.npy") if "global" in p.name]
    joint_files = [p for p in root.glob("*.npy") if "joint" in p.name]
    return len(global_files), len(joint_files)
