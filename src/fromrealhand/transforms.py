from __future__ import annotations

from pathlib import Path

import numpy as np


def ensure_homogeneous(matrix: np.ndarray, *, name: str = "matrix") -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {arr.shape}")
    return arr


def load_matrix(path: str | Path) -> np.ndarray:
    matrix = np.load(Path(path), allow_pickle=True)
    if isinstance(matrix, np.ndarray) and matrix.shape == ():
        matrix = matrix.item()
    return ensure_homogeneous(matrix, name=str(path))


def transform_pose(transform: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return ensure_homogeneous(transform, name="transform") @ ensure_homogeneous(pose, name="pose")


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = ensure_homogeneous(transform, name="transform")
    pts = np.asarray(points, dtype=float)
    if pts.shape[-1] != 3:
        raise ValueError(f"points must end with dimension 3, got {pts.shape}")
    flat = pts.reshape(-1, 3)
    hom = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    out = (transform @ hom.T).T[:, :3]
    return out.reshape(pts.shape)
