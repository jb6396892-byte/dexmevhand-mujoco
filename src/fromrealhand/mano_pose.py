from __future__ import annotations

import cv2
import numpy as np
import torch
from manopth.manolayer import ManoLayer


# MANO kinematic order corresponding to DexYCB's wrist, thumb, index,
# middle, ring, and little finger MCP/PIP/DIP joints.
DEXYCB_TO_MANO = (0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9)


def make_mano_layer(model_root: str, side: str = "right") -> ManoLayer:
    return ManoLayer(
        mano_root=model_root,
        side=side,
        use_pca=True,
        ncomps=45,
        flat_hand_mean=False,
    )


def recover_mano_frames(
    layer: ManoLayer, pose_m: np.ndarray, betas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return 16 global MANO frames and 21 joints in camera coordinates (metres)."""
    pose = np.asarray(pose_m, dtype=np.float32).reshape(51)
    shape = np.asarray(betas, dtype=np.float32).reshape(10)
    if not np.isfinite(pose).all() or not np.isfinite(shape).all():
        raise ValueError("MANO pose and betas must be finite")

    with torch.no_grad():
        pose_tensor = torch.from_numpy(pose[:48].copy()).reshape(1, 48)
        beta_tensor = torch.from_numpy(shape.copy()).reshape(1, 10)
        trans_tensor = torch.from_numpy(pose[48:51].copy()).reshape(1, 3)
        _, joints_mm = layer(pose_tensor, beta_tensor, trans_tensor)
        joints = joints_mm[0].numpy() / 1000.0

    components = layer.th_selected_comps.numpy()
    hand_mean = layer.th_hands_mean.numpy().reshape(45)
    full_pose = np.concatenate((pose[:3], hand_mean + pose[3:48] @ components)).reshape(16, 3)
    template = layer.th_v_template.numpy()[0]
    shapedirs = layer.th_shapedirs.numpy()
    shaped_vertices = template + np.einsum("vck,k->vc", shapedirs, shape)
    rest_joints = layer.th_J_regressor.numpy() @ shaped_vertices

    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], 16, axis=0)
    for index in range(16):
        local_rotation = cv2.Rodrigues(full_pose[index])[0]
        parent = int(layer.kintree_parents[index]) if index else -1
        if index == 0:
            transforms[index, :3, :3] = local_rotation
            transforms[index, :3, 3] = rest_joints[0] + pose[48:51]
        else:
            local = np.eye(4)
            local[:3, :3] = local_rotation
            local[:3, 3] = rest_joints[index] - rest_joints[parent]
            transforms[index] = transforms[parent] @ local

    return transforms[list(DEXYCB_TO_MANO)].astype(np.float32), joints.astype(np.float32)


def rotation_error_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(a, -1, -2) @ b
    trace = np.trace(relative, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
