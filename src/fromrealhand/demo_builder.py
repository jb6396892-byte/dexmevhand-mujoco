from __future__ import annotations

from pathlib import Path

import numpy as np

from fromrealhand.paths import configure_runtime_paths
from fromrealhand.pose_io import dump_pickle, load_object_pose_sequence, load_pickle, load_retargeting_sequence
from fromrealhand.transforms import load_matrix


def build_relocation_demo(
    *,
    retargeting_path: str | Path,
    object_dir: str | Path,
    output_path: str | Path,
    trajectory_id: str,
    object_name: str = "mug",
    object_id: str | None = None,
    object_scale: float = 0.8,
    camera_to_world_path: str | Path | None = None,
    skip_frame: int = 0,
    limit: int | None = None,
    append: bool = False,
    has_renderer: bool = False,
) -> dict:
    configure_runtime_paths()
    from hand_imitation.kinematics.demonstration.relocation_demo import RelocationDemonstration

    camera_to_world = load_matrix(camera_to_world_path) if camera_to_world_path else None
    retarget_qpos_seq = load_retargeting_sequence(retargeting_path, skip_frame=skip_frame, limit=limit)
    object_pose_seq = load_object_pose_sequence(
        object_dir,
        object_name=object_name,
        object_id=object_id,
        camera_to_world=camera_to_world,
        skip_frame=skip_frame,
        limit=limit,
    )

    data_len = min(len(retarget_qpos_seq), len(object_pose_seq))
    retarget_qpos_seq = retarget_qpos_seq[:data_len]
    object_pose_seq = object_pose_seq[:data_len]
    if data_len < 2:
        raise ValueError("need at least 2 aligned frames to build a demonstration")

    player = RelocationDemonstration(has_renderer=has_renderer, object_name=object_name, object_scale=object_scale)
    player.filter.init_value(np.asarray(retarget_qpos_seq[0]).copy())
    demo = player.play_hand_object_seq(retarget_qpos_seq, object_pose_seq, name=trajectory_id)
    if demo is None:
        raise RuntimeError("DexMV returned None while generating demonstration; check z position and pose alignment")

    output_path = Path(output_path)
    if append and output_path.exists():
        merged = load_pickle(output_path)
        if not isinstance(merged, dict):
            raise TypeError(f"existing demo file is not a dict: {output_path}")
    else:
        merged = {}
    merged[trajectory_id] = demo
    dump_pickle(merged, output_path)
    return merged
