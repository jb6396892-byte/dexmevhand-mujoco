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
    aligned_task_frame: bool = False,
) -> dict:
    configure_runtime_paths()
    from hand_imitation.kinematics.demonstration.relocation_demo import RelocationDemonstration

    class AlignedRelocationDemonstration(RelocationDemonstration):
        def strip_negative_origin(self, hand_sequence, object_sequence):
            return self.strip(hand_sequence, object_sequence)

        def hindsight_replay_sequence(self, hand_sequence, object_sequence, reference_object_name, init_object_lift=None):
            return hand_sequence, object_sequence

        def fetch_imitation_data(self, action_mean=None, action_range=None):
            limits = self.mjpy_model.jnt_range[:6]
            self.sim.data.qpos[:6] = np.clip(self.sim.data.qpos[:6], limits[:, 0], limits[:, 1])
            desired_qacc = self.sim.data.qacc.copy()
            self.sim.forward()
            self.sim.data.qacc[:] = desired_qacc
            return super().fetch_imitation_data(action_mean, action_range)


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

    player_class = AlignedRelocationDemonstration if aligned_task_frame else RelocationDemonstration
    player = player_class(has_renderer=has_renderer, object_name=object_name, object_scale=object_scale)
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
