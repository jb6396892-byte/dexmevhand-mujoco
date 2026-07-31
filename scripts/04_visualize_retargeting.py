#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import transforms3d

from fromrealhand.paths import configure_runtime_paths
from fromrealhand.pose_io import load_object_pose_sequence, load_retargeting_sequence
from fromrealhand.transforms import load_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize retargeted Adroit hand with object poses.")
    parser.add_argument("--retargeting", required=True, help="Retargeting pickle.")
    parser.add_argument("--object-dir", required=True, help="Directory with object pose .npy files.")
    parser.add_argument("--object-name", default="mug", help="DexMV object name.")
    parser.add_argument("--object-id", default=None, help="Optional key in dict-style object pose files.")
    parser.add_argument("--object-scale", type=float, default=0.8, help="DexMV object scale.")
    parser.add_argument("--camera-to-world", default=None, help="Optional 4x4 transform for object poses.")
    parser.add_argument("--skip-frame", type=int, default=0, help="Skip initial frames.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames.")
    parser.add_argument("--render-repeat", type=int, default=5, help="Renderer repeats per frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime_paths()
    from hand_imitation.kinematics.demonstration.relocation_demo import RelocationDemonstration

    camera_to_world = load_matrix(args.camera_to_world) if args.camera_to_world else None
    retargeting = load_retargeting_sequence(args.retargeting, skip_frame=args.skip_frame, limit=args.limit)
    object_poses = load_object_pose_sequence(
        args.object_dir,
        object_name=args.object_name,
        object_id=args.object_id,
        camera_to_world=camera_to_world,
        skip_frame=args.skip_frame,
        limit=args.limit,
    )
    data_len = min(len(retargeting), len(object_poses))
    player = RelocationDemonstration(has_renderer=True, object_name=args.object_name, object_scale=args.object_scale)
    player.filter.init_value(retargeting[0])
    dof = retargeting[0].shape[0]

    for i in range(data_len):
        pose = object_poses[i][args.object_name]
        robot_qpos = player.filter.next(retargeting[i])
        player.sim.data.qpos[:dof] = robot_qpos
        player.sim.data.qpos[player.object_trans_qpos_indices] = pose[:3, 3]
        player.sim.data.qpos[player.object_rot_qpos_indices] = transforms3d.quaternions.mat2quat(pose[:3, :3])
        player.sim.forward()
        for _ in range(args.render_repeat):
            player.render()


if __name__ == "__main__":
    main()
