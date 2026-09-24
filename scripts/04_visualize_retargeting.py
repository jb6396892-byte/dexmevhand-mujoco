#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import transforms3d
import cv2

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
    parser.add_argument("--output-dir", default=None, help="Save offscreen keyframes instead of opening a window.")
    parser.add_argument("--sample-count", type=int, default=10, help="Number of offscreen keyframes to save.")
    parser.add_argument("--render-camera", default="frontview", help="MuJoCo camera used for offscreen rendering.")
    parser.add_argument("--hindsight", action="store_true", help="Apply the same origin normalization used for demo generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime_paths()
    from hand_imitation.kinematics.demonstration.relocation_demo import RelocationDemonstration
    from mujoco_py import MjRenderContextOffscreen

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
    offscreen = args.output_dir is not None
    player = RelocationDemonstration(has_renderer=not offscreen, object_name=args.object_name, object_scale=args.object_scale)
    if args.hindsight:
        retargeting, object_poses = player.strip_negative_origin(retargeting, object_poses)
        retargeting, object_poses = player.hindsight_replay_sequence(retargeting, object_poses, args.object_name)
        player.hind_sight_environment_model(retargeting, object_poses, args.object_name)
    data_len = min(len(retargeting), len(object_poses))
    player.filter.init_value(retargeting[0])
    dof = retargeting[0].shape[0]
    selected = set()
    output_dir = None
    camera_id = None
    if offscreen:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        selected = set(np.linspace(0, data_len - 1, min(args.sample_count, data_len)).round().astype(int).tolist())
        camera_id = player.sim.model.camera_name2id(args.render_camera)
        if player.sim._render_context_offscreen is None:
            context = MjRenderContextOffscreen(player.sim)
            player.sim.add_render_context(context)

    for i in range(data_len):
        pose = object_poses[i][args.object_name]
        robot_qpos = player.filter.next(retargeting[i])
        player.sim.data.qpos[:dof] = robot_qpos
        player.sim.data.qpos[player.object_trans_qpos_indices] = pose[:3, 3]
        player.sim.data.qpos[player.object_rot_qpos_indices] = transforms3d.quaternions.mat2quat(pose[:3, :3])
        player.sim.forward()
        if offscreen and i in selected:
            context = player.sim._render_context_offscreen
            context.render(640, 480, camera_id=camera_id)
            image = context.read_pixels(640, 480, depth=False)[::-1, :, :]
            cv2.imwrite(str(output_dir / f"{i + args.skip_frame:06d}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        elif not offscreen:
            for _ in range(args.render_repeat):
                player.render()

    if offscreen:
        print(f"wrote {len(selected)} offscreen frames to {output_dir}")


if __name__ == "__main__":
    main()
