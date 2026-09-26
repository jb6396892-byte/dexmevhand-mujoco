#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
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
    parser.add_argument("--fps", type=float, default=25.0, help="Playback and video frame rate; use a lower value for slow motion.")
    parser.add_argument("--loop", action="store_true", help="Replay in the window until interrupted.")
    parser.add_argument("--hold-seconds", type=float, default=5.0, help="Keep the final window frame open after a single playback.")
    parser.add_argument("--output-dir", default=None, help="Save offscreen keyframes instead of opening a window.")
    parser.add_argument("--output-video", default=None, help="Save a continuous MP4 instead of opening a window.")
    parser.add_argument("--video-repeat", type=int, default=5, help="Number of times to repeat the sequence in the MP4.")
    parser.add_argument("--sample-count", type=int, default=10, help="Number of offscreen keyframes to save.")
    parser.add_argument("--render-camera", default="frontview", help="MuJoCo camera used for offscreen rendering.")
    parser.add_argument("--hindsight", action="store_true", help="Apply the same origin normalization used for demo generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.hold_seconds < 0 or args.video_repeat < 1:
        raise SystemExit("--fps must be positive, --hold-seconds nonnegative, and --video-repeat at least 1")
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
    offscreen = args.output_dir is not None or args.output_video is not None
    player = RelocationDemonstration(has_renderer=not offscreen, object_name=args.object_name, object_scale=args.object_scale)
    if args.hindsight:
        retargeting, object_poses = player.strip_negative_origin(retargeting, object_poses)
        retargeting, object_poses = player.hindsight_replay_sequence(retargeting, object_poses, args.object_name)
        player.hind_sight_environment_model(retargeting, object_poses, args.object_name)
    data_len = min(len(retargeting), len(object_poses))
    if data_len == 0:
        raise SystemExit("no frames available after skipping initial frames")
    dof = retargeting[0].shape[0]
    selected = set()
    output_dir = None
    camera_id = None
    video_writer = None
    if offscreen:
        if args.output_dir is not None:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            selected = set(np.linspace(0, data_len - 1, min(args.sample_count, data_len)).round().astype(int).tolist())
        camera_id = player.sim.model.camera_name2id(args.render_camera)
        if player.sim._render_context_offscreen is None:
            context = MjRenderContextOffscreen(player.sim)
            player.sim.add_render_context(context)
        if args.output_video is not None:
            video_path = Path(args.output_video)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (640, 480))
            if not video_writer.isOpened():
                raise RuntimeError("could not open MP4 writer for {}".format(video_path))

    pass_count = 0
    try:
        while True:
            player.filter.init_value(retargeting[0].copy())
            playback_start = time.monotonic()
            for i in range(data_len):
                pose = object_poses[i][args.object_name]
                robot_qpos = player.filter.next(retargeting[i])
                player.sim.data.qpos[:dof] = robot_qpos
                player.sim.data.qpos[player.object_trans_qpos_indices] = pose[:3, 3]
                player.sim.data.qpos[player.object_rot_qpos_indices] = transforms3d.quaternions.mat2quat(pose[:3, :3])
                player.sim.forward()
                if offscreen and (video_writer is not None or (pass_count == 0 and i in selected)):
                    context = player.sim._render_context_offscreen
                    context.render(640, 480, camera_id=camera_id)
                    image = context.read_pixels(640, 480, depth=False)[::-1, :, :]
                    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if video_writer is not None:
                        video_writer.write(bgr)
                    if output_dir is not None and pass_count == 0 and i in selected:
                        cv2.imwrite(str(output_dir / f"{i + args.skip_frame:06d}.jpg"), bgr)
                elif not offscreen:
                    for _ in range(args.render_repeat):
                        player.render()
                    time.sleep(max(0.0, playback_start + (i + 1) / args.fps - time.monotonic()))

            pass_count += 1
            if offscreen:
                if video_writer is None or pass_count >= args.video_repeat:
                    break
            elif not args.loop:
                break

        if not offscreen:
            hold_until = time.monotonic() + args.hold_seconds
            while time.monotonic() < hold_until:
                player.render()
                time.sleep(0.04)
    except KeyboardInterrupt:
        pass
    finally:
        if video_writer is not None:
            video_writer.release()

    if output_dir is not None:
        print(f"wrote {len(selected)} offscreen frames to {output_dir}")
    if video_writer is not None:
        print(f"wrote {pass_count * data_len} frames to {args.output_video}")


if __name__ == "__main__":
    main()
