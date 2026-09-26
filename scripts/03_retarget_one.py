#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fromrealhand.paths import configure_runtime_paths
from fromrealhand.pose_io import natural_key, nearest_valid_indices
from fromrealhand.transforms import load_matrix, transform_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget one real hand-pose directory to Adroit hand qpos.")
    parser.add_argument("--hand-dir", required=True, help="Directory with results_global_*.npy and joints_*.npy.")
    parser.add_argument("--output", required=True, help="Output retargeting pickle.")
    parser.add_argument("--name", default="real_retargeting", help="Name shown by the optimizer.")
    parser.add_argument("--link-count", type=int, default=6, help="Number of palm/finger links to match.")
    parser.add_argument("--limit-global-pose", action="store_true", help="Constrain the six hand-root joints to MuJoCo limits.")
    parser.add_argument("--camera-to-world", default=None, help="Optional 4x4 transform applied to hand joints and frames.")
    parser.add_argument("--no-auto-transform", action="store_true", help="Do not infer calib/camera_to_world.npy from hand-dir.")
    parser.add_argument(
        "--invalid-policy",
        choices=("error", "nearest"),
        default="error",
        help="How to handle frames marked invalid or containing non-finite hand data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime_paths()
    from hand_imitation.env.utils.mjcf_utils import xml_path_completion
    from hand_imitation.kinematics.retargeting_optimizer import ChainMatchingPositionKinematicsRetargeting

    hand_dir = Path(args.hand_dir)
    camera_to_world_path = Path(args.camera_to_world) if args.camera_to_world else None
    if camera_to_world_path is None and not args.no_auto_transform:
        candidate = hand_dir.parent / "calib" / "camera_to_world.npy"
        if candidate.exists():
            camera_to_world_path = candidate
    camera_to_world = load_matrix(camera_to_world_path) if camera_to_world_path else None
    hand_pose_files = sorted([p for p in hand_dir.glob("*.npy") if "global" in p.name], key=natural_key)
    hand_joint_files = sorted([p for p in hand_dir.glob("*.npy") if "joint" in p.name], key=natural_key)
    seq_len = min(len(hand_pose_files), len(hand_joint_files))
    if seq_len == 0:
        raise SystemExit(f"no matched hand pose files found in {hand_dir}")

    link_names = ["palm", "thmiddle", "ffmiddle", "mfmiddle", "rfmiddle", "lfmiddle", "thtip", "fftip", "mftip", "rftip", "lftip"][: args.link_count]
    target_joint_index = [0, 2, 6, 10, 14, 18, 4, 8, 12, 16, 20][: args.link_count]
    solver = ChainMatchingPositionKinematicsRetargeting(
        xml_path_completion("adroit/adroit_relocate.xml"),
        link_names,
        has_joint_limits=True,
        has_global_pose_limits=args.limit_global_pose,
    )

    hand_frame_seq = []
    hand_joint_seq = []
    for i in range(seq_len):
        hand_frame = np.load(hand_pose_files[i])
        hand_joint = np.load(hand_joint_files[i])
        hand_frame_seq.append(hand_frame)
        hand_joint_seq.append(hand_joint)

    hand_frame_seq = np.stack(hand_frame_seq, axis=0)
    hand_joint_seq = np.stack(hand_joint_seq, axis=0)
    valid = np.isfinite(hand_frame_seq).all(axis=(1, 2, 3)) & np.isfinite(hand_joint_seq).all(axis=(1, 2))
    validity_path = hand_dir.parent / "valid_frames.npy"
    if validity_path.is_file():
        source_valid = np.load(validity_path).astype(bool).reshape(-1)
        if len(source_valid) != seq_len:
            raise SystemExit(f"{validity_path} has {len(source_valid)} entries, expected {seq_len}")
        valid &= source_valid

    repaired_frames = np.flatnonzero(~valid).tolist()
    if repaired_frames:
        if args.invalid_policy == "error":
            values = ", ".join(map(str, repaired_frames))
            raise SystemExit(f"invalid hand frames: {values}; pass --invalid-policy nearest to repair explicitly")
        source_indices = nearest_valid_indices(valid)
        hand_frame_seq = hand_frame_seq[source_indices]
        hand_joint_seq = hand_joint_seq[source_indices]

    if camera_to_world is not None:
        hand_frame_seq = camera_to_world[None, None, :, :] @ hand_frame_seq
        hand_joint_seq = transform_points(camera_to_world, hand_joint_seq)

    robot_joints = solver.retarget(
        hand_joint_seq[:, target_joint_index, :],
        hand_frame_seq,
        name=args.name,
        verbose=True,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(robot_joints, f)
    metadata = {
        "source_hand_dir": str(hand_dir.resolve()),
        "frame_count": seq_len,
        "invalid_policy": args.invalid_policy,
        "repaired_frames": repaired_frames,
        "camera_to_world": str(camera_to_world_path.resolve()) if camera_to_world_path else None,
        "limit_global_pose": args.limit_global_pose,
    }
    metadata_path = output.with_name(f"{output.stem}_meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    transform_note = f" using {camera_to_world_path}" if camera_to_world_path else " in source coordinates"
    repair_note = f"; repaired frames {repaired_frames}" if repaired_frames else ""
    print(f"wrote {output} with {len(robot_joints)} frames{transform_note}{repair_note}")


if __name__ == "__main__":
    main()
