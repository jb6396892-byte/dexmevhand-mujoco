#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fromrealhand.paths import configure_runtime_paths
from fromrealhand.pose_io import natural_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget one real hand-pose directory to Adroit hand qpos.")
    parser.add_argument("--hand-dir", required=True, help="Directory with results_global_*.npy and joints_*.npy.")
    parser.add_argument("--output", required=True, help="Output retargeting pickle.")
    parser.add_argument("--name", default="real_retargeting", help="Name shown by the optimizer.")
    parser.add_argument("--link-count", type=int, default=6, help="Number of palm/finger links to match.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime_paths()
    from hand_imitation.env.utils.mjcf_utils import xml_path_completion
    from hand_imitation.kinematics.retargeting_optimizer import ChainMatchingPositionKinematicsRetargeting

    hand_dir = Path(args.hand_dir)
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
        has_global_pose_limits=False,
    )

    hand_frame_seq = []
    hand_joint_seq = []
    for i in range(seq_len):
        hand_frame_seq.append(np.load(hand_pose_files[i]))
        hand_joint_seq.append(np.load(hand_joint_files[i]))

    hand_frame_seq = np.stack(hand_frame_seq, axis=0)
    hand_joint_seq = np.stack(hand_joint_seq, axis=0)
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
    print(f"wrote {output} with {len(robot_joints)} frames")


if __name__ == "__main__":
    main()
