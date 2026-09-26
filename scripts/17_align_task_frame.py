#!/usr/bin/env python3
"""Align one DexYCB object frame to the MuJoCo relocation task frame."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fromrealhand.pose_io import pose_from_payload, sorted_npy_files
from fromrealhand.transforms import align_object_origin, load_matrix, transform_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--reference-index", type=int, required=True, help="Video frame used as the new demonstration start.")
    parser.add_argument("--target-xyz", type=float, nargs=3, default=(-0.0625, 0.0, 0.04065))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted_npy_files(args.sequence_dir / "object_pose")
    if not 0 <= args.reference_index < len(files):
        parser.error("reference-index must be within the object-pose sequence")
    camera_to_world = load_matrix(args.sequence_dir / "calib" / "camera_to_world.npy")
    object_pose = pose_from_payload(np.load(files[args.reference_index], allow_pickle=True))
    aligned = align_object_origin(camera_to_world, object_pose, args.target_xyz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, aligned)
    print("source object xyz:", transform_pose(camera_to_world, object_pose)[:3, 3])
    print("aligned object xyz:", transform_pose(aligned, object_pose)[:3, 3])
    print("wrote", args.output)


if __name__ == "__main__":
    main()
