#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a real trajectory directory template.")
    parser.add_argument("--root", default="data/real_data/relocate_mug", help="Trajectory root directory.")
    parser.add_argument("--seq", default="seq_000", help="Sequence id.")
    parser.add_argument("--object-name", default="mug", help="Object name used by DexMV.")
    parser.add_argument("--fps", type=float, default=30.0, help="Original video FPS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seq_dir = Path(args.root) / args.seq
    for name in ["rgb", "calib", "hand_pose", "object_pose"]:
        (seq_dir / name).mkdir(parents=True, exist_ok=True)

    meta_path = seq_dir / "meta.json"
    if not meta_path.exists():
        meta = {
            "sequence_id": args.seq,
            "task": "relocate",
            "object_name": args.object_name,
            "fps": args.fps,
            "camera_to_world": "calib/camera_to_world.npy",
            "notes": "Fill hand_pose/ and object_pose/ with external pose estimation results.",
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"prepared {seq_dir}")


if __name__ == "__main__":
    main()
