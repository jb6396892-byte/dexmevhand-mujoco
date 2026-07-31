#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fromrealhand.demo_builder import build_relocation_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DexMV relocation demo from retargeting and object poses.")
    parser.add_argument("--sequence-dir", default=None, help="Sequence directory containing retargeting.pkl and object_pose/.")
    parser.add_argument("--retargeting", default=None, help="Retargeting pickle. Defaults to sequence-dir/retargeting.pkl.")
    parser.add_argument("--object-dir", default=None, help="Object pose dir. Defaults to sequence-dir/object_pose.")
    parser.add_argument("--output", default="data/demonstrations/relocate-mug-real.pkl", help="Output demo pickle.")
    parser.add_argument("--trajectory-id", default="seq_000", help="Trajectory key in output pickle.")
    parser.add_argument("--object-name", default="mug", help="DexMV object name.")
    parser.add_argument("--object-id", default=None, help="Optional key in dict-style object pose files.")
    parser.add_argument("--object-scale", type=float, default=0.8, help="DexMV object scale.")
    parser.add_argument("--camera-to-world", default=None, help="Optional 4x4 transform for object poses.")
    parser.add_argument("--skip-frame", type=int, default=0, help="Skip initial frames.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames.")
    parser.add_argument("--append", action="store_true", help="Append/update one trajectory in an existing output file.")
    parser.add_argument("--render", action="store_true", help="Render while generating demo.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_dir = Path(args.sequence_dir) if args.sequence_dir else None
    retargeting = Path(args.retargeting) if args.retargeting else sequence_dir / "retargeting.pkl"
    object_dir = Path(args.object_dir) if args.object_dir else sequence_dir / "object_pose"
    camera_to_world = args.camera_to_world
    if camera_to_world is None and sequence_dir is not None:
        candidate = sequence_dir / "calib" / "camera_to_world.npy"
        if candidate.exists():
            camera_to_world = str(candidate)

    output = Path(args.output)
    merged = build_relocation_demo(
        retargeting_path=retargeting,
        object_dir=object_dir,
        output_path=output,
        trajectory_id=args.trajectory_id,
        object_name=args.object_name,
        object_id=args.object_id,
        object_scale=args.object_scale,
        camera_to_world_path=camera_to_world,
        skip_frame=args.skip_frame,
        limit=args.limit,
        append=args.append,
        has_renderer=args.render,
    )
    print(f"wrote {output} with {len(merged)} trajector(y/ies)")


if __name__ == "__main__":
    main()
