#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract video frames for real-hand data preparation.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output frame directory.")
    parser.add_argument("--every", type=int, default=1, help="Save every Nth frame.")
    parser.add_argument("--start", type=int, default=0, help="First frame index to consider.")
    parser.add_argument("--end", type=int, default=None, help="Stop before this frame index.")
    parser.add_argument("--prefix", default="frame", help="Output filename prefix.")
    parser.add_argument("--ext", default="jpg", choices=["jpg", "png"], help="Output image extension.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python is required; run this in the dexmv conda environment") from exc

    video = Path(args.video)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"failed to open video: {video}")

    manifest_path = output / "frames.csv"
    saved = 0
    frame_idx = 0
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["saved_index", "source_frame", "filename"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx >= args.start and (args.end is None or frame_idx < args.end):
                if (frame_idx - args.start) % args.every == 0:
                    name = f"{args.prefix}_{saved:06d}.{args.ext}"
                    path = output / name
                    cv2.imwrite(str(path), frame)
                    writer.writerow([saved, frame_idx, name])
                    saved += 1
            frame_idx += 1
            if args.end is not None and frame_idx >= args.end:
                break
    cap.release()
    print(f"saved {saved} frames to {output}")


if __name__ == "__main__":
    main()
