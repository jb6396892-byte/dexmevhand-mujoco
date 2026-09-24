#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fromrealhand.dexycb_io import (
    copy_or_link,
    frame_paths,
    grasped_object_id,
    hand_joints_from_label,
    is_valid_label,
    joint_frames_from_positions,
    load_camera_to_world,
    load_intrinsics,
    load_label,
    mano_pose_from_label,
    object_pose_from_label,
    read_yaml,
    sequence_hand_side,
    write_json,
    YCB_CLASSES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert one DexYCB sequence/camera into the fromrealhand format.")
    parser.add_argument("--root", required=True, help="DexYCB dataset root.")
    parser.add_argument("--sequence", required=True, help="Relative sequence path SUBJECT/SEQUENCE.")
    parser.add_argument("--camera", required=True, help="RealSense camera serial from meta.yml.")
    parser.add_argument("--output", required=True, help="Output trajectory directory.")
    parser.add_argument("--transfer", choices=("symlink", "copy"), default="symlink", help="How to expose RGB/depth files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise SystemExit(f"output is not empty: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)
    for name in ("rgb", "depth", "calib", "hand_pose", "object_pose", "source_labels", "annotations"):
        (output / name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.root).resolve()
    sequence_dir = dataset_root / args.sequence
    meta_path = sequence_dir / "meta.yml"
    if not meta_path.is_file():
        raise SystemExit(f"missing DexYCB sequence metadata: {meta_path}")
    source_meta = read_yaml(meta_path)
    serials = [str(value) for value in source_meta.get("serials", [])]
    if args.camera not in serials:
        raise SystemExit(f"camera {args.camera} is not listed in {meta_path}; available: {', '.join(serials)}")

    num_frames = int(source_meta.get("num_frames", 0))
    if num_frames <= 0:
        raise SystemExit(f"invalid num_frames in {meta_path}: {num_frames}")
    grasp_index = source_meta.get("ycb_grasp_ind")
    object_id = grasped_object_id(source_meta)
    if not isinstance(grasp_index, int) or object_id is None:
        raise SystemExit(f"invalid ycb_grasp_ind/ycb_ids in {meta_path}")

    output = Path(args.output)
    prepare_output(output, args.overwrite)
    camera_matrix, distortion, intrinsics_path = load_intrinsics(dataset_root, args.camera)
    camera_to_world, world_camera, extrinsics_path = load_camera_to_world(dataset_root, source_meta, args.camera)
    np.save(output / "calib" / "camera_matrix.npy", camera_matrix)
    np.save(output / "calib" / "dist_coeffs.npy", distortion)
    np.save(output / "calib" / "camera_to_world.npy", camera_to_world)

    valid_frames = np.zeros(num_frames, dtype=bool)
    invalid_reasons: Counter[str] = Counter()
    frame_mapping = []
    hand_side = sequence_hand_side(source_meta) or "unknown"
    for frame_index in range(num_frames):
        sources = frame_paths(sequence_dir, args.camera, frame_index)
        missing = [name for name, path in sources.items() if not path.is_file()]
        if missing:
            raise SystemExit(f"frame {frame_index:06d} is missing required files: {', '.join(missing)}")

        label = load_label(sources["label"])
        valid, reasons = is_valid_label(label, grasp_index)
        valid_frames[frame_index] = valid
        invalid_reasons.update(reasons)

        joints = hand_joints_from_label(label)
        joints_are_usable = np.isfinite(joints).all() and not np.all(joints == -1)
        frames = (
            joint_frames_from_positions(joints, hand_side=hand_side)
            if joints_are_usable
            else np.full((16, 4, 4), np.nan, dtype=np.float32)
        )
        object_pose = object_pose_from_label(label, grasp_index)
        mano_pose = mano_pose_from_label(label)

        output_index = frame_index
        copy_or_link(sources["color"], output / "rgb" / f"{output_index:06d}.jpg", mode=args.transfer)
        copy_or_link(sources["depth"], output / "depth" / f"{output_index:06d}.png", mode=args.transfer)
        copy_or_link(sources["label"], output / "source_labels" / f"{output_index:06d}.npz", mode=args.transfer)
        np.save(output / "hand_pose" / f"joints_{output_index:06d}.npy", joints)
        np.save(output / "hand_pose" / f"results_global_{output_index:06d}.npy", frames)
        np.save(output / "hand_pose" / f"mano_pose_{output_index:06d}.npy", mano_pose)
        np.save(output / "object_pose" / f"{output_index:06d}.npy", object_pose)
        frame_mapping.append({"output_frame": output_index, "source_frame": frame_index, "valid": valid, "invalid_reasons": reasons})

    np.save(output / "valid_frames.npy", valid_frames)
    write_json(frame_mapping, output / "frame_mapping.json")
    meta = {
        "sequence_id": output.name,
        "source": "DexYCB",
        "source_sequence": args.sequence,
        "source_meta": str(meta_path),
        "camera_id": args.camera,
        "world_camera_id": world_camera,
        "fps": float(source_meta.get("fps", 30.0)),
        "num_frames": num_frames,
        "valid_frames": int(valid_frames.sum()),
        "object_name": "mug" if object_id == 14 else YCB_CLASSES.get(object_id, str(object_id)),
        "object_model": YCB_CLASSES.get(object_id),
        "object_id": object_id,
        "object_scale": 0.8,
        "hand_side": hand_side,
        "pose_frame": "camera",
        "world_frame": f"DexYCB AprilTag table frame via master camera {world_camera}",
        "translation_unit": "meter",
        "camera_to_world_convention": "point_world = camera_to_world @ point_camera",
        "intrinsics_source": str(intrinsics_path),
        "extrinsics_source": str(extrinsics_path),
        "hand_frame_source": "geometry derived from joint_3d; validate against MANO before full training",
        "transfer_mode": args.transfer,
        "invalid_reason_counts": dict(invalid_reasons),
    }
    write_json(meta, output / "meta.json")
    print(f"converted {args.sequence}/{args.camera}: {int(valid_frames.sum())}/{num_frames} valid frames -> {output}")


if __name__ == "__main__":
    main()
