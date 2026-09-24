#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fromrealhand.dexycb_io import read_yaml, write_json
from fromrealhand.mano_pose import make_mano_layer, recover_mano_frames, rotation_error_degrees


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover DexYCB MANO rotations and compare with geometry-derived frames.")
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument("--dataset-root", default=os.environ.get("DEXYCB_ROOT", "/media/smgbro/shared/DexYCB/dataset"))
    parser.add_argument("--mano-root", default=os.environ.get("MANO_ROOT", "/media/smgbro/shared/DexYCB/mano/models"))
    parser.add_argument("--max-mean-mm", type=float, default=5.0)
    parser.add_argument("--max-joint-mm", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_dir = Path(args.sequence_dir)
    source_meta = json.loads((sequence_dir / "meta.json").read_text(encoding="utf-8"))
    if source_meta.get("source") != "DexYCB":
        raise SystemExit("expected a converted DexYCB sequence")
    original_meta = read_yaml(Path(args.dataset_root) / source_meta["source_sequence"] / "meta.yml")
    calib_names = original_meta["mano_calib"]
    if len(calib_names) != 1 or source_meta["hand_side"] != "right":
        raise SystemExit("this command requires one calibrated right hand")
    betas_file = Path(args.dataset_root) / "calibration" / f"mano_{calib_names[0]}" / "mano.yml"
    betas = np.asarray(read_yaml(betas_file)["betas"], dtype=np.float32)
    layer = make_mano_layer(args.mano_root, "right")

    valid = np.load(sequence_dir / "valid_frames.npy").astype(bool)
    geometry_dir = sequence_dir / "hand_pose"
    output_dir = sequence_dir / "hand_pose_mano"
    output_dir.mkdir(exist_ok=True)
    joint_errors = []
    frame_position_errors = []
    raw_angles = []
    relative_angles = []
    frame_rotations = []
    geometry_rotations = []

    for index in np.flatnonzero(valid):
        frame_id = f"{index:06d}"
        pose = np.load(geometry_dir / f"mano_pose_{frame_id}.npy")
        frames, reconstructed_joints = recover_mano_frames(layer, pose, betas)
        labeled_joints = np.load(geometry_dir / f"joints_{frame_id}.npy")
        geometry = np.load(geometry_dir / f"results_global_{frame_id}.npy")
        joint_errors.extend(np.linalg.norm(reconstructed_joints - labeled_joints, axis=1).tolist())
        frame_position_errors.extend(np.linalg.norm(frames[:, :3, 3] - geometry[:, :3, 3], axis=1).tolist())
        raw_angles.extend(rotation_error_degrees(geometry[:, :3, :3], frames[:, :3, :3]).tolist())
        frame_rotations.append(frames[:, :3, :3])
        geometry_rotations.append(geometry[:, :3, :3])
        np.save(output_dir / f"results_global_{frame_id}.npy", frames)

    # Compare world-frame motion so fixed local-axis conventions do not
    # contribute to the reported motion difference.
    frame_rotations = np.asarray(frame_rotations)
    geometry_rotations = np.asarray(geometry_rotations)
    if len(frame_rotations) > 1:
        geo_motion = geometry_rotations @ np.swapaxes(geometry_rotations[0], -1, -2)[None]
        mano_motion = frame_rotations @ np.swapaxes(frame_rotations[0], -1, -2)[None]
        relative_angles = rotation_error_degrees(geo_motion, mano_motion).reshape(-1).tolist()

    report = {
        "sequence": source_meta["source_sequence"],
        "mano_model": str(Path(args.mano_root) / "MANO_RIGHT.pkl"),
        "betas_file": str(betas_file),
        "valid_frames": int(valid.sum()),
        "joint_error_mean_mm": float(np.mean(joint_errors) * 1000),
        "joint_error_max_mm": float(np.max(joint_errors) * 1000),
        "frame_position_error_mean_mm": float(np.mean(frame_position_errors) * 1000),
        "frame_position_error_max_mm": float(np.max(frame_position_errors) * 1000),
        "raw_rotation_difference_mean_deg": float(np.mean(raw_angles)),
        "raw_rotation_difference_p95_deg": float(np.percentile(raw_angles, 95)),
        "motion_rotation_difference_mean_deg": float(np.mean(relative_angles)) if relative_angles else None,
        "motion_rotation_difference_p95_deg": float(np.percentile(relative_angles, 95)) if relative_angles else None,
    }
    report["passed"] = (
        report["joint_error_mean_mm"] <= args.max_mean_mm
        and report["joint_error_max_mm"] <= args.max_joint_mm
        and report["frame_position_error_max_mm"] <= args.max_joint_mm
    )
    write_json(report, sequence_dir / "mano_comparison.json")
    print(report)
    if not report["passed"]:
        raise SystemExit("MANO reconstruction failed the label-consistency gate")

    for index in range(len(valid)):
        frame_id = f"{index:06d}"
        if not valid[index]:
            np.save(output_dir / f"results_global_{frame_id}.npy", np.full((16, 4, 4), np.nan, dtype=np.float32))
        joint_link = output_dir / f"joints_{frame_id}.npy"
        if not joint_link.exists():
            joint_link.symlink_to((geometry_dir / joint_link.name).resolve())


if __name__ == "__main__":
    main()
