#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from fromrealhand.dexycb_io import FINGER_CHAINS, load_label, project_points, sample_evenly, write_json


COLORS = (
    (80, 180, 255),
    (70, 210, 120),
    (255, 160, 70),
    (220, 90, 190),
    (90, 110, 255),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate converted DexYCB camera geometry with RGB overlays and trajectory plots.")
    parser.add_argument("--sequence-dir", required=True, help="Converted trajectory directory.")
    parser.add_argument("--output", required=True, help="Output report directory.")
    parser.add_argument("--sample-count", type=int, default=10, help="Number of evenly spaced valid frames to render.")
    parser.add_argument("--mesh", default=None, help="Optional YCB OBJ mesh to project on RGB frames.")
    parser.add_argument("--mesh-points", type=int, default=1500, help="Maximum number of mesh vertices drawn per frame.")
    return parser.parse_args()


def draw_skeleton(image: np.ndarray, pixels: np.ndarray, visible: np.ndarray) -> None:
    wrist = tuple(np.round(pixels[0]).astype(int)) if visible[0] else None
    if wrist is not None:
        cv2.circle(image, wrist, 5, (255, 255, 255), -1, cv2.LINE_AA)
    for color, chain in zip(COLORS, FINGER_CHAINS):
        if wrist is not None and visible[chain[0]]:
            cv2.line(image, wrist, tuple(np.round(pixels[chain[0]]).astype(int)), color, 2, cv2.LINE_AA)
        for start, end in zip(chain[:-1], chain[1:]):
            if visible[start] and visible[end]:
                cv2.line(
                    image,
                    tuple(np.round(pixels[start]).astype(int)),
                    tuple(np.round(pixels[end]).astype(int)),
                    color,
                    2,
                    cv2.LINE_AA,
                )
        for joint_index in chain:
            if visible[joint_index]:
                cv2.circle(image, tuple(np.round(pixels[joint_index]).astype(int)), 3, color, -1, cv2.LINE_AA)


def draw_object_axes(image: np.ndarray, pose: np.ndarray, camera_matrix: np.ndarray, axis_length: float = 0.05) -> None:
    local = np.array(
        [[0.0, 0.0, 0.0], [axis_length, 0.0, 0.0], [0.0, axis_length, 0.0], [0.0, 0.0, axis_length]],
        dtype=np.float32,
    )
    camera = (pose[:3, :3] @ local.T).T + pose[:3, 3]
    pixels, visible = project_points(camera, camera_matrix)
    if not visible.all():
        return
    origin = tuple(np.round(pixels[0]).astype(int))
    for endpoint, color in zip(pixels[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.arrowedLine(image, origin, tuple(np.round(endpoint).astype(int)), color, 2, cv2.LINE_AA, tipLength=0.15)


def load_mesh_vertices(path: str | None, limit: int) -> np.ndarray | None:
    if not path:
        return None
    import trimesh

    mesh = trimesh.load(Path(path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) > limit:
        positions = np.linspace(0, len(vertices) - 1, limit).round().astype(int)
        vertices = vertices[positions]
    return vertices


def draw_mesh_points(image: np.ndarray, vertices: np.ndarray, pose: np.ndarray, camera_matrix: np.ndarray) -> None:
    camera = (pose[:3, :3] @ vertices.T).T + pose[:3, 3]
    pixels, valid = project_points(camera, camera_matrix)
    height, width = image.shape[:2]
    inside = valid & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    overlay = image.copy()
    for pixel in pixels[inside]:
        cv2.circle(overlay, tuple(np.round(pixel).astype(int)), 1, (40, 220, 240), -1)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0.0, image)


def movement_report(values: np.ndarray) -> dict[str, object]:
    if len(values) == 0:
        return {"count": 0, "min": None, "max": None, "max_step": None, "mean_step": None}
    steps = np.linalg.norm(np.diff(values, axis=0), axis=1) if len(values) > 1 else np.array([], dtype=float)
    return {
        "count": int(len(values)),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "max_step": float(steps.max()) if len(steps) else 0.0,
        "mean_step": float(steps.mean()) if len(steps) else 0.0,
    }


def save_trajectory_plot(wrists: np.ndarray, objects: np.ndarray, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8, 6))
    axes = figure.add_subplot(111, projection="3d")
    if len(wrists):
        axes.plot(wrists[:, 0], wrists[:, 1], wrists[:, 2], label="wrist", color="#176b4a")
        axes.scatter(*wrists[0], color="#176b4a", marker="o")
    if len(objects):
        axes.plot(objects[:, 0], objects[:, 1], objects[:, 2], label="mug", color="#b74e3c")
        axes.scatter(*objects[0], color="#b74e3c", marker="o")
    axes.set_xlabel("camera x (m)")
    axes.set_ylabel("camera y (m)")
    axes.set_zlabel("camera z (m)")
    axes.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    sequence_dir = Path(args.sequence_dir)
    output = Path(args.output)
    overlay_dir = output / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    camera_matrix = np.load(sequence_dir / "calib" / "camera_matrix.npy")
    valid_frames = np.load(sequence_dir / "valid_frames.npy").astype(bool)
    frame_indices = list(range(len(valid_frames)))
    selected = sample_evenly([index for index in frame_indices if valid_frames[index]], args.sample_count)
    if not selected:
        raise SystemExit("sequence has no valid frames to visualize")
    mesh_vertices = load_mesh_vertices(args.mesh, args.mesh_points)

    wrists = []
    object_positions = []
    reprojection_errors = []
    for frame_index in frame_indices:
        joints = np.load(sequence_dir / "hand_pose" / f"joints_{frame_index:06d}.npy")
        object_pose = np.load(sequence_dir / "object_pose" / f"{frame_index:06d}.npy")
        if valid_frames[frame_index]:
            wrists.append(joints[0])
            object_positions.append(object_pose[:3, 3])

        source_label = sequence_dir / "source_labels" / f"{frame_index:06d}.npz"
        if source_label.exists():
            label = load_label(source_label)
            annotated = np.asarray(label["joint_2d"], dtype=np.float32).reshape(-1, 21, 2)[0]
            projected, projected_valid = project_points(joints, camera_matrix)
            annotated_valid = np.isfinite(annotated).all(axis=1) & np.all(annotated >= 0, axis=1)
            mask = projected_valid & annotated_valid
            if mask.any():
                reprojection_errors.extend(np.linalg.norm(projected[mask] - annotated[mask], axis=1).tolist())

        if frame_index not in selected:
            continue
        image = cv2.imread(str(sequence_dir / "rgb" / f"{frame_index:06d}.jpg"), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"failed to read RGB frame {frame_index:06d}")
        pixels, visible = project_points(joints, camera_matrix)
        height, width = image.shape[:2]
        visible &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        if mesh_vertices is not None:
            draw_mesh_points(image, mesh_vertices, object_pose, camera_matrix)
        draw_skeleton(image, pixels, visible)
        draw_object_axes(image, object_pose, camera_matrix)
        cv2.putText(image, f"frame {frame_index:06d}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(overlay_dir / f"{frame_index:06d}.jpg"), image)

    wrist_array = np.asarray(wrists, dtype=np.float32).reshape(-1, 3)
    object_array = np.asarray(object_positions, dtype=np.float32).reshape(-1, 3)
    save_trajectory_plot(wrist_array, object_array, output / "trajectory_3d.png")
    error_array = np.asarray(reprojection_errors, dtype=float)
    report = {
        "sequence_dir": str(sequence_dir.resolve()),
        "frame_count": len(frame_indices),
        "valid_frame_count": int(valid_frames.sum()),
        "valid_frame_ratio": float(valid_frames.mean()),
        "sampled_overlay_frames": selected,
        "wrist_trajectory_m": movement_report(wrist_array),
        "object_trajectory_m": movement_report(object_array),
        "joint_reprojection_error_px": {
            "count": int(len(error_array)),
            "mean": float(error_array.mean()) if len(error_array) else None,
            "max": float(error_array.max()) if len(error_array) else None,
        },
        "mesh_projected": args.mesh,
    }
    write_json(report, output / "trajectory_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
