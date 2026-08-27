#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from manopth.manolayer import ManoLayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate licensed MANO v1.2 models with manopth.")
    parser.add_argument(
        "--mano-root",
        default=os.environ.get("MANO_ROOT", "/media/smgbro/shared/DexYCB/mano/models"),
        help="Directory containing MANO_RIGHT.pkl and MANO_LEFT.pkl.",
    )
    return parser.parse_args()


def validate_side(model_root: Path, side: str) -> dict[str, object]:
    model_file = model_root / f"MANO_{side.upper()}.pkl"
    if not model_file.is_file():
        raise FileNotFoundError(f"missing MANO model: {model_file}")

    layer = ManoLayer(
        mano_root=str(model_root),
        side=side,
        use_pca=True,
        ncomps=45,
        flat_hand_mean=False,
    )
    pose = torch.zeros(1, 48, dtype=torch.float32)
    shape = torch.zeros(1, 10, dtype=torch.float32)
    with torch.no_grad():
        vertices, joints = layer(pose, shape)

    if vertices.shape != (1, 778, 3):
        raise ValueError(f"unexpected {side} vertex shape: {tuple(vertices.shape)}")
    if joints.shape != (1, 21, 3):
        raise ValueError(f"unexpected {side} joint shape: {tuple(joints.shape)}")
    if not torch.isfinite(vertices).all() or not torch.isfinite(joints).all():
        raise ValueError(f"{side} MANO output contains non-finite values")

    return {
        "model": str(model_file),
        "vertices": list(vertices.shape),
        "joints": list(joints.shape),
        "finite": True,
    }


def main() -> None:
    args = parse_args()
    model_root = Path(args.mano_root).expanduser().resolve()
    report = {
        "mano_root": str(model_root),
        "right": validate_side(model_root, "right"),
        "left": validate_side(model_root, "left"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
