#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fromrealhand.dexycb_io import scan_sequences, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan DexYCB sequences and filter by grasped object and hand side.")
    parser.add_argument("--root", required=True, help="DexYCB root containing subject, calibration and models folders.")
    parser.add_argument("--object", default="025_mug", help="Target YCB class name or numeric DexYCB id.")
    parser.add_argument("--hand-side", choices=("right", "left", ""), default="right", help="Required MANO hand side.")
    parser.add_argument("--include-present", action="store_true", help="Match sequences where the object is present, not necessarily grasped.")
    parser.add_argument("--output", required=True, help="Output .json or .csv manifest.")
    return parser.parse_args()


def write_csv(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "subject",
        "sequence",
        "relative_sequence",
        "hand_side",
        "num_frames",
        "fps",
        "grasped_object_id",
        "grasped_object",
        "serials",
        "extrinsics",
        "complete_cameras",
    )
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {name: record.get(name) for name in fields}
            row["serials"] = ",".join(record["serials"])
            row["complete_cameras"] = ",".join(
                serial for serial, counts in record["cameras"].items() if counts["complete"]
            )
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.root)
    if not dataset_root.is_dir():
        raise SystemExit(f"DexYCB root does not exist: {dataset_root}")

    records = scan_sequences(
        dataset_root,
        object_name=args.object,
        hand_side=args.hand_side,
        require_grasp_target=not args.include_present,
    )
    output = Path(args.output)
    if output.suffix.lower() == ".csv":
        write_csv(records, output)
    elif output.suffix.lower() == ".json":
        write_json(
            {
                "dataset_root": str(dataset_root.resolve()),
                "filters": {
                    "object": args.object,
                    "hand_side": args.hand_side or None,
                    "require_grasp_target": not args.include_present,
                },
                "count": len(records),
                "sequences": records,
            },
            output,
        )
    else:
        raise SystemExit("output extension must be .json or .csv")

    complete = sum(any(camera["complete"] for camera in record["cameras"].values()) for record in records)
    print(json.dumps({"matches": len(records), "with_complete_camera": complete, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
