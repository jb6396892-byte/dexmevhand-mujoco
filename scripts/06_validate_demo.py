#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fromrealhand.validation import format_stats, validate_demo_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a DexMV demonstration pickle.")
    parser.add_argument("demo_file", help="Demo pickle to validate.")
    parser.add_argument("--warn-saturation", type=float, default=0.2, help="Warn if action saturation fraction exceeds this.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = validate_demo_file(args.demo_file)
    print(format_stats(stats))
    bad = [item for item in stats if item.action_saturation_fraction > args.warn_saturation]
    if bad:
        names = ", ".join(item.trajectory_id for item in bad)
        raise SystemExit(f"warning: high action saturation in {names}")
    print("demo validation passed")


if __name__ == "__main__":
    main()
