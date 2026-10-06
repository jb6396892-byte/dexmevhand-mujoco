#!/usr/bin/env python3
"""Inspect the scaffold only. Does not load weights or start simulation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.whole_table.config import inspect, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/whole-table-v1.json')
    args = parser.parse_args()
    print(json.dumps(inspect(load_config(args.config)), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
