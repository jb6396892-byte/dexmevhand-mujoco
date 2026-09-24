#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ.get("DEXMV_SIM", "/home/smgbro/dexmv-sim")) / "examples"))

import torch
import train as upstream
from mjrl.baselines.mlp_baseline import MLPBaseline
from tpi.core.config import assert_cfg, cfg


def available_baseline(*args, **kwargs):
    kwargs["use_gpu"] = torch.cuda.is_available()
    return MLPBaseline(*args, **kwargs)


def main() -> None:
    args = upstream.parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_cfg()
    cfg.freeze()
    upstream.MLPBaseline = available_baseline
    upstream.train()


if __name__ == "__main__":
    main()
