#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/31_view_video_faithful_gpu.sh" \
  --rollout "$ROOT/data/processed/seq_dexycb_001/surface_grasp_v2/best/diagnostic_rollout.pkl" "$@"
