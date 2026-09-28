#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf '%s\n' 'Second-video candidate: saved expert-controller actions, free-object physics; not a trained neural policy.'
exec bash "$ROOT/scripts/31_view_video_faithful_gpu.sh" \
  --geometry "$ROOT/data/processed/seq_dexycb_002/retarget_v1/geometry.npz" \
  --source-sequence "$ROOT/data/real_data/relocate_mug/seq_dexycb_002" \
  --rollout "$ROOT/data/processed/seq_dexycb_002/opposition_verified_v3/best/diagnostic_rollout.pkl" "$@"
