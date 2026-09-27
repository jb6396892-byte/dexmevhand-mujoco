#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf '%s\n' 'Experimental scene candidate: full admission and video fidelity have NOT passed.'
exec bash "$ROOT/scripts/31_view_video_faithful_gpu.sh" \
  --geometry "$ROOT/data/processed/seq_dexycb_001/scene_fidelity_v2/retarget/geometry.npz" \
  --rollout "$ROOT/data/processed/seq_dexycb_001/scene_fidelity_v2/slower_soft_control/best/diagnostic_rollout.pkl" "$@"
