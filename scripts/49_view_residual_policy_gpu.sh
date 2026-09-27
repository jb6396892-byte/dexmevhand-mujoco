#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf '%s\n' 'v4: saved actions from fixed-reference + learned-residual policy; free-object physics, not live network inference.'
exec bash "$ROOT/scripts/31_view_video_faithful_gpu.sh" \
  --geometry "$ROOT/data/processed/seq_dexycb_001/scene_fidelity_v2/retarget/geometry.npz" \
  --rollout "$ROOT/data/processed/seq_dexycb_001/learning_v4/residual_holdout/seed_20/diagnostic_rollout.pkl" "$@"
