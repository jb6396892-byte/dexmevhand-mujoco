#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf '%s\n' 'Video-faithful v3: saved controller actions, free-object physics. This is a demonstration, not a trained policy.'
exec bash "$ROOT/scripts/31_view_video_faithful_gpu.sh" \
  --geometry "$ROOT/data/processed/seq_dexycb_001/scene_fidelity_v2/retarget/geometry.npz" \
  --rollout "$ROOT/data/processed/seq_dexycb_001/scene_fidelity_v3/approach_feedback/best/diagnostic_rollout.pkl" "$@"
