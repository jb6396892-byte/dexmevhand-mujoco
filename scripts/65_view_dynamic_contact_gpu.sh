#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf '%s\n' 'Second video v9: optimized expert actions, free-cup physics; not a trained policy.'
exec bash "$ROOT/scripts/31_view_video_faithful_gpu.sh" \
  --geometry "$ROOT/data/processed/seq_dexycb_002/retarget_v1/geometry.npz" \
  --rollout "$ROOT/data/processed/seq_dexycb_002/dynamic_contact_v9_verified/nominal/diagnostic_rollout.pkl" \
  --source-sequence "$ROOT/data/real_data/relocate_mug/seq_dexycb_002" "$@"
