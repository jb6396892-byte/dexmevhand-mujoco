#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-second}"
if [[ $# -gt 0 ]]; then shift; fi
case "$VERSION" in
  first)
    GEOMETRY="$ROOT/data/processed/seq_dexycb_001/scene_fidelity_v2/retarget/geometry.npz"
    ROLLOUT="$ROOT/data/processed/seq_dexycb_001/transport_v6/best/diagnostic_rollout.pkl"
    SOURCE="$ROOT/data/real_data/relocate_mug/seq_dexycb_001"
    LABEL='First-video transport feedback: saved expert actions in free-object physics.'
    ;;
  second)
    GEOMETRY="$ROOT/data/processed/seq_dexycb_002/retarget_v1/geometry.npz"
    ROLLOUT="$ROOT/data/processed/seq_dexycb_002/transport_v5/best/diagnostic_rollout.pkl"
    SOURCE="$ROOT/data/real_data/relocate_mug/seq_dexycb_002"
    LABEL='Second-video transport feedback: saved expert actions in free-object physics.'
    ;;
  *) printf '%s\n' 'Usage: bash scripts/57_view_transport_gpu.sh [first|second] [viewer arguments]' >&2; exit 2 ;;
esac
printf '%s\n' "$LABEL"
exec bash "$ROOT/scripts/31_view_video_faithful_gpu.sh" \
  --geometry "$GEOMETRY" --rollout "$ROLLOUT" --source-sequence "$SOURCE" "$@"
