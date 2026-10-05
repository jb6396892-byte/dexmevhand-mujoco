#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
POLICY=${TABLETOP_POLICY:-/media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt}
if [[ ! -f "$POLICY" ]]; then
  printf 'Learning checkpoint not found: %s\n' "$POLICY" >&2
  exit 1
fi
exec bash "$ROOT/scripts/145_launch_tabletop_qt.sh" --allow-unvalidated-tabletop --checkpoint "$POLICY" "$@"
