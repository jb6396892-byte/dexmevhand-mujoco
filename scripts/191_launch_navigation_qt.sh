#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
POLICY=${TABLETOP_POLICY:-/media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt}
exec bash "$ROOT/scripts/145_launch_tabletop_qt.sh" --allow-unvalidated-tabletop --navigation-mode \
  --checkpoint "$POLICY" --random-protocol "$ROOT/configs/tabletop-navigation-v5.json" "$@"
