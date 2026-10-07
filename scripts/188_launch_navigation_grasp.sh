#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$ROOT/scripts/137_tabletop_gpu.sh" "$ROOT/scripts/187_view_navigation_grasp.py" "$@"
