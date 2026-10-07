#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$ROOT/scripts/137_tabletop_gpu.sh" "$ROOT/scripts/176_view_whole_table_f1.py" "$@"
