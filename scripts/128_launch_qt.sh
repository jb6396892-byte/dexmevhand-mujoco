#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STUDY_ROOT=${STUDY_ROOT:-/media/smgbro/shared/lora}
GUI="$STUDY_ROOT/language/gui-runtime"
if [[ -z "${DISPLAY:-}" ]]; then
  printf '%s\n' 'No X11 DISPLAY. Launch from a terminal on the Ubuntu desktop.' >&2
  exit 1
fi
if [[ ! -f "$GUI/libxcb-cursor.so.0" || ! -d "$GUI/packages/PySide6" ]]; then
  printf '%s\n' "Qt runtime is unavailable: $GUI. Check the shared disk." >&2
  exit 1
fi
unset LD_PRELOAD PYTHONHOME QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH
export PYTHONPATH="$GUI/packages:$ROOT/src"
export LD_LIBRARY_PATH="$GUI/packages/PySide6/Qt/lib:$GUI"
export QT_QPA_PLATFORM=xcb PYTHONNOUSERSITE=1
export QT_IM_MODULE=${FROMREALHAND_IM_MODULE:-ibus}
cd "$ROOT"
exec "$ROOT/data/runtime/stage6-study-venv/bin/python" "$ROOT/scripts/127_qt_grasp.py" --root "$STUDY_ROOT" "$@"
