#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STUDY_ROOT=${STUDY_ROOT:-/media/smgbro/shared/lora}
GUI="$STUDY_ROOT/language/gui-runtime"
PYTHON="$ROOT/data/runtime/stage6-study-venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  printf '%s\n' 'Existing stage6 Python bootstrap is missing.' >&2
  exit 1
fi
mkdir -p "$GUI"
if [[ ! -d "$GUI/packages/PySide6" ]]; then
  env -u LD_PRELOAD -u PYTHONPATH "$PYTHON" -m pip install --no-cache-dir \
    --target "$GUI/packages" -r "$ROOT/configs/requirements-stage6-qt.txt"
fi
if [[ ! -f "$GUI/libxcb-cursor.so.0" ]]; then
  TMP=$(mktemp -d /tmp/fromrealhand-qt.XXXXXX)
  trap 'rm -rf "$TMP"' EXIT
  (
    cd "$TMP"
    apt-get download libxcb-cursor0
    packages=(libxcb-cursor0_*.deb)
    dpkg-deb -x "${packages[0]}" extracted
    cp -L extracted/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0 "$GUI/"
  )
fi
printf '%s\n' "Qt runtime ready: $GUI" "Launch: bash $ROOT/scripts/128_launch_qt.sh"
