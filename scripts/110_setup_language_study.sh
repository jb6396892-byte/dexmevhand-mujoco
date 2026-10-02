#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:?Supply a shared delivery directory}"
case "$DEST" in /media/smgbro/shared/*) ;; *) printf 'Destination must be on shared\n' >&2; exit 2;; esac
mountpoint -q /media/smgbro/shared
OPTIONS="$(findmnt -n -o OPTIONS --target /media/smgbro/shared)"
[[ ",$OPTIONS," != *,ro,* ]] || { printf 'Shared is read-only in this permission context\n' >&2; exit 2; }
BASE="$DEST/language/runtime"
BOOT="$ROOT/data/runtime/stage6-study-venv"
[[ ! -e "$BASE" && ! -e "$BOOT" ]] || { printf 'Runtime exists; inspect before resuming\n' >&2; exit 2; }
mkdir -p "$BASE/packages" "$BASE/tmp" "$BASE/huggingface"
/home/smgbro/miniconda3/bin/python -m venv --copies "$BOOT"
export TMPDIR="$BASE/tmp" HF_HOME="$BASE/huggingface" PYTHONNOUSERSITE=1
"$BOOT/bin/python" -m pip install --no-cache-dir --target "$BASE/packages" \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r "$ROOT/requirements-stage6.txt"
PYTHONPATH="$BASE/packages" "$BOOT/bin/python" -c \
  'import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
