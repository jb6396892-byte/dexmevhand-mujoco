#!/usr/bin/env bash
# Explicit setup only, with large files on a writable shared filesystem.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED="${STAGE6_SHARED:-/media/smgbro/shared}"
BASE="$SHARED/fromrealhand-stage6"
if ! mountpoint -q "$SHARED"; then
  printf 'Shared storage is not mounted: %s\n' "$SHARED" >&2
  exit 2
fi
OPTIONS="$(findmnt -n -o OPTIONS --target "$SHARED")"
if [[ ",$OPTIONS," == *,ro,* ]]; then
  printf 'Shared storage is read-only. No installation or disk repair performed.\n' >&2
  exit 2
fi
if [[ "${1:-}" == "--check-only" ]]; then
  printf 'Writable mount detected; setup has NOT run. Destination: %s\n' "$BASE"
  exit 0
fi
if [[ "${1:-}" != "--install" ]]; then
  printf 'Use --check-only or --install. No implicit download.\n' >&2
  exit 2
fi
AVAILABLE="$(df -PB1 "$SHARED" | awk 'NR==2 {print $4}')"
if (( AVAILABLE < 10737418240 )); then
  printf 'At least 10 GiB free shared storage is required.\n' >&2
  exit 2
fi
if [[ -e "$BASE" ]]; then
  printf 'Refusing to overwrite existing runtime: %s\n' "$BASE" >&2
  exit 2
fi
mkdir -p "$BASE/packages" "$BASE/tmp" "$BASE/huggingface"
# exFAT cannot host venv symlinks. Only the small bootstrap lives on Linux.
BOOT="$ROOT/data/runtime/stage6-venv"
if [[ -e "$BOOT" ]]; then
  printf 'Existing bootstrap found; inspect it before resuming: %s\n' "$BOOT" >&2
  exit 2
fi
/home/smgbro/miniconda3/bin/python -m venv --copies "$BOOT"
export TMPDIR="$BASE/tmp" HF_HOME="$BASE/huggingface"
"$BOOT/bin/python" -m pip install --no-cache-dir --target "$BASE/packages" \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r "$ROOT/requirements-stage6.txt"
PYTHONPATH="$BASE/packages:$ROOT/src" "$BOOT/bin/python" -c \
  'import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__); assert torch.cuda.is_available()'
printf 'Runtime installed. Model download and LoRA training remain separate explicit steps.\n'
