#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${STAGE6_SHARED:-/media/smgbro/shared}/fromrealhand-stage6"
BOOT="$ROOT/data/runtime/stage6-venv/bin/python"
if [[ ! -x "$BOOT" || ! -d "$BASE/packages/transformers" ]]; then
  printf 'Stage6 runtime is not installed. Read docs/STAGE6_LANGUAGE.md first.\n' >&2
  exit 2
fi
export PYTHONPATH="$BASE/packages:$ROOT/src" HF_HOME="$BASE/huggingface"
export TMPDIR="$BASE/tmp" HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
exec "$BOOT" "$@"
