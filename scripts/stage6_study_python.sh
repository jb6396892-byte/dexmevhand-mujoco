#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${STUDY_ROOT:?Set STUDY_ROOT to the shared delivery directory}/language/runtime"
export PYTHONPATH="$BASE/packages:$ROOT/src" HF_HOME="$BASE/huggingface" TMPDIR="$BASE/tmp"
export PYTHONNOUSERSITE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
exec "$ROOT/data/runtime/stage6-study-venv/bin/python" "$@"
