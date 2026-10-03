#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export VISUAL_GRASP_ROOT=${VISUAL_GRASP_ROOT:-/media/smgbro/shared/visual_grasp}
unset LD_PRELOAD PYTHONHOME QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH
export PYTHONPATH="$VISUAL_GRASP_ROOT/runtime/packages:/media/smgbro/shared/lora/language/runtime/packages:$ROOT/src"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export HF_HOME="$VISUAL_GRASP_ROOT/runtime/huggingface" TMPDIR="$VISUAL_GRASP_ROOT/runtime/tmp"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
cd "$ROOT"
exec "$ROOT/data/runtime/stage6-study-venv/bin/python" "$@"
