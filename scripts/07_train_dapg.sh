#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
elif [[ -f "$ROOT/.env.example" ]]; then
  set -a
  source "$ROOT/.env.example"
  set +a
fi

DEXMV_SIM="${DEXMV_SIM:-/home/smgbro/dexmv-sim}"
DEXMV_LEARN="${DEXMV_LEARN:-/home/smgbro/dexmv-learn}"
CONDA_ENV="${CONDA_ENV:-dexmv}"
CONDA_BIN="${CONDA_BIN:-/home/smgbro/miniconda3/bin/conda}"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu}"
LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/fromrealhand-pycache}"
PYTHONPATH="$ROOT/src:$DEXMV_SIM:$DEXMV_LEARN:$DEXMV_LEARN/mjrl:${PYTHONPATH:-}"

CFG="${1:-$ROOT/configs/dapg-mug-real.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "$DEXMV_SIM/examples"
LD_PRELOAD="$LD_PRELOAD" \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
PYTHONPATH="$PYTHONPATH" \
PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
"$CONDA_BIN" run -n "$CONDA_ENV" python train.py --cfg "$CFG" "$@"
