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

echo "Project root: $ROOT"
echo "DexMV sim: $DEXMV_SIM"
echo "DexMV learn: $DEXMV_LEARN"
echo "Conda env: $CONDA_ENV"

test -d "$DEXMV_SIM"
test -d "$DEXMV_LEARN"
test -x "$CONDA_BIN"
test -d "/home/smgbro/.mujoco/mujoco200/bin"

LD_PRELOAD="$LD_PRELOAD" \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
PYTHONPATH="$PYTHONPATH" \
PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
"$CONDA_BIN" run -n "$CONDA_ENV" python -c "import mujoco_py, hand_imitation, mjrl, tpi; from hand_imitation.kinematics.demonstration.relocation_demo import RelocationDemonstration; print('imports ok')"

echo "Environment check passed."
