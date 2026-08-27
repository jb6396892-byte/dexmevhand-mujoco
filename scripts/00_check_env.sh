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
DEXYCB_ROOT="${DEXYCB_ROOT:-/media/smgbro/shared/DexYCB/dataset}"
MANO_ROOT="${MANO_ROOT:-/media/smgbro/shared/DexYCB/mano/models}"
PYTHONPATH="$ROOT/src:$DEXMV_SIM:$DEXMV_LEARN:$DEXMV_LEARN/mjrl:${PYTHONPATH:-}"

echo "Project root: $ROOT"
echo "DexMV sim: $DEXMV_SIM"
echo "DexMV learn: $DEXMV_LEARN"
echo "Conda env: $CONDA_ENV"
echo "DexYCB root: $DEXYCB_ROOT"
echo "MANO root: $MANO_ROOT"

test -d "$DEXMV_SIM"
test -d "$DEXMV_LEARN"
test -x "$CONDA_BIN"
test -d "/home/smgbro/.mujoco/mujoco200/bin"

LD_PRELOAD="$LD_PRELOAD" \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
PYTHONPATH="$PYTHONPATH" \
PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
"$CONDA_BIN" run -n "$CONDA_ENV" python -c "import chumpy, manopth, mujoco_py, hand_imitation, mjrl, tpi; from manopth.manolayer import ManoLayer; from hand_imitation.kinematics.demonstration.relocation_demo import RelocationDemonstration; print('imports ok')"

echo "Environment check passed."

if [[ -d "$DEXYCB_ROOT/calibration" ]]; then
  echo "DexYCB calibration: ready"
else
  echo "DexYCB calibration: missing (run scripts/12_prepare_external_assets.sh --extract)"
fi

if [[ -d "$DEXYCB_ROOT/models/025_mug" ]]; then
  echo "YCB 025_mug model: ready"
else
  echo "YCB 025_mug model: missing (run scripts/12_prepare_external_assets.sh --extract)"
fi

if [[ -f "$MANO_ROOT/MANO_RIGHT.pkl" && -f "$MANO_ROOT/MANO_LEFT.pkl" ]]; then
  echo "MANO models: ready"
else
  echo "MANO models: awaiting licensed mano_v1_2.zip"
fi
