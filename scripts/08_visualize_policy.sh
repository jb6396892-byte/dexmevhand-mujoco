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

POLICY_PATH="${1:-}"
if [[ -z "$POLICY_PATH" ]]; then
  echo "Usage: bash scripts/08_visualize_policy.sh /path/to/best_policy.pickle" >&2
  exit 2
fi

OBJECT_NAME="${OBJECT_NAME:-mug}"
OBJECT_SCALE="${OBJECT_SCALE:-0.8}"
RANDOMNESS_SCALE="${RANDOMNESS_SCALE:-0.25}"

cd "$DEXMV_SIM"
LD_PRELOAD="$LD_PRELOAD" \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
PYTHONPATH="$PYTHONPATH" \
PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \
"$CONDA_BIN" run -n "$CONDA_ENV" python vb2.py \
  --env_name relocate \
  --object_name "$OBJECT_NAME" \
  --policy_path "$POLICY_PATH" \
  --randomness_scale "$RANDOMNESS_SCALE" \
  --object_scale "$OBJECT_SCALE"
