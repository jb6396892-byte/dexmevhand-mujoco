#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
cd "$ROOT"
exec /home/smgbro/miniconda3/bin/conda run --no-capture-output -n dexmv \
  python "$ROOT/scripts/26_train_verified_curriculum.py" "$@"
