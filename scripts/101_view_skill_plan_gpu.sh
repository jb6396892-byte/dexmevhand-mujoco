#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
printf '%s\n' 'Hierarchy uses verified expert actions, not the running DAPG checkpoint. Close the window to exit.'
exec /home/smgbro/miniconda3/envs/dexmv/bin/python "$ROOT/scripts/16_run_instruction.py" "$@" --render
