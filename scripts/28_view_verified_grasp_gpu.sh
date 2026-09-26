#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_PACKAGE="$ROOT/../.local/mujoco-py-gpu"
EXTENSION="$GPU_PACKAGE/mujoco_py/generated/cymj_2.0.2.8_37_linuxgpuextensionbuilder_37.so"
if [[ ! -f "$EXTENSION" ]]; then
  echo "GPU mujoco-py extension is missing: $EXTENSION" >&2
  exit 1
fi
if [[ -z "${DISPLAY:-}" ]]; then
  echo "No X11 display is available. Run this command in a terminal on the Ubuntu desktop." >&2
  exit 1
fi

export PYTHONPATH="$GPU_PACKAGE:$ROOT/src:/home/smgbro/dexmv-sim:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so"
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

cd "$ROOT"
exec /home/smgbro/miniconda3/bin/conda run --no-capture-output -n dexmv \
  python "$ROOT/scripts/28_view_verified_grasp.py" "$@"
