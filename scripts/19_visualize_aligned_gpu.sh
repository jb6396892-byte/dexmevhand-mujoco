#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_PACKAGE="$ROOT/../.local/mujoco-py-gpu"
if [[ ! -f "$GPU_PACKAGE/mujoco_py/generated/cymj_2.0.2.8_37_linuxgpuextensionbuilder_37.so" ]]; then
  echo "GPU mujoco-py extension is missing: $GPU_PACKAGE" >&2
  exit 1
fi

export PYTHONPATH="$GPU_PACKAGE:$ROOT/src:/home/smgbro/dexmv-sim:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so"
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

echo "Pose replay only: hand and mug states are set per frame; this is not a trained-policy physics rollout." >&2
exec /home/smgbro/miniconda3/bin/conda run --no-capture-output -n dexmv \
  python "$ROOT/scripts/04_visualize_retargeting.py" \
  --retargeting "$ROOT/data/real_data/relocate_mug/seq_dexycb_001/retargeting_mano_aligned.pkl" \
  --object-dir "$ROOT/data/real_data/relocate_mug/seq_dexycb_001/object_pose" \
  --camera-to-world "$ROOT/data/real_data/relocate_mug/seq_dexycb_001/calib/camera_to_mujoco.npy" \
  --skip-frame 20 "$@"
