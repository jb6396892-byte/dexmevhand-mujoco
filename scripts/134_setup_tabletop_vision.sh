#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STORE=${VISUAL_GRASP_ROOT:-/media/smgbro/shared/visual_grasp}
PY="$ROOT/data/runtime/stage6-study-venv/bin/python"
mkdir -p "$STORE/runtime/packages" "$STORE/runtime/tmp" "$STORE/models"
export TMPDIR="$STORE/runtime/tmp" PIP_CACHE_DIR="$STORE/runtime/pip-cache"
env -u LD_PRELOAD -u PYTHONPATH "$PY" -m pip install --no-cache-dir --target "$STORE/runtime/packages" \
  'numpy==1.26.4' 'scipy==1.13.1' 'pillow==11.1.0' 'opencv-python-headless==4.10.0.84' \
  'open3d==0.19.0' 'trimesh==4.6.8'
env -u LD_PRELOAD -u PYTHONPATH "$PY" -m pip install --no-deps --no-cache-dir \
  --target "$STORE/runtime/packages" 'torchvision==0.20.1' --index-url https://download.pytorch.org/whl/cu121
