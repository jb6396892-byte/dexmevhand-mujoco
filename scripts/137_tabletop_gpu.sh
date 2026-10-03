#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
exec /home/smgbro/miniconda3/envs/dexmv/bin/python -c \
 'import os,sys; from fromrealhand.desktop.runtime import physics_environment,LEGACY_PYTHON; os.execve(LEGACY_PYTHON,[LEGACY_PYTHON]+sys.argv[1:],physics_environment())' "$@"
