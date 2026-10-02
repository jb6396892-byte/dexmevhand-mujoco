#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export STUDY_ROOT=${STUDY_ROOT:-/media/smgbro/shared/lora}
if [[ $# -eq 0 ]]; then
  printf '%s\n' 'Usage: bash scripts/116_run_stage6_model.sh "instruction" [--scene first|second] [--execute --render]'
  exit 2
fi
cd "$ROOT"
for arg in "$@"; do
  if [[ "$arg" == --render || "$arg" == --verify-render ]]; then
    exec bash scripts/stage6_study_python.sh scripts/130_run_language_render.py "$@" --root "$STUDY_ROOT"
  fi
done
exec bash scripts/stage6_study_python.sh scripts/123_run_guarded_language.py "$@" --root "$STUDY_ROOT"
