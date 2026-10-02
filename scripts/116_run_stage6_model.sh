#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export STUDY_ROOT=${STUDY_ROOT:-/media/smgbro/shared/lora}
LANGUAGE="$STUDY_ROOT/language"
STUDY="$LANGUAGE/study_v3"
if [[ $# -eq 0 ]]; then
  printf '%s\n' 'Usage: bash scripts/116_run_stage6_model.sh "instruction" [--scene first|second] [--execute --render]'
  exit 2
fi
OUTPUT="$STUDY/instructions/$(date -u +%Y%m%dT%H%M%S%NZ)"
cd "$ROOT"
exec bash scripts/stage6_study_python.sh scripts/106_run_language_plan.py "$@" \
  --model "$LANGUAGE/model" --adapter "$STUDY/formal/adapter" \
  --acceptance-report "$STUDY/physical-acceptance/summary.json" --output "$OUTPUT"
