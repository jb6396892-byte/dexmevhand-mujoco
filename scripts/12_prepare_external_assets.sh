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

DEXYCB_ROOT="${DEXYCB_ROOT:-/media/smgbro/shared/DexYCB/dataset}"
DEXYCB_ARCHIVES="${DEXYCB_ARCHIVES:-/media/smgbro/shared/DexYCB/archives}"
MANO_BASE="${MANO_BASE:-/media/smgbro/shared/DexYCB/mano}"
MANO_ROOT="${MANO_ROOT:-$MANO_BASE/models}"
CHECKSUM_DIR="${CHECKSUM_DIR:-/media/smgbro/shared/DexYCB/checksums}"
EXTRACT=false

if [[ "${1:-}" == "--extract" ]]; then
  EXTRACT=true
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--extract]" >&2
  exit 2
fi

mkdir -p "$DEXYCB_ROOT" "$DEXYCB_ARCHIVES" "$MANO_ROOT" "$CHECKSUM_DIR"

check_tarball() {
  local archive="$1"
  if [[ ! -f "$archive" ]]; then
    echo "MISSING: $archive"
    return 1
  fi
  if ! tar -tzf "$archive" >/dev/null; then
    echo "INVALID: $archive" >&2
    return 1
  fi
  echo "VALID: $archive"
}

calibration_archive="$DEXYCB_ARCHIVES/calibration.tar.gz"
models_archive="$DEXYCB_ARCHIVES/models.tar.gz"

missing=0
check_tarball "$calibration_archive" || missing=1
check_tarball "$models_archive" || missing=1

if $EXTRACT && [[ "$missing" -eq 0 ]]; then
  tar -xzf "$calibration_archive" -C "$DEXYCB_ROOT"
  tar -xzf "$models_archive" -C "$DEXYCB_ROOT"
  sha256sum "$calibration_archive" "$models_archive" > "$CHECKSUM_DIR/dexycb-assets.sha256"
fi

mano_archive="$(find "$DEXYCB_ARCHIVES" -maxdepth 1 -type f -iname 'mano_v*.zip' -print -quit)"
if [[ -n "$mano_archive" ]]; then
  unzip -tq "$mano_archive" >/dev/null
  echo "VALID: $mano_archive"
  if $EXTRACT; then
    unzip -q -o "$mano_archive" -d "$MANO_BASE/source"
    right_model="$(find "$MANO_BASE/source" -type f -name MANO_RIGHT.pkl -print -quit)"
    left_model="$(find "$MANO_BASE/source" -type f -name MANO_LEFT.pkl -print -quit)"
    if [[ -z "$right_model" || -z "$left_model" ]]; then
      echo "MANO archive does not contain both MANO_RIGHT.pkl and MANO_LEFT.pkl" >&2
      exit 1
    fi
    cp -f "$right_model" "$MANO_ROOT/MANO_RIGHT.pkl"
    cp -f "$left_model" "$MANO_ROOT/MANO_LEFT.pkl"
    sha256sum "$mano_archive" > "$CHECKSUM_DIR/mano-assets.sha256"
  fi
else
  echo "MISSING: $DEXYCB_ARCHIVES/mano_v1_2.zip (download requires MANO account and license acceptance)"
fi

[[ -d "$DEXYCB_ROOT/calibration" ]] && echo "READY: calibration"
[[ -d "$DEXYCB_ROOT/models/025_mug" ]] && echo "READY: models/025_mug"
[[ -f "$MANO_ROOT/MANO_RIGHT.pkl" ]] && echo "READY: MANO_RIGHT.pkl"
[[ -f "$MANO_ROOT/MANO_LEFT.pkl" ]] && echo "READY: MANO_LEFT.pkl"

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi
