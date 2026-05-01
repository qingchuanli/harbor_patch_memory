#!/usr/bin/env bash
# Smoke-test the bn-fit-modify chain through both agent variants.
# Usage:  scripts/smoke_bn_fit_modify.sh [pm|baseline|both]
set -euo pipefail

VARIANT="${1:-pm}"
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${HARBOR_PM_DATASET:-$HOME/qingchuan/Terminal-shift-main}"
RUNS_BASE="${HARBOR_PM_RUNS_DIR:-$PKG_ROOT/runs}"

cd "$PKG_ROOT"

run_variant () {
  local v="$1"
  local out="$RUNS_BASE/bn-fit-modify-$v"
  echo "===> launching $v variant -> $out"
  python scripts/run_chain.py \
    --dataset "$DATASET" \
    --chain bn-fit-modify \
    --variant "$v" \
    --trials-dir "$out"
}

case "$VARIANT" in
  pm)        run_variant pm ;;
  baseline)  run_variant baseline ;;
  both)
    run_variant baseline
    run_variant pm
    ;;
  *)
    echo "unknown variant: $VARIANT (expected pm|baseline|both)" >&2
    exit 2
    ;;
esac
