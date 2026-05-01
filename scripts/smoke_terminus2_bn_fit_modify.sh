#!/usr/bin/env bash
# Smoke-test the bn-fit-modify chain through Terminus2 baseline/PM.
# Usage: scripts/smoke_terminus2_bn_fit_modify.sh [terminus2_pm|terminus2_baseline|both]
set -euo pipefail

VARIANT="${1:-terminus2_pm}"
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${HARBOR_PM_DATASET:-$HOME/qingchuan/Terminal-shift-main}"
RUNS_BASE="${HARBOR_PM_RUNS_DIR:-$PKG_ROOT/runs}"
ENV_FILE="${TERMINUS2_LLM_ENV:-$PKG_ROOT/scripts/terminus2_llm.env}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

cd "$PKG_ROOT"

run_variant () {
  local v="$1"
  local out="$RUNS_BASE/bn-fit-modify-$v"
  local -a cmd=(
    python scripts/run_chain.py
    --dataset "$DATASET"
    --chain bn-fit-modify
    --variant "$v"
    --trials-dir "$out"
    --model "${LLM_MODEL:-gpt-5-mini}"
  )
  if [ -n "${LLM_API_KEY:-}" ]; then
    cmd+=(--ae "LLM_API_KEY=${LLM_API_KEY}")
  fi
  if [ -n "${LLM_BASE_URL:-}" ]; then
    cmd+=(--ae "LLM_BASE_URL=${LLM_BASE_URL}")
  fi
  if [ -n "${LLM_MODEL:-}" ]; then
    cmd+=(--ae "LLM_MODEL=${LLM_MODEL}")
  fi
  echo "===> launching $v variant -> $out"
  "${cmd[@]}"
}

case "$VARIANT" in
  terminus2_pm)        run_variant terminus2_pm ;;
  terminus2_baseline)  run_variant terminus2_baseline ;;
  both)
    run_variant terminus2_baseline
    run_variant terminus2_pm
    ;;
  *)
    echo "unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac
