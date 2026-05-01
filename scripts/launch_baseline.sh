#!/usr/bin/env bash
# scripts/launch_baseline.sh — start the OpenHands (no Patch Memory)
# runner over the full Terminal-shift dataset, detached in its own
# tmux session. Same shape as launch_pm.sh but uses the
# memory-less OpenHandsBaseline agent.
#
# By default we **skip singleton chains** so the baseline runs over the
# same task set that the pm variant runs over (apples-to-apples
# comparison). Pass ``--include-singletons`` to override.
#
# Equivalent of:
#   scripts/launch_runs.sh --variant baseline --parallel 6 --max-chains 30 --no-singletons \
#       --extra-run-chain-args "--model gpt-5.5" [extra args ...]

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

NO_SINGLETONS_ARG="--no-singletons"
PASSTHROUGH=()
for arg in "$@"; do
    if [ "$arg" = "--include-singletons" ]; then
        NO_SINGLETONS_ARG=""
    else
        PASSTHROUGH+=("$arg")
    fi
done

EXEC_ARGS=(
    --variant baseline
    --parallel 6
    --max-chains 30
    --extra-run-chain-args "--model gpt-5.5"
    --start-chain-index 1
)
[ -n "$NO_SINGLETONS_ARG" ] && EXEC_ARGS+=("$NO_SINGLETONS_ARG")
EXEC_ARGS+=("${PASSTHROUGH[@]+${PASSTHROUGH[@]}}")

exec "$HERE/launch_runs.sh" "${EXEC_ARGS[@]}"
