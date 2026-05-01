#!/usr/bin/env bash
# scripts/launch_pm.sh — start the OpenHands + Patch Memory runner over
# the full Terminal-shift dataset, detached in its own tmux session.
#
# Sequential within a chain (the prototype's records prime each
# variant), up to 6 chains concurrent across the dataset.
#
# By default we **skip singleton chains** (chains where the prototype
# has no detected variants) — Patch Memory cannot help nor be evaluated
# on a single-task chain, and we want the pm and baseline runs to
# evaluate the same task set. Pass ``--include-singletons`` to override.
#
# Equivalent of:
#   scripts/launch_runs.sh --variant pm --parallel 6 --max-chains 30 --no-singletons \
#       --extra-run-chain-args "--model gpt-5.5" [extra args ...]
#
# Pass any extra ``launch_runs.sh`` flag verbatim, e.g.:
#   scripts/launch_pm.sh --include-singletons
#   scripts/launch_pm.sh --chains 'bn-fit-modify build-pmars'
#   scripts/launch_pm.sh --parallel 4

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# Allow opting back in.
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
    --variant pm
    --parallel 6
    --max-chains 30
    --start-chain-index 1
    --extra-run-chain-args "--model gpt-5.5"
)
[ -n "$NO_SINGLETONS_ARG" ] && EXEC_ARGS+=("$NO_SINGLETONS_ARG")
EXEC_ARGS+=("${PASSTHROUGH[@]+${PASSTHROUGH[@]}}")

exec "$HERE/launch_runs.sh" "${EXEC_ARGS[@]}"
