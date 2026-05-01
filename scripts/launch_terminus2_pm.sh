#!/usr/bin/env bash
# scripts/launch_terminus2_pm.sh — start the Terminus2 + Patch Memory
# runner over the full Terminal-shift dataset, detached in its own tmux
# session.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${TERMINUS2_LLM_ENV:-$HERE/terminus2_llm.env}"

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
fi

NO_SINGLETONS_ARG="--no-singletons"
PASSTHROUGH=()
for arg in "$@"; do
    if [ "$arg" = "--include-singletons" ]; then
        NO_SINGLETONS_ARG=""
    else
        PASSTHROUGH+=("$arg")
    fi
done

EXTRA_CHAIN_ARGS="--model ${LLM_MODEL:-gpt-5-mini}"
if [ -n "${LLM_API_KEY:-}" ]; then
    EXTRA_CHAIN_ARGS="${EXTRA_CHAIN_ARGS} --ae LLM_API_KEY=${LLM_API_KEY}"
fi
if [ -n "${LLM_BASE_URL:-}" ]; then
    EXTRA_CHAIN_ARGS="${EXTRA_CHAIN_ARGS} --ae LLM_BASE_URL=${LLM_BASE_URL}"
fi
if [ -n "${LLM_MODEL:-}" ]; then
    EXTRA_CHAIN_ARGS="${EXTRA_CHAIN_ARGS} --ae LLM_MODEL=${LLM_MODEL}"
fi

EXEC_ARGS=(
    --variant terminus2_pm
    --parallel 6
    --start-chain-index 1
    --max-chains 10
    --tmux-session harbor-terminus2-pm
    --extra-run-chain-args "$EXTRA_CHAIN_ARGS"
)
[ -n "$NO_SINGLETONS_ARG" ] && EXEC_ARGS+=("$NO_SINGLETONS_ARG")
EXEC_ARGS+=("${PASSTHROUGH[@]+${PASSTHROUGH[@]}}")

exec "$HERE/launch_runs.sh" "${EXEC_ARGS[@]}"
