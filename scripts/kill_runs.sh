#!/usr/bin/env bash
# scripts/kill_runs.sh — clean teardown of a launch_runs.sh deployment.
#
# What gets stopped, in order:
#   1. The tmux session                     (so detached users notice)
#   2. The per-chain process groups         (each chain has its own pgid)
#   3. The dispatcher process group         (xargs + bash subshell pool)
#   4. Optional: any leftover docker containers spawned by openhands
#
# We start with SIGTERM so harbor + openhands get a chance to run their
# atexit handlers (which is how ``DockerEnvironment.delete=True`` cleans
# the per-task container). After ``--grace`` seconds we follow up with
# SIGKILL for any straggler.
#
# Usage:
#   scripts/kill_runs.sh --variant pm
#   scripts/kill_runs.sh --variant baseline --grace 60
#   scripts/kill_runs.sh --all
#   scripts/kill_runs.sh --variant pm --reap-docker

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 (--variant {pm|baseline|terminus2-pm|terminus2-baseline} | --all) [options]

  --variant {pm|baseline|terminus2-pm|terminus2-baseline}
                            tear down one variant's runner
  --all                     tear down all supported variants
  --grace SECONDS           seconds to wait between SIGTERM and SIGKILL (default: 30)
  --trials-dir PATH         override trials dir (defaults to ~/qingchuan/harbor_patch_memory/runs/full-<variant>)
  --tmux-session NAME       override tmux session name (default: harbor-<variant>)
  --reap-docker             after kill, also ``docker rm -f`` any leftover
                            containers whose name starts with ``openhands-runtime-``
  --dry-run                 print what would be killed, take no action

The harbor agent records pgids for every chain it runs at:
  <trials-dir>/_logs/runner/{dispatcher,chain.<id>}.pgid
We use those files; if they're missing we still tmux-kill and best-effort
``pkill`` stragglers.
EOF
}

VARIANTS=()
GRACE=30
TRIALS_DIR=""
TMUX_SESSION=""
REAP_DOCKER=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --variant)       VARIANTS+=("$2"); shift 2 ;;
        --all)           VARIANTS=(pm baseline terminus2-pm terminus2-baseline); shift ;;
        --grace)         GRACE="$2"; shift 2 ;;
        --trials-dir)    TRIALS_DIR="$2"; shift 2 ;;
        --tmux-session)  TMUX_SESSION="$2"; shift 2 ;;
        --reap-docker)   REAP_DOCKER=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [ ${#VARIANTS[@]} -eq 0 ]; then
    echo "error: pass --variant or --all" >&2
    usage
    exit 2
fi
if [ ${#VARIANTS[@]} -gt 1 ] && { [ -n "$TRIALS_DIR" ] || [ -n "$TMUX_SESSION" ]; }; then
    echo "error: --trials-dir / --tmux-session don't make sense with multiple variants" >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---------- helpers --------------------------------------------------
maybe() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] $*"
    else
        "$@" || true
    fi
}

is_alive() {
    # Negative arg means "is any process in this pgid alive?".
    kill -0 "-$1" 2>/dev/null
}

term_pgid() {
    local label="$1" pgid="$2"
    if is_alive "$pgid"; then
        echo "  SIGTERM  $label  pgid=$pgid"
        maybe kill -TERM "-$pgid"
    else
        echo "  (gone)   $label  pgid=$pgid"
    fi
}

kill_pgid() {
    local label="$1" pgid="$2"
    if is_alive "$pgid"; then
        echo "  SIGKILL  $label  pgid=$pgid"
        maybe kill -KILL "-$pgid"
    fi
}

# ---------- per-variant kill ---------------------------------------
kill_variant() {
    local variant="$1"
    local default_trials="$PROJECT_ROOT/runs/full-$variant"
    local trials_dir="${TRIALS_DIR:-$default_trials}"
    local tmux_session="${TMUX_SESSION:-harbor-$variant}"
    local logs_dir="$trials_dir/_logs/runner"

    echo "================================================================"
    echo " tearing down harbor-$variant"
    echo "   tmux_session : $tmux_session"
    echo "   trials_dir   : $trials_dir"
    echo "   logs_dir     : $logs_dir"
    echo "================================================================"

    # 1. Kill the tmux session — detaches any attached client and
    # SIGHUPs the foreground process. This is mostly for the user's
    # mental model; the actual process kill happens via pgid below.
    if tmux has-session -t "$tmux_session" 2>/dev/null; then
        echo "  tmux kill-session -t $tmux_session"
        maybe tmux kill-session -t "$tmux_session"
    else
        echo "  (no tmux session named $tmux_session)"
    fi

    # 2. SIGTERM each chain's pgid, then the dispatcher's.
    if [ -d "$logs_dir" ]; then
        echo "  -- SIGTERM phase --"
        for pgid_file in "$logs_dir"/chain.*.pgid; do
            [ -e "$pgid_file" ] || continue
            local chain pgid
            chain=$(basename "$pgid_file" .pgid)
            pgid=$(cat "$pgid_file" 2>/dev/null || true)
            [ -z "$pgid" ] && continue
            term_pgid "$chain" "$pgid"
        done
        if [ -f "$logs_dir/dispatcher.pid" ]; then
            local dpid
            dpid=$(cat "$logs_dir/dispatcher.pid")
            term_pgid "dispatcher" "$dpid"
        fi

        # 3. Grace period, then SIGKILL stragglers.
        echo "  -- waiting ${GRACE}s for graceful shutdown --"
        if [ "$DRY_RUN" != "1" ]; then sleep "$GRACE"; fi

        echo "  -- SIGKILL phase --"
        for pgid_file in "$logs_dir"/chain.*.pgid; do
            [ -e "$pgid_file" ] || continue
            local chain pgid
            chain=$(basename "$pgid_file" .pgid)
            pgid=$(cat "$pgid_file" 2>/dev/null || true)
            [ -z "$pgid" ] && continue
            kill_pgid "$chain" "$pgid"
        done
        if [ -f "$logs_dir/dispatcher.pid" ]; then
            local dpid
            dpid=$(cat "$logs_dir/dispatcher.pid")
            kill_pgid "dispatcher" "$dpid"
        fi
    else
        echo "  (no logs_dir at $logs_dir; nothing to read pgids from)"
    fi

    # 4. Optional Docker cleanup. OpenHands' LocalRuntime spawns
    # ``openhands-runtime-<id>``; ``DockerEnvironment.delete=True``
    # *should* remove them on exit but a SIGKILL bypasses that.
    #
    # CAREFUL: this server commonly has unrelated
    # ``openhands-runtime-*`` containers from other workloads. We
    # therefore only kill containers whose .Created timestamp is
    # AFTER the run's recorded start time (with a small safety
    # margin), so we never touch someone else's run.
    if [ "$REAP_DOCKER" = "1" ]; then
        echo "  -- reaping leftover docker containers (created after this run started) --"
        local meta_file="$logs_dir/run_meta.json"
        local since_iso=""
        if [ -f "$meta_file" ]; then
            since_iso=$(grep -oE '"started_at"\s*:\s*"[^"]+"' "$meta_file" \
                        | head -1 | sed 's/.*"started_at"\s*:\s*"\([^"]*\)".*/\1/')
        fi
        if [ -z "$since_iso" ] && [ -f "$logs_dir/dispatcher.pid" ]; then
            # Fallback: ctime of dispatcher.pid (1s precision is fine).
            since_iso=$(stat -c '%y' "$logs_dir/dispatcher.pid" 2>/dev/null \
                        | awk '{print $1"T"$2}' | cut -c1-19)
        fi
        if [ -z "$since_iso" ]; then
            echo "  (could not determine run start time; refusing to reap docker)"
            return 0
        fi
        # Convert to epoch (seconds). Subtract 30s safety margin.
        local since_epoch
        since_epoch=$(date -d "$since_iso" +%s 2>/dev/null) || since_epoch=""
        if [ -z "$since_epoch" ]; then
            echo "  (could not parse start timestamp '$since_iso'; refusing to reap docker)"
            return 0
        fi
        since_epoch=$((since_epoch - 30))
        echo "  reap-cutoff: containers created after $(date -d "@$since_epoch" -Iseconds)"

        local candidates
        candidates=$(docker ps -a --format '{{.Names}}' 2>/dev/null \
                     | grep -E '^(openhands-runtime-|harbor-)' || true)
        local n_killed=0
        for name in $candidates; do
            local created
            created=$(docker inspect --format '{{.Created}}' "$name" 2>/dev/null) || continue
            local created_epoch
            created_epoch=$(date -d "$created" +%s 2>/dev/null) || continue
            if [ "$created_epoch" -ge "$since_epoch" ]; then
                echo "  docker rm -f $name  (created $created)"
                maybe docker rm -f "$name" >/dev/null
                n_killed=$((n_killed + 1))
            fi
        done
        if [ "$n_killed" -eq 0 ]; then
            echo "  (no eligible containers; nothing reaped)"
        fi
    fi

    echo "  done."
    echo
}

for variant in "${VARIANTS[@]}"; do
    if [ "$variant" != "pm" ] \
       && [ "$variant" != "baseline" ] \
       && [ "$variant" != "terminus2-pm" ] \
       && [ "$variant" != "terminus2-baseline" ]; then
        echo "skipping unknown variant: $variant" >&2
        continue
    fi
    kill_variant "$variant"
done
