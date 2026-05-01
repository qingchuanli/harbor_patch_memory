#!/usr/bin/env python3
"""Run one or more Terminal-shift chains through Harbor agents [+ PM].

A "chain" is a directory layout like::

    <dataset_root>/<chain_id>/                    (prototype)
    <dataset_root>/<chain_id>-EVO-1/              (variant)
    <dataset_root>/<chain_id>-EVO-2/
    <dataset_root>/<chain_id>-v-rocker/           (or -v-* style variants)

For Patch Memory to work, the prototype must run *before* its variants
and successive runs must observe each other's records. This script
enforces sequential order **inside** a chain and lets multiple chains
run in parallel via a process pool.

Each task is launched by spawning ``harbor trials start`` as a
subprocess, so the chain runner is decoupled from any in-process
Harbor state.

Examples
--------

Single chain, with OpenHands + PM (the default smoke-test command)::

    python scripts/run_chain.py \
        --dataset ~/qingchuan/Terminal-shift-main \
        --chain bn-fit-modify \
        --variant pm

Same chain, baseline ablation (no PM)::

    python scripts/run_chain.py \
        --dataset ~/qingchuan/Terminal-shift-main \
        --chain bn-fit-modify \
        --variant baseline

Multiple chains in parallel::

    python scripts/run_chain.py \
        --dataset ~/qingchuan/Terminal-shift-main \
        --chain bn-fit-modify \
        --chain adaptive-rejection-sampler \
        --variant pm \
        --max-parallel 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Map agent variant -> (Harbor --agent-import-path, friendly tag)
VARIANTS = {
    "pm": "harbor_patch_memory.agents.openhands_pm:OpenHandsPatchMemory",
    "baseline": "harbor_patch_memory.agents.openhands_baseline:OpenHandsBaseline",
    "terminus2_pm": "harbor_patch_memory.agents.terminus2_pm:Terminus2PatchMemory",
    "terminus2_baseline": "harbor_patch_memory.agents.terminus2_baseline:Terminus2Baseline",
}

# Suffixes recognised as "this is a chain variant of <prefix>". Mirrors
# harbor_patch_memory.chain_id but expressed for filesystem discovery.
_EVO_RE = re.compile(r"^(?P<base>.+?)-EVO-\d+$", re.IGNORECASE)
# Variant token after ``-v-`` may include hyphens (``-v-registers-xyz``).
_V_RE = re.compile(r"^(?P<base>.+?)-v-[A-Za-z0-9_-]+$")


def _uses_patch_memory(variant: str) -> bool:
    return variant in {"pm", "terminus2_pm"}


def _has_kwarg(entries: list[str], key: str) -> bool:
    prefix = f"{key}="
    return any(entry.startswith(prefix) for entry in entries)


@dataclass
class TaskRun:
    chain_id: str
    task_name: str
    task_dir: Path
    trials_dir: Path
    log_path: Path
    cmd: list[str]
    env: dict[str, str]


def discover_chain_tasks(dataset: Path, chain_id: str) -> list[Path]:
    """Find prototype + variants for ``chain_id`` and return them in order."""
    prototype = dataset / chain_id
    if not prototype.is_dir():
        raise FileNotFoundError(f"Prototype task not found: {prototype}")

    variants: list[Path] = []
    for child in sorted(dataset.iterdir()):
        if not child.is_dir() or child == prototype:
            continue
        m_evo = _EVO_RE.match(child.name)
        m_v = _V_RE.match(child.name)
        if m_evo and m_evo.group("base") == chain_id:
            variants.append(child)
        elif m_v and m_v.group("base") == chain_id:
            variants.append(child)

    # Sort variants lexicographically; works for both -EVO-N (1..4) and
    # -v-<token>. Prototype always comes first.
    variants.sort(key=lambda p: p.name)
    return [prototype, *variants]


def build_run(
    chain_id: str,
    task_dir: Path,
    *,
    variant: str,
    trials_dir: Path,
    model: Optional[str],
    extra_env: list[str],
    extra_kwargs: list[str],
    extra_args: list[str],
    harbor_bin: str,
) -> TaskRun:
    task_name = task_dir.name
    log_path = trials_dir / "_logs" / f"{task_name}.harbor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    extra_kwargs = list(extra_kwargs)

    if _uses_patch_memory(variant) and not _has_kwarg(extra_kwargs, "host_root"):
        pm_host_root = trials_dir / "_pm_store"
        pm_host_root.mkdir(parents=True, exist_ok=True)
        extra_kwargs.append(f"host_root={pm_host_root}")

    cmd: list[str] = [
        harbor_bin,
        "trials",
        "start",
        "--path", str(task_dir),
        "--agent-import-path", VARIANTS[variant],
        "--trials-dir", str(trials_dir),
        "--ae", f"HARBOR_PM_CHAIN_ID={chain_id}",
    ]
    if model:
        cmd += ["--model", model]
    for entry in extra_env:
        cmd += ["--ae", entry]
    for entry in extra_kwargs:
        cmd += ["--agent-kwarg", entry]
    cmd += list(extra_args)

    env = os.environ.copy()
    env.setdefault("HARBOR_PM_CHAIN_ID", chain_id)

    return TaskRun(
        chain_id=chain_id,
        task_name=task_name,
        task_dir=task_dir,
        trials_dir=trials_dir,
        log_path=log_path,
        cmd=cmd,
        env=env,
    )


def run_one_task(run: TaskRun, *, dry_run: bool = False) -> dict:
    """Execute a single task (subprocess). Returns a status dict."""
    started = time.time()
    if dry_run:
        return {
            "chain_id": run.chain_id,
            "task_name": run.task_name,
            "skipped": True,
            "cmd": run.cmd,
            "elapsed_sec": 0.0,
        }
    with run.log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# cwd={os.getcwd()}\n# cmd={run.cmd}\n\n")
        logf.flush()
        proc = subprocess.run(
            run.cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=run.env,
            check=False,
        )
    elapsed = time.time() - started
    return {
        "chain_id": run.chain_id,
        "task_name": run.task_name,
        "exit_code": proc.returncode,
        "log_path": str(run.log_path),
        "elapsed_sec": round(elapsed, 1),
    }


def run_chain(
    chain_id: str,
    dataset: Path,
    *,
    variant: str,
    trials_dir: Path,
    model: Optional[str],
    extra_env: list[str],
    extra_kwargs: list[str],
    extra_args: list[str],
    harbor_bin: str,
    stop_on_failure: bool,
    dry_run: bool,
) -> list[dict]:
    tasks = discover_chain_tasks(dataset, chain_id)
    print(
        f"[chain {chain_id}] {len(tasks)} task(s): {', '.join(t.name for t in tasks)}",
        flush=True,
    )
    results: list[dict] = []
    for idx, task_dir in enumerate(tasks):
        run = build_run(
            chain_id=chain_id,
            task_dir=task_dir,
            variant=variant,
            trials_dir=trials_dir,
            model=model,
            extra_env=extra_env,
            extra_kwargs=extra_kwargs,
            extra_args=extra_args,
            harbor_bin=harbor_bin,
        )
        print(
            f"[chain {chain_id}] ({idx + 1}/{len(tasks)}) starting {run.task_name}",
            flush=True,
        )
        result = run_one_task(run, dry_run=dry_run)
        results.append(result)
        print(
            f"[chain {chain_id}] ({idx + 1}/{len(tasks)}) {run.task_name} -> "
            f"exit={result.get('exit_code')} elapsed={result.get('elapsed_sec')}s",
            flush=True,
        )
        if stop_on_failure and result.get("exit_code", 0) != 0 and not dry_run:
            print(
                f"[chain {chain_id}] aborting due to failure on {run.task_name}",
                flush=True,
            )
            break
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("~/qingchuan/Terminal-shift-main").expanduser(),
        help="Root directory containing per-task subdirectories.",
    )
    parser.add_argument(
        "--chain",
        action="append",
        required=True,
        help="Chain id (i.e. prototype task dir name). May be passed multiple times.",
    )
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANTS),
        default="pm",
        help="Which agent to use (default: pm).",
    )
    parser.add_argument(
        "--trials-dir",
        type=Path,
        default=Path("~/qingchuan/harbor_patch_memory/runs").expanduser(),
        help="Where Harbor trial outputs are stored.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional --model passthrough. Defaults to Harbor's default; the agent "
        "will fall back to OpenHands' config.toml [llm.swe_evo_pm] if unset.",
    )
    parser.add_argument(
        "--ae",
        action="append",
        default=[],
        help="Extra --ae KEY=VALUE entries forwarded to harbor trials start. "
        "Use this to pass LLM_API_KEY=... etc explicitly.",
    )
    parser.add_argument(
        "--kwarg",
        action="append",
        default=[],
        help="Extra --agent-kwarg key=value entries (forwarded verbatim).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Number of chains to run in parallel (each chain is itself sequential).",
    )
    parser.add_argument(
        "--harbor-bin",
        default=os.environ.get("HARBOR_BIN", "harbor"),
        help="Path to the harbor CLI (default: $HARBOR_BIN or 'harbor').",
    )
    parser.add_argument(
        "--no-stop-on-failure",
        action="store_true",
        help="Continue subsequent tasks in a chain even after one fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the harbor commands but don't execute them.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Anything after `--` is forwarded verbatim to `harbor trials start`.",
    )
    args = parser.parse_args()

    # argparse with REMAINDER also captures the trailing `--`; drop it.
    extra_args = [a for a in args.extra_args if a != "--"]

    args.trials_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.trials_dir / "_logs" / "chain_runner_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    chains = args.chain
    if args.max_parallel <= 1 or len(chains) == 1:
        all_results: dict[str, list[dict]] = {}
        for chain_id in chains:
            results = run_chain(
                chain_id=chain_id,
                dataset=args.dataset,
                variant=args.variant,
                trials_dir=args.trials_dir,
                model=args.model,
                extra_env=args.ae,
                extra_kwargs=args.kwarg,
                extra_args=extra_args,
                harbor_bin=args.harbor_bin,
                stop_on_failure=not args.no_stop_on_failure,
                dry_run=args.dry_run,
            )
            all_results[chain_id] = results
    else:
        all_results = {}
        with ProcessPoolExecutor(max_workers=args.max_parallel) as pool:
            futures = {
                pool.submit(
                    run_chain,
                    chain_id,
                    args.dataset,
                    variant=args.variant,
                    trials_dir=args.trials_dir,
                    model=args.model,
                    extra_env=args.ae,
                    extra_kwargs=args.kwarg,
                    extra_args=extra_args,
                    harbor_bin=args.harbor_bin,
                    stop_on_failure=not args.no_stop_on_failure,
                    dry_run=args.dry_run,
                ): chain_id
                for chain_id in chains
            }
            for fut in as_completed(futures):
                chain_id = futures[fut]
                try:
                    all_results[chain_id] = fut.result()
                except Exception as exc:
                    print(f"[chain {chain_id}] crashed: {exc}", file=sys.stderr)
                    all_results[chain_id] = [{"error": str(exc)}]

    summary_path.write_text(
        json.dumps(
            {
                "variant": args.variant,
                "dataset": str(args.dataset),
                "trials_dir": str(args.trials_dir),
                "chains": all_results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSummary -> {summary_path}", flush=True)

    failures = [
        (chain, task)
        for chain, tasks in all_results.items()
        for task in tasks
        if isinstance(task, dict) and task.get("exit_code") not in (0, None)
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
