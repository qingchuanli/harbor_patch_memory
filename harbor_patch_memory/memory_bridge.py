"""Glue between Harbor's per-trial agent run and chain-scoped Patch Memory.

Harbor runs each task in an isolated trial directory, but Patch Memory
needs state to persist *across* trials in the same chain (prototype +
variants). We therefore park the chain-scoped PM state on the host file
system at::

    ~/.harbor_patch_memory/<chain_id>/patch_memory/{records,bundles,...}

and instantiate :class:`PatchMemoryManager` rooted there each time an
agent runs. The manager itself is dataclass+JSON; nothing is held in
memory across trials, so this works regardless of how many parallel
chains share the host.

LLM credentials for the summariser are resolved with the precedence
agreed with the user::

    explicit kwargs (passed in by the agent constructor)
        > extra_env / os.environ (for LLM_API_KEY / LLM_MODEL / LLM_BASE_URL)
        > ``[llm.swe_evo_pm]`` in ``$OPENHANDS_CONFIG_TOML`` (default
          ``~/qingchuan/OpenHands_upstream/config.toml``)

That mirrors the SWE-EVO + Patch Memory configuration from the user's
existing setup.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:  # py3.11+
    import tomllib
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from .memory import PatchMemoryManager
from .memory.trajectory_extractor import EditIntentContext, SweEvoTrajectoryExtractor


DEFAULT_OH_CONFIG = Path("~/qingchuan/OpenHands_upstream/config.toml").expanduser()
DEFAULT_LLM_BLOCK = "swe_evo_pm"
DEFAULT_HOST_ROOT = Path("~/.harbor_patch_memory").expanduser()


@dataclass
class LLMConfig:
    """Resolved summariser config (None values fall back inside the summariser)."""

    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


def load_llm_from_oh_config(
    config_path: Path | str | None = None,
    block: str = DEFAULT_LLM_BLOCK,
) -> LLMConfig:
    """Read ``[llm.<block>]`` from an OpenHands ``config.toml``.

    Returns an empty :class:`LLMConfig` if the file is missing or the
    block is absent. ``base_url=""`` (the user's default for native
    OpenAI) is treated as ``None``.
    """
    path = Path(config_path).expanduser() if config_path else DEFAULT_OH_CONFIG
    if not path.exists():
        return LLMConfig()
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        return LLMConfig()
    llm_section = data.get("llm")
    if not isinstance(llm_section, dict):
        return LLMConfig()
    block_section = llm_section.get(block)
    if not isinstance(block_section, dict):
        return LLMConfig()

    def _val(key: str) -> Optional[str]:
        v = block_section.get(key)
        if v is None:
            return None
        if isinstance(v, str):
            return v.strip() or None
        return str(v)

    base_url = _val("base_url")
    return LLMConfig(
        model=_val("model"),
        api_key=_val("api_key"),
        base_url=base_url,
    )


def resolve_llm_config(
    *,
    explicit_model: Optional[str] = None,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    extra_env: Optional[dict[str, str]] = None,
    oh_config_path: Path | str | None = None,
    oh_config_block: str = DEFAULT_LLM_BLOCK,
) -> LLMConfig:
    """Compose the summariser LLM config with the agreed precedence.

    Priority (highest first):

    1. Explicit kwargs on the agent.
    2. Process / agent ``extra_env`` (``LLM_API_KEY``, ``LLM_MODEL``,
       ``LLM_BASE_URL``) — same names Harbor's stock OpenHands agent
       respects.
    3. ``[llm.<block>]`` in the OpenHands ``config.toml``.
    """
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    fallback = load_llm_from_oh_config(oh_config_path, oh_config_block)

    model = explicit_model or env.get("LLM_MODEL") or fallback.model
    api_key = explicit_api_key or env.get("LLM_API_KEY") or fallback.api_key
    base_url = explicit_base_url or env.get("LLM_BASE_URL") or fallback.base_url

    return LLMConfig(model=model, api_key=api_key, base_url=base_url)


def chain_root(chain_id: str, host_root: Path | str | None = None) -> Path:
    root = Path(host_root).expanduser() if host_root else DEFAULT_HOST_ROOT
    if "HARBOR_PM_HOST_ROOT" in os.environ and host_root is None:
        root = Path(os.environ["HARBOR_PM_HOST_ROOT"]).expanduser()
    safe = chain_id.replace("/", "_").replace("..", "_")
    target = root / safe
    target.mkdir(parents=True, exist_ok=True)
    return target


def build_manager(
    chain_id: str,
    *,
    repo_name: Optional[str] = None,
    host_root: Path | str | None = None,
    llm: Optional[LLMConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> PatchMemoryManager:
    """Instantiate a :class:`PatchMemoryManager` rooted at the chain dir."""
    trial_root = chain_root(chain_id, host_root)
    repo = repo_name or chain_id
    cfg = llm or LLMConfig()
    if logger:
        logger.info(
            "Patch memory manager: chain=%s root=%s model=%s",
            chain_id,
            trial_root,
            cfg.model,
        )
    return PatchMemoryManager(
        trial_root=trial_root,
        repo_name=repo,
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    )


def render_memory_for_problem(
    manager: PatchMemoryManager,
    problem_statement: str,
    milestone_id: str,
    char_budget: int = 3500,
) -> str:
    """Render markdown to prepend to the next instance's prompt.

    Mirrors :meth:`PatchMemoryManager.render_context_from_problem_statement`
    on a single statement so callers don't have to wrap it in a list.
    Returns ``""`` if nothing is retrieved (caller should not inject a
    block in that case).
    """
    md = manager.render_context_from_problem_statement(
        problem_statements=[problem_statement],
        milestone_ids=[milestone_id],
        char_budget=char_budget,
    )
    if not md or "No overlapping" in md:
        return ""
    return md


def load_openhands_history(trajectory_json: Path | str) -> list[dict[str, Any]]:
    """Load the event list OpenHands writes via ``SAVE_TRAJECTORY_PATH``.

    Harbor's stock OpenHands agent sets
    ``SAVE_TRAJECTORY_PATH=/logs/agent/openhands.trajectory.json``. The
    file is the JSON-serialised output of ``controller.get_trajectory()``
    — i.e. exactly what SWE-EVO sees as ``output.jsonl[history]``.
    """
    path = Path(trajectory_json)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    if isinstance(data, dict):
        # Defensive: occasionally the file is wrapped {"events": [...]}.
        for key in ("history", "events", "trajectory"):
            value = data.get(key)
            if isinstance(value, list):
                return [e for e in value if isinstance(e, dict)]
    return []


def _flatten_atif_content(value: Any) -> str:
    """Coerce an ATIF content field into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(value)


def load_terminus2_history(trajectory_json: Path | str) -> list[dict[str, Any]]:
    """Translate a Terminus2 ATIF trajectory into patch-memory history events.

    Patch Memory's extractor expects an OpenHands/SWE-EVO-style event list.
    Terminus2 writes ATIF ``trajectory.json`` instead, so we synthesize a
    compatible event stream from its steps:

    - agent text/reasoning -> ``source=agent`` message events
    - ``bash_command`` tool calls -> ``source=agent`` terminal actions
    - observations -> ``source=environment`` terminal results
    """
    path = Path(trajectory_json)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list):
        return []

    history: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue

        timestamp = step.get("timestamp")
        source = str(step.get("source") or "").lower()
        message = _flatten_atif_content(step.get("message"))
        reasoning = _flatten_atif_content(step.get("reasoning_content"))

        if source in {"system", "user"}:
            if message:
                history.append(
                    {
                        "timestamp": timestamp,
                        "source": source,
                        "type": "message",
                        "content": message,
                    }
                )
            continue

        if source != "agent":
            continue

        if message or reasoning:
            event: dict[str, Any] = {
                "timestamp": timestamp,
                "source": "agent",
                "type": "message",
                "content": message,
            }
            if reasoning:
                event["thought"] = reasoning
            history.append(event)

        bash_commands: list[str] = []
        for tool_call in step.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            fn = str(tool_call.get("function_name") or "")
            args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
            if fn == "bash_command":
                command = str(args.get("keystrokes") or "").strip()
                if not command:
                    continue
                bash_commands.append(command)
                history.append(
                    {
                        "timestamp": timestamp,
                        "source": "agent",
                        "tool_name": "terminal",
                        "type": "tool_call",
                        "content": command,
                        "action": {"command": command},
                    }
                )
            elif fn == "mark_task_complete":
                history.append(
                    {
                        "timestamp": timestamp,
                        "source": "agent",
                        "type": "message",
                        "content": "task_complete",
                    }
                )

        observation = step.get("observation")
        if not isinstance(observation, dict):
            continue
        results = observation.get("results")
        if not isinstance(results, list):
            continue
        rendered_results = [
            _flatten_atif_content(result.get("content"))
            for result in results
            if isinstance(result, dict)
        ]
        rendered_results = [text for text in rendered_results if text]
        if not rendered_results:
            continue
        combined_output = "\n\n".join(rendered_results)
        history.append(
            {
                "timestamp": timestamp,
                "source": "environment",
                "tool_name": "terminal",
                "type": "tool_result",
                "content": combined_output,
                "observation": {
                    "command": " && ".join(bash_commands) if bash_commands else None,
                    "content": [{"text": combined_output}],
                    "exit_code": None,
                    "is_error": False,
                },
            }
        )
    return history


def _filter_changed_files(candidates: list[str]) -> list[str]:
    """Drop apt / URL / system noise from a candidate ``changed_files`` list.

    Mirrors the noise-filter the trajectory extractor applies on its way
    in, but is run again here as a *defence-in-depth* pass. Callers may
    have built ``candidates`` from sources that bypass the extractor
    (for example, a raw ``git diff --name-only`` over a workdir that
    happens to track ``.git/`` internals because we ran ``git init`` for
    diff capture).
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in candidates or []:
        if not isinstance(raw, str):
            continue
        path = raw.strip()
        # Normalise common relative-path prefixes without eating real
        # leading dots (we must keep ``.git/HEAD`` as ``.git/HEAD`` so
        # the ``.git/`` prefix in ``_NON_PATH_PREFIXES`` still matches).
        if path.startswith("./"):
            path = path[2:]
        if not path or path in seen:
            continue
        if SweEvoTrajectoryExtractor._is_noisy_path(path):
            continue
        seen.add(path)
        cleaned.append(path)
    return cleaned


def store_submission_records(
    manager: PatchMemoryManager,
    *,
    chain_id: str,
    task_name: str,
    history: list[dict[str, Any]],
    diff_text: str,
    changed_files: list[str],
    problem_statement: str,
    head_sha: Optional[str] = None,
    attempt: int = 0,
) -> int:
    """Feed a finished trial into the patch-memory pipeline.

    The OpenHands trajectory is the **primary** signal — diff and changed
    files are merely supplementary. So we proceed even when no diff was
    captured (Terminal-shift-style tasks often touch state that ``git``
    cannot see, e.g. produced CSVs, or the agent might run before any
    file write). When git capture missed the files but the trajectory
    didn't, we derive ``changed_files`` from
    ``trajectory_slice.files_touched`` so the summariser still has a
    scoping signal.

    All candidate ``changed_files`` (whether from the in-container ``git
    diff`` or from the trajectory) are run through
    :func:`_filter_changed_files` to drop apt / system / archive paths
    that previously polluted the chain. When *nothing* survives the
    filter we still call the manager — it will trigger the milestone-
    fallback path which writes exactly one record summarising the run
    instead of dozens of spurious feature groups.

    Returns the number of ``FeaturePatchRecord``s written.
    """
    manager.load_instance_history(history)

    # Defence-in-depth: even files coming from ``git diff`` can be noisy
    # if the workdir tracks ``.git/`` internals or apt-cache directories.
    changed_files = _filter_changed_files(changed_files)

    # Augment changed_files with trajectory-derived files_touched when
    # the in-container git capture saw nothing. The trajectory extractor
    # already filters the noise, but we filter again to stay consistent
    # in case callers pre-stripped or transformed the slice.
    if not changed_files:
        slice_ = manager.trajectory_extractor.get_recent_slice_for_submission(
            session_id=task_name,
            changed_files=[],
            tag_hash=head_sha,
        )
        if slice_ and slice_.files_touched:
            changed_files = _filter_changed_files(list(slice_.files_touched))

    records = manager.store_submission_feature_patches(
        milestone_id=task_name,
        attempt=attempt,
        agent_tag=f"harbor:{task_name}",
        tag_hash=head_sha,
        changed_files=changed_files,
        diff_text=diff_text,
        srs_text=problem_statement,
        session_id=task_name,
    )
    return len(records or [])


__all__ = [
    "DEFAULT_OH_CONFIG",
    "DEFAULT_LLM_BLOCK",
    "DEFAULT_HOST_ROOT",
    "LLMConfig",
    "load_llm_from_oh_config",
    "resolve_llm_config",
    "chain_root",
    "build_manager",
    "render_memory_for_problem",
    "load_openhands_history",
    "load_terminus2_history",
    "store_submission_records",
    "EditIntentContext",
]
