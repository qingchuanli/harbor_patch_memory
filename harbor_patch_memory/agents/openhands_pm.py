"""Harbor agent: OpenHands + EvoClaw-style Patch Memory.

Lifecycle inside one Harbor trial:

1. Resolve the chain ID from the task name (or ``HARBOR_PM_CHAIN_ID``).
2. Build a chain-rooted :class:`PatchMemoryManager` whose state lives at
   ``~/.harbor_patch_memory/<chain_id>/patch_memory/``.
3. Render an "edit guardrail" markdown block from records belonging to
   prior trials in this chain, prepend it to the task instruction.
4. Init a scratch git repo at ``/app`` and commit the baseline.
5. Run upstream OpenHands as usual (same install / trajectory / metrics
   path as Harbor's stock OpenHands agent).
6. Capture ``git diff`` against the baseline; load the OpenHands
   ``openhands.trajectory.json`` (== SWE-EVO's ``output.jsonl[history]``);
   feed both into ``PatchMemoryManager.store_submission_feature_patches``,
   which calls the LLM summariser to produce structured
   ``FeaturePatchRecord``s, indexes them, and re-renders the chain
   memory README.

The injected memory is the *summarised* descriptive patch (intent /
constraints / behavior + small diff hunks) — not the raw diff. The raw
diff is recorded inside each record but is not pasted into the next
prompt verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .. import memory_bridge, patch_capture
from ..chain_id import derive_chain_id
from ..memory import PatchMemoryManager
from ..memory_bridge import LLMConfig, resolve_llm_config
from .openhands_baseline import OpenHandsBaseline


MEMORY_BLOCK_OPEN = "<!-- BEGIN PATCH MEMORY (injected by harbor_patch_memory) -->"
MEMORY_BLOCK_CLOSE = "<!-- END PATCH MEMORY -->"


def _inject_memory(problem: str, memory_md: str) -> str:
    """Prepend ``memory_md`` to ``problem`` between explicit fences."""
    body = (memory_md or "").strip()
    if not body:
        return problem
    return f"{MEMORY_BLOCK_OPEN}\n{body}\n{MEMORY_BLOCK_CLOSE}\n\n{problem}"


class OpenHandsPatchMemory(OpenHandsBaseline):
    """OpenHands wrapped with chain-scoped Patch Memory.

    Init kwargs (additional to :class:`OpenHandsBaseline`):
      * ``chain_id`` — manual override of the auto-detected chain id.
        Same effect as setting ``HARBOR_PM_CHAIN_ID`` via ``--ae``.
      * ``host_root`` — host directory for chain-scoped memory state.
        Defaults to ``~/.harbor_patch_memory``.
      * ``char_budget`` — soft cap on the rendered memory markdown that
        gets prepended to the prompt. Defaults to 3500.
      * ``oh_config_path`` / ``oh_llm_block`` — same as the baseline,
        but the resolved LLM is also handed to the Patch Memory
        summariser, not just to OpenHands itself.
    """

    def __init__(
        self,
        chain_id: Optional[str] = None,
        host_root: Optional[str] = None,
        char_budget: int = 3500,
        *args: Any,
        **kwargs: Any,
    ):
        self._explicit_chain_id = chain_id
        self._host_root = host_root
        self._char_budget = int(char_budget)
        self._chain_id: Optional[str] = None
        self._manager: Optional[PatchMemoryManager] = None
        self._task_name: Optional[str] = None
        self._injected_instruction: Optional[str] = None
        self._original_instruction: Optional[str] = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Lazy manager construction (needs the env name from the trial)
    # ------------------------------------------------------------------

    def _ensure_manager(self, task_name: Optional[str]) -> Optional[PatchMemoryManager]:
        # Resolution priority for chain id (highest first):
        #   1. Explicit ``chain_id`` kwarg passed via --agent-kwarg.
        #   2. ``HARBOR_PM_CHAIN_ID`` value supplied to Harbor via --ae
        #      (lands in self._extra_env, not os.environ on the host).
        #   3. ``HARBOR_PM_CHAIN_ID`` already in os.environ (set by the
        #      chain runner before spawning the harbor subprocess).
        #   4. Auto-detection from the task name suffix.
        extra_env = self._extra_env or {}
        explicit = self._explicit_chain_id or extra_env.get("HARBOR_PM_CHAIN_ID")
        host_root = self._host_root or extra_env.get("HARBOR_PM_HOST_ROOT")
        chain = derive_chain_id(task_name, explicit)
        if not chain:
            self.logger.info(
                "patch_memory: no chain id resolvable (task=%s); skipping memory",
                task_name,
            )
            return None
        if self._manager is not None and self._chain_id == chain:
            return self._manager

        cfg: LLMConfig = resolve_llm_config(
            explicit_model=getattr(self, "model_name", None),
            extra_env=extra_env,
            oh_config_path=self._oh_config_path,
            oh_config_block=self._oh_llm_block,
        )
        self._chain_id = chain
        self._manager = memory_bridge.build_manager(
            chain_id=chain,
            repo_name=chain,
            host_root=host_root,
            llm=cfg,
            logger=self.logger,
        )
        return self._manager

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    async def run(  # type: ignore[override]
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._task_name = environment.environment_name
        manager = self._ensure_manager(self._task_name)

        rendered = ""
        if manager is not None:
            try:
                rendered = memory_bridge.render_memory_for_problem(
                    manager,
                    problem_statement=instruction,
                    milestone_id=self._task_name or "task",
                    char_budget=self._char_budget,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("patch_memory render failed: %s", exc)
                rendered = ""

        self._original_instruction = instruction
        self._injected_instruction = _inject_memory(instruction, rendered)
        self._dump_pre_run_artefacts(rendered)

        # Hand the (possibly augmented) instruction to the baseline run,
        # which performs the git baseline / OpenHands run / diff capture.
        try:
            await super().run(
                instruction=self._injected_instruction,
                environment=environment,
                context=context,
            )
        finally:
            # Always ingest whatever trajectory exists so the chain still
            # learns from an agent that crashed mid-run.
            try:
                self._ingest_into_memory(manager)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("patch_memory ingest failed: %s", exc)

    def _dump_pre_run_artefacts(self, rendered: str) -> None:
        out = self.harbor_pm_dir
        try:
            meta = {
                "task_name": self._task_name,
                "chain_id": self._chain_id,
                "memory_chars": len(rendered or ""),
                "char_budget": self._char_budget,
                "injection": "patch_memory_v1" if rendered else "none",
            }
            (out / "pre_run_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            if rendered:
                (out / "rendered_memory.md").write_text(rendered, encoding="utf-8")
            if self._original_instruction is not None:
                (out / "original_instruction.md").write_text(
                    self._original_instruction, encoding="utf-8"
                )
            if self._injected_instruction is not None:
                (out / "injected_instruction.md").write_text(
                    self._injected_instruction, encoding="utf-8"
                )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("failed to dump pre-run artefacts: %s", exc)

    def _ingest_into_memory(self, manager: Optional[PatchMemoryManager]) -> None:
        if manager is None:
            return
        captured = self.captured
        if captured is None:
            return

        history = self._load_history()
        memory_bridge.store_submission_records(
            manager,
            chain_id=self._chain_id or "unknown",
            task_name=self._task_name or "task",
            history=history,
            diff_text=captured.diff_text,
            changed_files=captured.changed_files,
            problem_statement=self._original_instruction or "",
            head_sha=captured.head_sha,
            attempt=0,
        )

    def _load_history(self) -> list[dict[str, Any]]:
        # Harbor's OpenHands writes SAVE_TRAJECTORY_PATH=/logs/agent/openhands.trajectory.json,
        # which the host sees under self.logs_dir.
        candidate = Path(self.logs_dir) / "openhands.trajectory.json"
        events = memory_bridge.load_openhands_history(candidate)
        if events:
            return events
        # Fallback: walk events/*.json under the session dir.
        try:
            sessions_dir = Path(self.logs_dir) / "sessions"
            if not sessions_dir.exists():
                return []
            sess_dirs = [p for p in sessions_dir.iterdir() if p.is_dir()]
            if len(sess_dirs) != 1:
                return []
            events_dir = sess_dirs[0] / "events"
            if not events_dir.exists():
                return []
            files = sorted(events_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
            history: list[dict[str, Any]] = []
            for f in files:
                try:
                    history.append(json.loads(f.read_text(encoding="utf-8")))
                except Exception:
                    continue
            return history
        except Exception:
            return []
