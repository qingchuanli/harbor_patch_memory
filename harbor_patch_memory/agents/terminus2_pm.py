"""Harbor agent: Terminus2 + EvoClaw-style Patch Memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .. import memory_bridge
from ..chain_id import derive_chain_id
from ..memory import PatchMemoryManager
from ..memory_bridge import LLMConfig, resolve_llm_config
from .terminus2_baseline import Terminus2Baseline


MEMORY_BLOCK_OPEN = "<!-- BEGIN PATCH MEMORY (injected by harbor_patch_memory) -->"
MEMORY_BLOCK_CLOSE = "<!-- END PATCH MEMORY -->"


def _inject_memory(problem: str, memory_md: str) -> str:
    body = (memory_md or "").strip()
    if not body:
        return problem
    return f"{MEMORY_BLOCK_OPEN}\n{body}\n{MEMORY_BLOCK_CLOSE}\n\n{problem}"


class Terminus2PatchMemory(Terminus2Baseline):
    """Terminus2 wrapped with chain-scoped Patch Memory."""

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

    def _ensure_manager(self, task_name: Optional[str]) -> Optional[PatchMemoryManager]:
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

        try:
            await super().run(
                instruction=self._injected_instruction,
                environment=environment,
                context=context,
            )
        finally:
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
        if manager is None or self.captured is None:
            return
        history = self._load_history()
        memory_bridge.store_submission_records(
            manager,
            chain_id=self._chain_id or "unknown",
            task_name=self._task_name or "task",
            history=history,
            diff_text=self.captured.diff_text,
            changed_files=self.captured.changed_files,
            problem_statement=self._original_instruction or "",
            head_sha=self.captured.head_sha,
            attempt=0,
        )

    def _load_history(self) -> list[dict[str, Any]]:
        candidate = Path(self.logs_dir) / "trajectory.json"
        return memory_bridge.load_terminus2_history(candidate)
