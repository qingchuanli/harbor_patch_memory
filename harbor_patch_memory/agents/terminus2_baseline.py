"""Baseline Harbor agent: Terminus2 + git-diff capture, no Patch Memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from harbor.agents.terminus_2 import Terminus2
from harbor.agents.utils import get_api_key_var_names_from_model_name
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .. import patch_capture
from ..memory_bridge import LLMConfig, resolve_llm_config


class Terminus2Baseline(Terminus2):
    """Terminus2 bracketed by scratch-git baseline/diff capture."""

    def __init__(
        self,
        workdir: str = patch_capture.DEFAULT_WORKDIR,
        oh_config_path: Optional[str] = None,
        oh_llm_block: str = "swe_evo_pm",
        *args: Any,
        **kwargs: Any,
    ):
        self._workdir = workdir
        self._oh_config_path = oh_config_path
        self._oh_llm_block = oh_llm_block
        self._capture: Optional[patch_capture.CapturedDiff] = None
        super().__init__(*args, **kwargs)
        self._apply_llm_fallback()

    def _apply_llm_fallback(self) -> None:
        extra_env = dict(self._extra_env or {})
        cfg: LLMConfig = resolve_llm_config(
            explicit_model=getattr(self, "model_name", None),
            extra_env=extra_env,
            oh_config_path=self._oh_config_path,
            oh_config_block=self._oh_llm_block,
        )

        if cfg.api_key:
            self._llm_kwargs = dict(getattr(self, "_llm_kwargs", None) or {})
            self._llm_kwargs.setdefault("api_key", cfg.api_key)
            llm_kwargs = dict(getattr(self._llm, "_llm_kwargs", None) or {})
            llm_kwargs.setdefault("api_key", cfg.api_key)
            self._llm._llm_kwargs = llm_kwargs  # type: ignore[attr-defined]

            merged_env = dict(self._extra_env or {})
            merged_env.setdefault("LLM_API_KEY", cfg.api_key)
            if self.model_name:
                try:
                    for env_var in get_api_key_var_names_from_model_name(self.model_name):
                        merged_env.setdefault(env_var, cfg.api_key)
                except Exception:
                    merged_env.setdefault("OPENAI_API_KEY", cfg.api_key)
            self._extra_env = merged_env

        if cfg.base_url and not getattr(self._llm, "_api_base", None):
            self._llm._api_base = cfg.base_url  # type: ignore[attr-defined]

    @property
    def harbor_pm_dir(self) -> Path:
        out = self.logs_dir / "harbor_pm"
        out.mkdir(parents=True, exist_ok=True)
        return out

    async def _setup_baseline(self, environment: BaseEnvironment) -> Optional[str]:
        try:
            return await patch_capture.init_baseline(
                environment, workdir=self._workdir, logger=self.logger
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("patch baseline setup failed: %s", exc)
            return None

    async def _capture_post_run(
        self, environment: BaseEnvironment, baseline_sha: Optional[str]
    ) -> patch_capture.CapturedDiff:
        try:
            captured = await patch_capture.capture_diff(
                environment, workdir=self._workdir, logger=self.logger
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("patch capture failed: %s", exc)
            captured = patch_capture.CapturedDiff(
                diff_text="",
                changed_files=[],
                baseline_sha=baseline_sha,
                head_sha=None,
            )
        self._dump_capture(captured)
        return captured

    def _dump_capture(self, captured: patch_capture.CapturedDiff) -> None:
        out = self.harbor_pm_dir
        try:
            (out / "diff.patch").write_text(captured.diff_text or "", encoding="utf-8")
            (out / "changed_files.txt").write_text(
                "\n".join(captured.changed_files) + ("\n" if captured.changed_files else ""),
                encoding="utf-8",
            )
            meta = {
                "workdir": self._workdir,
                "baseline_sha": captured.baseline_sha,
                "head_sha": captured.head_sha,
                "n_changed_files": len(captured.changed_files),
                "diff_chars": len(captured.diff_text or ""),
            }
            (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("failed to dump capture artefacts: %s", exc)

    async def run(  # type: ignore[override]
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        baseline_sha = await self._setup_baseline(environment)
        try:
            await super().run(
                instruction=instruction,
                environment=environment,
                context=context,
            )
        finally:
            self._capture = await self._capture_post_run(environment, baseline_sha)

    @property
    def captured(self) -> Optional[patch_capture.CapturedDiff]:
        return self._capture
