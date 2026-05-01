"""Baseline Harbor agent: stock OpenHands + git-diff capture, no Patch Memory.

The point of this agent is to give us an apples-to-apples comparison
target for :class:`OpenHandsPatchMemory`. Both agents run the same
upstream OpenHands binary, install the same way, and write the same
ATIF trajectory. The only thing the baseline does extra is bracket the
agent run with a one-shot git baseline + post-run diff so we record
*what the agent changed* in the trial directory:

    /logs/agent/harbor_pm/diff.patch
    /logs/agent/harbor_pm/changed_files.txt
    /logs/agent/harbor_pm/meta.json

The PM agent reuses these same artefacts. Outside of that, this agent
is effectively just :class:`harbor.agents.installed.openhands.OpenHands`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from harbor.agents.installed.openhands import OpenHands
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .. import patch_capture
from ..memory_bridge import LLMConfig, resolve_llm_config


class OpenHandsBaseline(OpenHands):
    """OpenHands + git-baseline capture, intentionally without Patch Memory.

    Init kwargs (all optional, all forwarded transparently):
      * ``workdir`` — work directory inside the container where the
        scratch git repo lives. Defaults to ``/app`` (Terminal-shift's
        Dockerfile WORKDIR).
      * ``oh_config_path`` / ``oh_llm_block`` — where to read the
        OpenHands ``[llm.<block>]`` fallback (used for LLM_API_KEY /
        LLM_MODEL / LLM_BASE_URL when Harbor was started without
        ``--ae``).
      * Everything else flows to :class:`harbor.agents.installed.openhands.OpenHands`.
    """

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
        # Resolve LLM fallback from OpenHands config.toml if Harbor's CLI
        # didn't pass any model / api key. We splice the resolved values
        # back into self._extra_env so OpenHands' own run() resolution
        # picks them up via _get_env / _has_env without further changes.
        self._apply_llm_fallback()

    @staticmethod
    def name() -> str:  # type: ignore[override]
        # Reuse the canonical OpenHands agent name so Harbor's trajectory
        # tooling treats us as an OpenHands variant.
        return OpenHands.name()

    # ------------------------------------------------------------------
    # LLM fallback resolution from OpenHands config.toml
    # ------------------------------------------------------------------

    def _apply_llm_fallback(self) -> None:
        cfg: LLMConfig = resolve_llm_config(
            explicit_model=getattr(self, "model_name", None),
            extra_env=self._extra_env,
            oh_config_path=self._oh_config_path,
            oh_config_block=self._oh_llm_block,
        )
        # Populate model_name on self if Harbor didn't set one
        if cfg.model and not getattr(self, "model_name", None):
            self.model_name = cfg.model  # OpenHands.run() reads self.model_name
            # Refresh parsed model info so to_agent_info() reflects the fallback.
            try:
                self._init_model_info()  # type: ignore[attr-defined]
            except Exception:
                pass
        if cfg.api_key and "LLM_API_KEY" not in self._extra_env:
            self._extra_env["LLM_API_KEY"] = cfg.api_key
        if cfg.model and "LLM_MODEL" not in self._extra_env:
            self._extra_env["LLM_MODEL"] = cfg.model
        if cfg.base_url and "LLM_BASE_URL" not in self._extra_env:
            self._extra_env["LLM_BASE_URL"] = cfg.base_url

        # OpenAI reasoning models in the gpt-5.x family currently reject
        # ``function tools`` combined with ``reasoning_effort`` on
        # ``/v1/chat/completions`` (the API redirects to /v1/responses).
        # Switching OpenHands to prompted (non-native) tool calling
        # avoids the conflict and keeps the agent working unchanged
        # against any OpenAI / OpenAI-compatible endpoint.
        # Caller can override with --ae LLM_NATIVE_TOOL_CALLING=true.
        self._extra_env.setdefault("LLM_NATIVE_TOOL_CALLING", "false")
        # And let litellm silently drop other unsupported params instead
        # of bubbling 400s for non-OpenAI proxies.
        self._extra_env.setdefault("LLM_DROP_PARAMS", "true")

    # ------------------------------------------------------------------
    # Helpers shared with the PM subclass
    # ------------------------------------------------------------------

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
        except Exception as exc:  # noqa: BLE001 — best-effort
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

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    async def run(  # type: ignore[override]
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Bracket OpenHands' run() with git baseline + diff capture."""
        baseline_sha = await self._setup_baseline(environment)

        # Delegate to the upstream OpenHands.run; let its decorator
        # render any prompt template the user supplied.
        try:
            await super().run(
                instruction=instruction,
                environment=environment,
                context=context,
            )
        finally:
            self._capture = await self._capture_post_run(environment, baseline_sha)

    # Expose the captured artefact to subclasses without leaking it to
    # callers that depend on the OpenHands return shape.
    @property
    def captured(self) -> Optional[patch_capture.CapturedDiff]:
        return self._capture
