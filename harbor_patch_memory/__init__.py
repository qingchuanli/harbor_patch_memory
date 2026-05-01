"""Harbor agent wrappers with descriptive Patch Memory integration.

Four Harbor agents are exposed:

* :class:`harbor_patch_memory.agents.openhands_pm.OpenHandsPatchMemory`
  — OpenHands wrapped with EvoClaw-style descriptive Patch Memory.
* :class:`harbor_patch_memory.agents.openhands_baseline.OpenHandsBaseline`
  — Plain OpenHands with the same git-baseline / git-diff capture so
    its outputs line up with the PM variant for ablation.
* :class:`harbor_patch_memory.agents.terminus2_pm.Terminus2PatchMemory`
  — Terminus2 wrapped with the same chain-scoped Patch Memory flow.
* :class:`harbor_patch_memory.agents.terminus2_baseline.Terminus2Baseline`
  — Plain Terminus2 with matching git-baseline / git-diff capture.

The OpenHands wrappers subclass Harbor's stock
:class:`harbor.agents.installed.openhands.OpenHands`; the Terminus2 wrappers
subclass :class:`harbor.agents.terminus_2.terminus_2.Terminus2`.

Use them by passing ``--agent-import-path`` to ``harbor trials start``::

    harbor trials start \
        --path ~/qingchuan/Terminal-shift-main/bn-fit-modify \
        --agent-import-path harbor_patch_memory.agents.openhands_pm:OpenHandsPatchMemory \
        --ae HARBOR_PM_CHAIN_ID=bn-fit-modify
"""

__all__ = ["agents", "memory", "chain_id", "memory_bridge", "patch_capture"]
