"""Patch Memory subsystem (vendored from swe_evo_patch_memory).

Originally adapted from EvoClaw `harness/e2e/patch_memory*.py`. Copied
here so this package is self-contained and Harbor can import it without
depending on the SWE-EVO source tree.

The only mechanical change vs the SWE-EVO copy is the trajectory module
filename: `trajectory_extractor_swe_evo` -> `trajectory_extractor`. The
SWE-EVO-flavoured extractor still ingests OpenHands `history` event
lists, which is exactly what Harbor's OpenHands writes to
`/logs/agent/openhands.trajectory.json` via `SAVE_TRAJECTORY_PATH`.
"""

from .manager import (
    FeaturePatchRecord,
    MilestonePatchBundle,
    PatchMemoryManager,
)
from .retriever import PatchMemoryRetriever
from .summarizer import PatchMemorySummarizer
from .trajectory_extractor import (
    EditIntentContext,
    SweEvoTrajectoryExtractor,
    TrajectoryEvent,
    TrajectorySlice,
    ValidationEpisode,
    ValidationTest,
)

__all__ = [
    "FeaturePatchRecord",
    "MilestonePatchBundle",
    "PatchMemoryManager",
    "PatchMemoryRetriever",
    "PatchMemorySummarizer",
    "SweEvoTrajectoryExtractor",
    "TrajectoryEvent",
    "TrajectorySlice",
    "ValidationTest",
    "ValidationEpisode",
    "EditIntentContext",
]
