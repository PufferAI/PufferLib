"""Legacy post-processing wrappers for backward compatibility.

This module keeps the historical ``pufferlib.postprocess`` import path alive by
re-exporting the wrappers that now live in ``pufferlib.pufferlib``.
"""

from pufferlib.pufferlib import (
    ClipAction,
    EpisodeStats,
    MeanOverAgents,
    MultiagentEpisodeStats,
    PettingZooWrapper,
)

__all__ = [
    "ClipAction",
    "EpisodeStats",
    "MeanOverAgents",
    "MultiagentEpisodeStats",
    "PettingZooWrapper",
]
