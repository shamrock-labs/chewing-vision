"""Abstract engine base class for chewing analysis backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from chewing.types import Result


class EngineBase(ABC):
    """Common interface every chewing engine implements.

    Engines take a local video path and return a unified Result with
    detected chew events, window-level labels, and summary statistics.
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Stable identifier for this engine (e.g. ``ours``, ``orofac``)."""

    @abstractmethod
    def analyze(
        self,
        video_path: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> Result:
        """Run chewing analysis on a video file.

        Args:
            video_path: Path to a local video file.
            start: Optional start time in seconds.
            end: Optional end time in seconds.

        Returns:
            Populated Result.
        """
