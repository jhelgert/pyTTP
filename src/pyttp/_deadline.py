"""A wall-clock budget that can be shared between the phases of the solver."""

from __future__ import annotations

import math
import time


class Deadline:
    """Tracks the time left of an optional budget. ``None`` means unlimited."""

    def __init__(self, seconds: float | None) -> None:
        self._end = None if seconds is None else time.monotonic() + seconds

    @property
    def expired(self) -> bool:
        return self._end is not None and time.monotonic() >= self._end

    def remaining(self) -> float:
        """Seconds left (``inf`` if unlimited, never negative)."""
        if self._end is None:
            return math.inf
        return max(0.0, self._end - time.monotonic())
