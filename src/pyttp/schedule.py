"""Helpers to build, check and display schedules.

A schedule is an ``int32`` array of shape ``(n, 2n - 2)``. Entry ``[t, r]``
describes the game of team ``t`` (0-based) in round ``r`` (0-based): its absolute
value minus one is the opponent, a positive sign marks a home game, a negative
sign an away game.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["Schedule", "Solution", "canonical_schedule", "format_schedule", "print_schedule"]

Schedule: TypeAlias = NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class Solution:
    """A feasible schedule together with its total travel distance."""

    objective: int
    schedule: Schedule


def _canonical_pairings(n: int) -> NDArray[np.int32]:
    """Circle-method 1-factorization of K_n with a canonical home-away pattern.

    Returns an array ``E`` of shape ``(n - 1, n // 2, 2)``; ``E[r, k]`` is the
    ``k``-th game of round ``r`` as a 1-based ``(home, away)`` pair.
    """
    m = n - 1
    pairs = np.zeros((m, n // 2, 2), dtype=np.int32)
    for i in range(1, n):
        pairs[i - 1, 0] = (i, n) if i % 2 == 0 else (n, i)
        for j in range(1, n // 2):
            a = (i - j) % m or m
            b = (i + j) % m or m
            pairs[i - 1, j] = (a, b) if j % 2 else (b, a)
    return pairs


def canonical_schedule(n: int) -> Schedule:
    """Return the canonical double round robin schedule for ``n`` teams.

    Raises:
        ValueError: If ``n`` is odd or smaller than 4.
    """
    if n < 4 or n % 2:
        raise ValueError(f"the number of teams must be even and at least 4, got {n}")
    pairs = _canonical_pairings(n)
    first_half = np.zeros((n, n - 1), dtype=np.int32)
    for r in range(n - 1):
        for home, away in pairs[r]:
            first_half[home - 1, r] = away
            first_half[away - 1, r] = -home
    return np.hstack((first_half, -first_half))


def format_schedule(schedule: Schedule, team_names: Sequence[str] | None = None) -> str:
    """Render a schedule as a table with one column per team."""
    schedule = np.asarray(schedule)
    n, rounds = schedule.shape
    names = list(team_names) if team_names is not None else [f"T{i + 1}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"expected {n} team names, got {len(names)}")
    width = max(len(name) for name in names) + 1
    lines = ["Slot  " + " ".join(f"{name:>{width}}" for name in names)]
    for r in range(rounds):
        cells = []
        for t in range(n):
            entry = int(schedule[t, r])
            label = f"@{names[abs(entry) - 1]}" if entry < 0 else names[entry - 1]
            cells.append(f"{label:>{width}}")
        lines.append(f"{r:>4}  " + " ".join(cells))
    return "\n".join(lines)


def print_schedule(schedule: Schedule, team_names: Sequence[str] | None = None) -> None:
    """Print a schedule, see :func:`format_schedule`."""
    print(format_schedule(schedule, team_names))
