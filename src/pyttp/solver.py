"""Hybrid local search and MIP heuristic for the traveling tournament problem."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from ._mip import TTPMip
from .schedule import Schedule, Solution, canonical_schedule

__all__ = ["Solution", "solve"]

logger = logging.getLogger("pyttp")

_POOL_SIZE = 3


def _local_search(
    start: Solution, distances: NDArray[np.int32], max_k: int, max_runs: int
) -> list[Solution]:
    """Phase I: best-improvement descent, remembering the best schedules seen.

    Returns up to ``_POOL_SIZE`` schedules that are strictly better than
    ``start``, best first. The list is empty if there is none.
    """
    seen: dict[int, Schedule] = {}
    current = start
    for _ in range(max_runs):
        candidates = _core.local_search_step(current.schedule, distances, max_k)
        if not candidates:
            break
        for _, objective, schedule in candidates:
            seen.setdefault(objective, schedule)
        best_objective = min(objective for _, objective, _ in candidates)
        current = Solution(best_objective, seen[best_objective])
    pool = sorted(seen.items())[:_POOL_SIZE]
    return [Solution(objective, schedule) for objective, schedule in pool]


def solve(
    distances: ArrayLike,
    max_k: int = 3,
    max_phase1_runs: int = 10,
    max_mip_gap: float = 0.05,
) -> Solution:
    """Solve the TTP with a hybrid local search and MIP heuristic.

    Args:
        distances: Square distance matrix; ``distances[i, j]`` is the distance
            from team ``i`` to team ``j``. The number of teams must be even.
        max_k: Maximum number of consecutive home games and of consecutive away games.
        max_phase1_runs: Maximum number of local search steps per phase I.
        max_mip_gap: Relative MIP gap used by the solver in phase II.

    Returns:
        The best schedule found.
    """
    d = np.ascontiguousarray(distances, dtype=np.int32)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distances must be a square matrix")
    if max_k < 1:
        raise ValueError("max_k must be at least 1")

    mip = TTPMip(d, max_mip_gap, max_k)
    schedule = canonical_schedule(d.shape[0])
    if _core.satisfies_stand_limits(schedule, max_k):
        best = Solution(_core.objective(schedule, d), schedule)
    else:
        logger.info("canonical schedule violates max_k=%d, searching a feasible start", max_k)
        best = mip.feasible_start()
    logger.info("start solution: %d", best.objective)

    while True:
        pool = _local_search(best, d, max_k, max_phase1_runs)
        if pool:
            best = pool[0]
            logger.info("phase I: %d", best.objective)
        improved = False
        for start in pool or [best]:
            result = mip.improve(start.objective, start.schedule)
            if result is not None:
                best = result
                improved = True
                break
            logger.info("phase II could not improve %d", start.objective)
        if not pool and not improved:
            logger.info("finished: %d", best.objective)
            return best
