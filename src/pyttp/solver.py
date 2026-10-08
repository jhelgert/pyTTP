"""Hybrid local search and MIP heuristic for the traveling tournament problem."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core, _lns
from ._deadline import Deadline
from ._mip import TTPMip
from .schedule import Schedule, Solution, canonical_schedule

__all__ = ["Phase2", "Solution", "solve"]

Phase2 = Literal["lns", "mip"]
_PHASE2_MODES = ("lns", "mip")

logger = logging.getLogger("pyttp")

_POOL_SIZE = 3


def _local_search(
    start: Solution,
    distances: NDArray[np.int32],
    max_k: int,
    max_runs: int,
    deadline: Deadline | None = None,
) -> list[Solution]:
    """Phase I: best-improvement descent, remembering the best schedules seen.

    Returns up to ``_POOL_SIZE`` schedules that are strictly better than
    ``start``, best first. The list is empty if there is none. Stops early when the
    deadline has passed (checked before every step).
    """
    deadline = deadline or Deadline(None)
    seen: dict[int, Schedule] = {}
    current = start
    for _ in range(max_runs):
        if deadline.expired:
            break
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
    time_limit: float | None = None,
    phase2: Phase2 = "lns",
    seed: int | None = None,
    lns_patience: int = 100,
) -> Solution:
    """Solve the TTP with a hybrid local search and MIP heuristic.

    Args:
        distances: Square distance matrix; ``distances[i, j]`` is the distance
            from team ``i`` to team ``j``. The number of teams must be even.
        max_k: Maximum number of consecutive home games and of consecutive away games.
        max_phase1_runs: Maximum number of local search steps per phase I.
        max_mip_gap: Relative MIP gap of the global model (only used with ``phase2="mip"``).
        time_limit: Wall-clock budget in seconds, or ``None`` for no limit. When it is
            used up, the best schedule found so far is returned. It is checked before every
            local search step and handed to the MIP solver, so it can be overshot by one
            step. Building the MIP model is not interruptible (about a second for 16 teams,
            and it grows quickly with the number of teams); it is skipped entirely if the
            budget is already used up.
        phase2: How phase II improves the schedule found by the local search.

            * ``"lns"`` (default): large neighborhood search. Repeatedly re-optimizes the
              games in a few rounds or among a few teams with a small MIP whose size does not
              depend on the number of teams. Works up to 30+ teams. Without ``time_limit`` it
              stops after ``lns_patience`` neighborhoods in a row without improvement, or as
              soon as optimality is proven (small instances).
            * ``"mip"``: alternately optimizes the pairings and the home-away pattern with
              one global MIP. It only fits small instances: the model has millions of rows
              for 32 teams.
        seed: Seed of the random neighborhood choice of ``"lns"``. With a fixed seed the
            result is reproducible unless a MIP hits its time limit, which depends on the
            speed of the machine.
        lns_patience: Number of consecutive fruitless neighborhoods after which ``"lns"``
            stops when there is no ``time_limit``.

    Returns:
        The best schedule found; it satisfies all rules of the TTP.

    Raises:
        TimeoutError: If the limit expires before any feasible schedule was found. This can
            only happen if the canonical schedule violates ``max_k``.
    """
    d = np.ascontiguousarray(distances, dtype=np.int32)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distances must be a square matrix")
    if max_k < 1:
        raise ValueError("max_k must be at least 1")
    if time_limit is not None and not time_limit >= 0:
        raise ValueError("time_limit must be non-negative")
    if phase2 not in _PHASE2_MODES:
        raise ValueError(f"phase2 must be one of {_PHASE2_MODES}, got {phase2!r}")
    if lns_patience < 1:
        raise ValueError("lns_patience must be at least 1")
    deadline = Deadline(time_limit)

    mip: TTPMip | None = None

    def get_mip() -> TTPMip:
        nonlocal mip
        if mip is None:
            mip = TTPMip(d, max_mip_gap, max_k)
        return mip

    schedule = canonical_schedule(d.shape[0])
    if _core.satisfies_stand_limits(schedule, max_k):
        best = Solution(_core.objective(schedule, d), schedule)
    else:
        logger.info("canonical schedule violates max_k=%d, searching a feasible start", max_k)
        best = get_mip().feasible_start(deadline)
    logger.info("start solution: %d", best.objective)

    if phase2 == "lns":
        pool = _local_search(best, d, max_k, max_phase1_runs, deadline)
        if pool:
            best = pool[0]
            logger.info("phase I: %d", best.objective)

        def polish(current: Solution, outer: Deadline) -> Solution:
            budget = Deadline(min(5.0, outer.remaining()))
            found = _local_search(current, d, max_k, 5, budget)
            return found[0] if found else current

        best = _lns.improve(
            best,
            d,
            max_k,
            deadline,
            np.random.default_rng(seed),
            polish,
            patience=lns_patience,
        )
        logger.info("finished: %d", best.objective)
        return best

    while not deadline.expired:
        pool = _local_search(best, d, max_k, max_phase1_runs, deadline)
        if pool:
            best = pool[0]
            logger.info("phase I: %d", best.objective)
        improved = False
        for start in pool or [best]:
            if deadline.expired:
                break
            result = get_mip().improve(start.objective, start.schedule, deadline)
            if result is not None:
                # `start` can be a weaker pool entry: never trade a better schedule for it.
                if result.objective < best.objective:
                    best = result
                improved = True
                break
            logger.info("phase II could not improve %d", start.objective)
        if not pool and not improved:
            logger.info("finished: %d", best.objective)
            return best
    logger.info("time limit reached: %d", best.objective)
    return best
