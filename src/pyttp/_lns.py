"""Large neighborhood search (phase II): re-optimize small parts of the schedule exactly.

Each step frees a few cells of a feasible schedule, rebuilds a small MIP for just the games in
those cells and solves it. The model only contains the free games, so its size does not depend on
the number of teams; everything else is a constant. This is what makes 30+ teams tractable, where
the global model of ``_mip.py`` has millions of rows.

Two kinds of neighborhoods are used:

* ``window``: all games in ``w`` consecutive rounds are re-placed within those rounds.
* ``subset``: the games among ``s`` teams are re-placed within the slots they already occupy.

Their sizes adapt: a neighborhood that is solved to optimality quickly grows, one that hits the
time limit shrinks.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from mip import MINIMIZE, Model, OptimizationStatus, xsum
from numpy.typing import NDArray

from . import _core
from ._deadline import Deadline
from .schedule import Schedule, Solution, is_feasible

logger = logging.getLogger("pyttp")

Mask = NDArray[np.bool_]
_SOLVED = (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE)


def window_mask(teams: int, rounds: int, start: int, width: int) -> Mask:
    """Free all cells in the rounds ``start .. start + width - 1``."""
    free = np.zeros((teams, rounds), dtype=bool)
    free[:, start : start + width] = True
    return free


def subset_mask(schedule: Schedule, subset: list[int]) -> Mask:
    """Free the cells of the games among the teams in ``subset``."""
    inside = np.zeros(schedule.shape[0], dtype=bool)
    inside[subset] = True
    opponents = np.abs(schedule) - 1
    return inside[:, None] & inside[opponents]


class FreeCellModel:
    """MIP that optimally re-places the games in the free cells of a feasible schedule.

    ``x[(away, home), r] = 1`` puts the game "``away`` plays at ``home``" into round ``r``. A game
    may go to every round in which both of its teams have a free cell. Stand limits, repeaters and
    the travel legs next to the fixed cells are taken into account. The objective only counts the
    travel legs that touch a free cell, so objective differences equal true differences.
    """

    def __init__(
        self, schedule: Schedule, distances: NDArray[np.int32], max_k: int, free: Mask
    ) -> None:
        self.schedule = schedule
        self.distances = distances
        self.max_k = max_k
        self.free = free
        self.teams, self.rounds = schedule.shape
        opponent = np.abs(schedule) - 1
        if not np.array_equal(free, free[opponent, np.arange(self.rounds)]):
            raise ValueError("both teams of a game must have a free cell in the same rounds")
        self.model = Model(sense=MINIMIZE, solver_name="HiGHS")
        self.model.verbose = 0
        self.model.max_mip_gap = 0.0
        self.status: OptimizationStatus | None = None
        self._build()

    # -- public ----------------------------------------------------------------------------

    @property
    def num_games(self) -> int:
        return len(self._games)

    @property
    def num_binaries(self) -> int:
        return len(self._x)

    def solve(self, time_limit: float) -> OptimizationStatus:
        self.model.max_seconds = time_limit
        self.status = self.model.optimize()
        return self.status

    @property
    def objective_value(self) -> float:
        return float(self.model.objective_value)

    def new_schedule(self) -> Schedule:
        """The schedule of the current solution (after a successful `solve`)."""
        result = self.schedule.copy()
        result[self.free] = 0
        for (away, home), r in (key for key, var in self._x.items() if var.x > 0.5):
            result[away, r] = -(home + 1)
            result[home, r] = away + 1
        return result

    def fix_to_start(self) -> None:
        """Pin every game to its round in the start schedule (used to validate the model)."""
        for (game, r), var in self._x.items():
            var.lb = var.ub = 1.0 if self._start_round[game] == r else 0.0

    # -- model -----------------------------------------------------------------------------

    def _build(self) -> None:
        s, free, n, rounds, k = self.schedule, self.free, self.teams, self.rounds, self.max_k
        m = self.model
        # Every game appears once, as the away team's negative entry in its round.
        placed = [
            ((t, int(-s[t, r]) - 1), r)
            for r in range(rounds)
            for t in range(n)
            if free[t, r] and s[t, r] < 0
        ]
        games = [g for g, _ in placed]
        self._games = games
        self._start_round = dict(placed)
        allowed = {g: [r for r in range(rounds) if free[g[0], r] and free[g[1], r]] for g in games}
        x = {(g, r): m.add_var(var_type="B") for g in games for r in allowed[g]}
        self._x = x
        homes: dict[int, list[tuple[int, int]]] = {t: [] for t in range(n)}
        aways: dict[int, list[tuple[int, int]]] = {t: [] for t in range(n)}
        for g in games:
            aways[g[0]].append(g)
            homes[g[1]].append(g)

        def home_vars(t: int, r: int) -> list:
            return [x[g, r] for g in homes[t] if (g, r) in x]

        def away_vars(t: int, r: int) -> list:
            return [x[g, r] for g in aways[t] if (g, r) in x]

        for g in games:
            m += xsum(x[g, r] for r in allowed[g]) == 1
        for t, r in itertools.product(range(n), range(rounds)):
            if free[t, r]:
                m += xsum(home_vars(t, r) + away_vars(t, r)) == 1

        # At most k consecutive home games and k consecutive away games. Fixed cells in the
        # window count as constants.
        for t in range(n):
            for first in range(rounds - k):
                cells = range(first, first + k + 1)
                if not any(free[t, r] for r in cells):
                    continue
                for variables, at_home in ((home_vars, True), (away_vars, False)):
                    fixed = sum(1 for r in cells if not free[t, r] and (s[t, r] > 0) == at_home)
                    terms = [v for r in cells if free[t, r] for v in variables(t, r)]
                    if terms:
                        m += xsum(terms) <= k - fixed

        # No repeaters: the same two teams never meet in consecutive rounds.
        def meets(t: int, o: int, r: int) -> tuple[int, list]:
            """(constant, variables) of "t plays o in round r"."""
            if not free[t, r]:
                return int(abs(int(s[t, r])) - 1 == o), []
            return 0, [
                x[g, r]
                for g in aways[t] + homes[t]
                if (g, r) in x and (g[1] if g[0] == t else g[0]) == o
            ]

        def opponents(t: int, r: int) -> set[int]:
            if not free[t, r]:
                return {abs(int(s[t, r])) - 1}
            return {(g[1] if g[0] == t else g[0]) for g in aways[t] + homes[t] if (g, r) in x}

        for t in range(n):
            for r in range(rounds - 1):
                if not (free[t, r] or free[t, r + 1]):
                    continue
                for o in opponents(t, r) & opponents(t, r + 1):
                    c1, v1 = meets(t, o, r)
                    c2, v2 = meets(t, o, r + 1)
                    if v1 or v2:
                        m += xsum(v1 + v2) <= 1 - c1 - c2

        # Travel legs next to the free cells. A leg between two free cells needs an
        # indicator y >= (at u) + (at v) - 1; its value is pushed down by the objective, so
        # it can be continuous.
        def venues(t: int, r: int) -> list[tuple[int, list | None]]:
            """(venue, variables) alternatives of team t in round r; None marks a constant."""
            if r < 0 or r >= rounds:
                return [(t, None)]
            if not free[t, r]:
                return [(t if s[t, r] > 0 else int(-s[t, r]) - 1, None)]
            options: list[tuple[int, list | None]] = []
            if home_vars(t, r):
                options.append((t, home_vars(t, r)))
            options += [(g[1], [x[g, r]]) for g in aways[t] if (g, r) in x]
            return options

        d = self.distances
        terms = []
        for t in range(n):
            for r in range(-1, rounds):
                if not ((r >= 0 and free[t, r]) or (r + 1 < rounds and free[t, r + 1])):
                    continue
                for u, vu in venues(t, r):
                    for v, vv in venues(t, r + 1):
                        cost = int(d[u, v])
                        if cost == 0 or (vu is None and vv is None):
                            continue
                        if vu is None:
                            terms.append(cost * xsum(vv or []))
                        elif vv is None:
                            terms.append(cost * xsum(vu))
                        else:
                            y = m.add_var(lb=0.0, ub=1.0)
                            m += y >= xsum(vu) + xsum(vv) - 1
                            terms.append(cost * y)
        m.objective = xsum(terms)
        # Warm start with the current placement.
        m.start = [(var, 1.0 if self._start_round[g] == r else 0.0) for (g, r), var in x.items()]


# -- adaptive neighborhood sizes ---------------------------------------------------------------


@dataclass
class _Neighborhood:
    """One kind of neighborhood together with its current size.

    The size follows the cost of a solve, as a fraction of the per-solve time limit:

    * it shrinks when the solve hit the limit or took more than ``shrink_above``;
    * it grows when ``stall_grow`` solves in a row at this size found nothing: a neighborhood
      that is too small to contain an improvement is useless however cheap it is;
    * optionally (``grow_below`` > 0) it also grows after a very cheap proof of optimality.
      This is off by default: growing as soon as solves are cheap made the neighborhoods big
      and slow, and clearly gave worse schedules in the same time (NFL16, NFL32 and CIRC20,
      every run). Small cheap neighborhoods give the most improvement per second.
    """

    name: str
    size: int
    low: int
    high: int  # the size at which the neighborhood is the whole problem
    grow_below: float = 0.0
    shrink_above: float = 0.25
    stall_grow: int = 10
    fruitless: int = 0

    def adapt(
        self, solved_to_optimality: bool, seconds: float, limit: float, improved: bool
    ) -> None:
        self.fruitless = 0 if improved else self.fruitless + 1
        if not solved_to_optimality or seconds > self.shrink_above * limit:
            self.size = max(self.low, self.size - 1)
            self.fruitless = 0
        elif seconds < self.grow_below * limit or self.fruitless >= self.stall_grow:
            self.size = min(self.high, self.size + 1)
            self.fruitless = 0


def make_neighborhoods(teams: int, rounds: int) -> list[_Neighborhood]:
    return [
        _Neighborhood("window", size=min(3, rounds), low=2, high=rounds),
        _Neighborhood("subset", size=min(4, teams), low=3, high=teams),
    ]


def improve(
    best: Solution,
    distances: NDArray[np.int32],
    max_k: int,
    deadline: Deadline,
    rng: np.random.Generator,
    polish: Callable[[Solution, Deadline], Solution],
    patience: int = 100,
    sub_time_limit: float = 2.0,
    polish_every: int = 25,
) -> Solution:
    """Improve a feasible solution by LNS until the deadline or ``patience`` fruitless steps.

    ``best`` must satisfy all rules of the TTP. The result always does, too. When a
    neighborhood that covers the whole problem is solved to optimality, the solution is
    optimal and the search stops at once.
    """
    n, rounds = best.schedule.shape
    neighborhoods = make_neighborhoods(n, rounds)
    stall = 0
    step = 0
    while stall < patience and not deadline.expired:
        kind = neighborhoods[step % len(neighborhoods)]
        step += 1
        if kind.name == "window":
            start = int(rng.integers(0, rounds - kind.size + 1))
            free = window_mask(n, rounds, start, kind.size)
        else:
            free = subset_mask(
                best.schedule, [int(t) for t in rng.choice(n, kind.size, replace=False)]
            )
        model = FreeCellModel(best.schedule, distances, max_k, free)
        if model.num_games == 0:
            stall += 1
            continue

        limit = min(sub_time_limit, deadline.remaining())
        began = time.monotonic()
        status = model.solve(limit)
        seconds = time.monotonic() - began
        optimal = status == OptimizationStatus.OPTIMAL
        improved = False
        if status in _SOLVED:
            candidate = model.new_schedule()
            value = _core.objective(candidate, distances)
            if value < best.objective:
                if is_feasible(candidate, max_k):
                    best = Solution(value, candidate)
                    improved = True
                    logger.info("phase II (lns %s %d): %d", kind.name, kind.size, value)
                else:  # cannot happen if the model is right; never return an infeasible schedule
                    logger.warning("lns neighborhood returned an infeasible schedule, ignored")
        if optimal and model.num_games == n * (n - 1):
            # Every game was free and the model was solved to optimality: nothing is better.
            logger.info("phase II (lns): proved optimality of %d", best.objective)
            return best
        kind.adapt(optimal, seconds, limit, improved)
        stall = 0 if improved else stall + 1

        if step % polish_every == 0:
            polished = polish(best, deadline)
            if polished.objective < best.objective:
                best = polished
                stall = 0
    return best
