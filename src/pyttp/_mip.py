"""MIP neighborhood search (phase II) built on python-mip with the HiGHS solver."""

from __future__ import annotations

import itertools
import logging

import numpy as np
from mip import MINIMIZE, Model, OptimizationStatus, xsum
from numpy.typing import NDArray

from .schedule import Schedule, Solution

logger = logging.getLogger("pyttp")


class TTPMip:
    """The TTP as a MIP, with the option to fix parts of a known schedule.

    Variable ``x[i, j, r]`` is 1 if team ``i`` plays at the home venue of team
    ``j`` in round ``r``, i.e. ``i`` is away and ``j`` is home.
    """

    def __init__(self, distances: NDArray[np.int32], max_gap: float, max_k: int) -> None:
        self.distances = distances
        self.n = distances.shape[0]
        self.rounds = 2 * self.n - 2
        self.max_k = max_k
        self.teams = range(self.n)
        self._build_model()
        self.model.verbose = 0
        self.model.max_mip_gap = max_gap

    def improve(self, objective: int, schedule: Schedule) -> Solution | None:
        """Alternate between optimizing the pairings and the home-away pattern.

        Returns the improved schedule, or ``None`` if the start could not be improved.
        """
        best_objective, best_schedule = objective, schedule
        improved = True
        while improved:
            improved = False
            for label, fix in (
                ("pairings", self._fix_pairings),
                ("home-away", self._fix_pattern),
            ):
                constraints = fix(best_schedule)
                status = self.model.optimize()
                try:
                    if status not in (
                        OptimizationStatus.OPTIMAL,
                        OptimizationStatus.FEASIBLE,
                    ):
                        return self._result(objective, best_objective, best_schedule)
                    value = round(self.model.objective_value)
                    candidate = self._extract_schedule()
                finally:
                    self.model.remove(constraints)
                if value < best_objective:
                    best_objective, best_schedule, improved = value, candidate, True
                    logger.info("phase II (%s): %d", label, value)
        return self._result(objective, best_objective, best_schedule)

    def feasible_start(self) -> Solution:
        """Solve the model without fixing anything to obtain any feasible schedule."""
        status = self.model.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            raise ValueError(f"no feasible schedule exists for max_k={self.max_k}")
        return Solution(round(self.model.objective_value), self._extract_schedule())

    @staticmethod
    def _result(start: int, best: int, schedule: Schedule) -> Solution | None:
        return Solution(best, schedule) if best < start else None

    def _extract_schedule(self) -> Schedule:
        schedule = np.zeros((self.n, self.rounds), dtype=np.int32)
        for i, j in itertools.permutations(self.teams, 2):
            for r in range(self.rounds):
                if self.x[i, j, r].x > 0.5:
                    schedule[i, r] = -(j + 1)
                    schedule[j, r] = i + 1
        return schedule

    def _visits(self, schedule: Schedule) -> NDArray[np.bool_]:
        """``visits[i, j, r]`` is True if ``i`` plays away at ``j`` in round ``r``."""
        visits = np.zeros((self.n, self.n, self.rounds), dtype=bool)
        for i, r in itertools.product(self.teams, range(self.rounds)):
            if schedule[i, r] < 0:
                visits[i, -schedule[i, r] - 1, r] = True
        return visits

    def _build_model(self) -> None:
        rounds, k = self.rounds, self.max_k
        teams = self.teams
        d = self.distances
        model = Model("ttp", sense=MINIMIZE, solver_name="HiGHS")
        self.model = model

        x = {
            (i, j, r): model.add_var(var_type="B", name=f"x[{i},{j},{r}]")
            for i, j in itertools.product(teams, teams)
            for r in range(rounds)
        }
        # y[t, i, j] = 1 if team t travels directly from i to j
        y = {
            (t, i, j): model.add_var(var_type="B", name=f"y[{t},{i},{j}]")
            for t, i, j in itertools.product(teams, teams, teams)
        }
        # z[t, i, r] = 1 if team t is at the venue of team i in round r
        z = {
            (t, i, r): model.add_var(var_type="B", name=f"z[{t},{i},{r}]")
            for t, i in itertools.product(teams, teams)
            for r in range(rounds)
        }
        self.x = x

        # Home -> first venue, venue-to-venue trips, last venue -> home.
        model.objective = (
            xsum(int(d[i, j]) * x[i, j, 0] for i, j in itertools.product(teams, teams))
            + xsum(int(d[i, j]) * y[t, i, j] for t, i, j in itertools.product(teams, teams, teams))
            + xsum(int(d[j, i]) * x[i, j, rounds - 1] for i, j in itertools.product(teams, teams))
        )

        for i in teams:
            for r in range(rounds):
                model += x[i, i, r] == 0

        # every team plays exactly once per round
        for r in range(rounds):
            for i in teams:
                model += xsum(x[i, j, r] + x[j, i, r] for j in teams) == 1

        # every team visits every other team exactly once
        for i, j in itertools.permutations(teams, 2):
            model += xsum(x[i, j, r] for r in range(rounds)) == 1

        # at most k consecutive away games / home games
        for r in range(rounds - k):
            for i in teams:
                model += xsum(x[i, j, r + s] for s in range(k + 1) for j in teams) <= k
                model += xsum(x[j, i, r + s] for s in range(k + 1) for j in teams) <= k

        # no repeaters
        for i, j in itertools.product(teams, teams):
            for r in range(rounds - 1):
                model += x[i, j, r] + x[j, i, r] + x[i, j, r + 1] + x[j, i, r + 1] <= 1

        # location of each team: at home if it is not away
        for i in teams:
            for r in range(rounds):
                model += z[i, i, r] == xsum(x[j, i, r] for j in teams)
        for i, j in itertools.permutations(teams, 2):
            for r in range(rounds):
                model += z[i, j, r] == x[i, j, r]

        for t, i, j in itertools.product(teams, teams, teams):
            for r in range(rounds - 1):
                model += y[t, i, j] >= z[t, i, r] + z[t, j, r + 1] - 1

    def _fix_pairings(self, schedule: Schedule) -> list:
        """Fix who plays whom in each round; the venues stay free."""
        visits = self._visits(schedule)
        constraints = []
        for i, j in itertools.permutations(self.teams, 2):
            for r in range(self.rounds):
                played = int(visits[i, j, r] or visits[j, i, r])
                constraints.append(
                    self.model.add_constr(self.x[i, j, r] + self.x[j, i, r] == played)
                )
        return constraints

    def _fix_pattern(self, schedule: Schedule) -> list:
        """Fix which teams are home in each round; the pairings stay free."""
        visits = self._visits(schedule)
        constraints = []
        for j in self.teams:
            others = [i for i in self.teams if i != j]
            for r in range(self.rounds):
                hosting = int(visits[:, j, r].any())
                constraints.append(
                    self.model.add_constr(xsum(self.x[i, j, r] for i in others) == hosting)
                )
                constraints.append(
                    self.model.add_constr(xsum(self.x[j, i, r] for i in others) == 1 - hosting)
                )
        return constraints
