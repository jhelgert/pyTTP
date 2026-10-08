from __future__ import annotations

import itertools

import numpy as np
import pytest

import pyttp
from helpers import random_distances, random_schedule
from pyttp import _core
from pyttp import neighborhoods as nb


def brute_force_best(schedule, d, max_k):
    """Best feasible (stand limits, no repeaters) strictly improving neighbor per neighborhood."""
    n, rounds = schedule.shape
    current = pyttp.objective(schedule, d)
    best: dict[str, int] = {}

    def consider(name, neighbor):
        value = pyttp.objective(neighbor, d)
        if value < current and pyttp.is_feasible(neighbor, max_k):
            best[name] = min(best.get(name, value), value)

    for i, j in itertools.combinations(range(n), 2):
        consider("swap_homes", nb.swap_homes(schedule, i, j))
        consider("swap_teams", nb.swap_teams(schedule, i, j))
        for r in range(rounds):
            consider("partial_swap_teams", nb.partial_swap_teams(schedule, i, j, r))
    for r1, r2 in itertools.combinations(range(rounds), 2):
        consider("swap_rounds", nb.swap_rounds(schedule, r1, r2))
        for t in range(n):
            consider("partial_swap_rounds", nb.partial_swap_rounds(schedule, t, r1, r2))
    return best


@pytest.mark.parametrize("n", [4, 6, 8])
@pytest.mark.parametrize("max_k", [2, 3])
def test_step_matches_brute_force(rng, n, max_k):
    for _ in range(3):
        d = random_distances(rng, n)
        schedule = random_schedule(rng, n)
        found = _core.local_search_step(schedule, d, max_k)
        assert {name: objective for name, objective, _ in found} == brute_force_best(
            schedule, d, max_k
        )


def test_candidates_are_consistent(rng):
    d = random_distances(rng, 8)
    schedule = pyttp.canonical_schedule(8)
    start = pyttp.objective(schedule, d)
    found = _core.local_search_step(schedule, d, 3)
    assert found
    for _, objective, candidate in found:
        assert candidate.dtype == np.int32
        assert pyttp.is_valid_schedule(candidate)
        assert pyttp.satisfies_stand_limits(candidate, 3)
        assert pyttp.objective(candidate, d) == objective < start


def test_step_returns_nothing_at_a_local_optimum():
    d = np.ones((4, 4), dtype=np.int32) - np.eye(4, dtype=np.int32)
    schedule = pyttp.canonical_schedule(4)
    while found := _core.local_search_step(schedule, d, 3):
        schedule = min(found, key=lambda c: c[1])[2]
    assert _core.local_search_step(schedule, d, 3) == []


def test_step_does_not_modify_the_input(rng):
    d = random_distances(rng, 6)
    schedule = random_schedule(rng, 6)
    before = schedule.copy()
    _core.local_search_step(schedule, d, 3)
    np.testing.assert_array_equal(schedule, before)
