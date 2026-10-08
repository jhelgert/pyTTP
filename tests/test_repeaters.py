"""The "no repeaters" rule: the two games of a pair of teams must not follow each other directly."""

from __future__ import annotations

import numpy as np
import pytest

import pyttp
from helpers import random_distances, random_schedule
from pyttp import _core, solver
from pyttp import neighborhoods as nb


def reference_has_repeaters(schedule: np.ndarray) -> bool:
    n, rounds = schedule.shape
    return any(
        abs(int(schedule[t, r])) == abs(int(schedule[t, r + 1]))
        for t in range(n)
        for r in range(rounds - 1)
    )


def schedule_with_repeater(n: int) -> np.ndarray:
    """Canonical schedule with round 1 replaced by the mirror of round 0."""
    schedule = pyttp.canonical_schedule(n)
    return nb.swap_rounds(schedule, 1, n - 1)


@pytest.mark.parametrize("n", [4, 6, 8, 12, 16, 20, 32, 64])
def test_canonical_schedule_has_no_repeaters(n):
    assert not pyttp.has_repeaters(pyttp.canonical_schedule(n))


@pytest.mark.parametrize("n", [4, 6, 10, 32])
def test_detects_a_constructed_repeater(n):
    schedule = schedule_with_repeater(n)
    assert pyttp.is_valid_schedule(schedule)
    assert reference_has_repeaters(schedule)
    assert pyttp.has_repeaters(schedule)


def test_detects_a_repeater_in_the_last_round_pair():
    schedule = pyttp.canonical_schedule(6)
    last = schedule.shape[1] - 1
    schedule = nb.swap_rounds(schedule, last - 1, 4)  # mirror of round 4 is round 9 = last
    assert reference_has_repeaters(schedule)
    assert pyttp.has_repeaters(schedule)


def test_has_repeaters_matches_reference_on_random_schedules(rng):
    seen = {True: 0, False: 0}
    for _ in range(400):
        n = int(rng.choice([4, 6, 8, 10, 16]))
        schedule = random_schedule(rng, n, steps=int(rng.integers(0, 25)))
        expected = reference_has_repeaters(schedule)
        assert pyttp.has_repeaters(schedule) == expected
        seen[expected] += 1
    assert min(seen.values()) > 20, "the test should see both repeater and repeater-free schedules"


def test_has_repeaters_rejects_malformed_input():
    with pytest.raises(ValueError, match="shape"):
        pyttp.has_repeaters(np.zeros((4, 5), dtype=np.int32))
    bad = pyttp.canonical_schedule(4)
    bad[0, 0] = 9
    with pytest.raises(ValueError, match="entries"):
        pyttp.has_repeaters(bad)


def test_is_feasible_combines_all_rules():
    schedule = pyttp.canonical_schedule(6)
    assert pyttp.is_feasible(schedule, 3)
    assert not pyttp.is_feasible(schedule_with_repeater(6), 3)  # repeater
    assert not pyttp.is_feasible(schedule, 1)  # stand limit
    broken = schedule.copy()
    broken[0, 0] = -broken[0, 0]
    assert not pyttp.is_feasible(broken, 3)  # not a tournament


@pytest.mark.parametrize("n", [6, 8, 12, 16])
@pytest.mark.parametrize("max_k", [2, 3])
def test_local_search_never_returns_repeaters(rng, n, max_k):
    """Regression: the search used to accept neighbors with repeaters (8 of them at n=8)."""
    for _ in range(3):
        d = random_distances(rng, n)
        schedule = pyttp.canonical_schedule(n)
        for _ in range(8):  # follow the search for several steps
            found = _core.local_search_step(schedule, d, max_k)
            for name, _, candidate in found:
                assert pyttp.is_valid_schedule(candidate), name
                assert not pyttp.has_repeaters(candidate), name
                assert pyttp.satisfies_stand_limits(candidate, max_k), name
            if not found:
                break
            schedule = min(found, key=lambda c: c[1])[2]


@pytest.mark.parametrize("n", [8, 12, 16])
def test_phase_one_pool_is_feasible(rng, n):
    """The schedules handed to the MIP must be feasible for it (the original bug)."""
    d = random_distances(rng, n, symmetric=True)
    start_schedule = pyttp.canonical_schedule(n)
    start = pyttp.Solution(pyttp.objective(start_schedule, d), start_schedule)
    pool = solver._local_search(start, d, max_k=3, max_runs=10)
    assert pool
    for solution in pool:
        assert pyttp.is_feasible(solution.schedule, 3)
        assert pyttp.objective(solution.schedule, d) == solution.objective


def test_local_search_may_leave_an_infeasible_start(rng):
    """An input that has repeaters is still improvable, but only into feasible neighbors."""
    n = 8
    d = random_distances(rng, n)
    start = schedule_with_repeater(n)
    found = _core.local_search_step(start, d, 3)
    for _, objective, candidate in found:
        assert not pyttp.has_repeaters(candidate)
        assert objective < pyttp.objective(start, d)
