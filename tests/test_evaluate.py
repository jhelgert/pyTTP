from __future__ import annotations

import numpy as np
import pytest

import pyttp
from helpers import NL4_DISTANCES, NL4_SCHEDULE, random_distances, random_schedule
from pyttp import neighborhoods


def reference_objective(schedule: np.ndarray, d: np.ndarray) -> int:
    total = 0
    for team, row in enumerate(schedule):
        position = team
        for entry in row:
            target = -entry - 1 if entry < 0 else team
            total += d[position, target]
            position = target
        total += d[position, team]
    return int(total)


def reference_stand_limits(schedule: np.ndarray, max_k: int) -> bool:
    for row in schedule:
        for r in range(len(row) - max_k):
            window = row[r : r + max_k + 1]
            if (window < 0).all() or (window > 0).all():
                return False
    return True


def test_objective_nl4():
    assert pyttp.objective(NL4_SCHEDULE, NL4_DISTANCES) == 8276


def test_stand_limits_nl4():
    assert pyttp.satisfies_stand_limits(NL4_SCHEDULE, 3)
    assert not pyttp.satisfies_stand_limits(NL4_SCHEDULE, 2)


@pytest.mark.parametrize("n", [4, 6, 8, 10, 16])
def test_objective_matches_reference_for_asymmetric_distances(rng, n):
    for _ in range(10):
        d = random_distances(rng, n)
        schedule = random_schedule(rng, n)
        assert pyttp.objective(schedule, d) == reference_objective(schedule, d)


def test_stand_limits_match_reference_on_random_sign_matrices(rng):
    for _ in range(3000):
        n = int(rng.choice([4, 6, 8, 10]))
        rounds = 2 * n - 2
        entries = rng.integers(1, n + 1, (n, rounds)) * rng.choice([-1, 1], (n, rounds))
        schedule = entries.astype(np.int32)
        max_k = int(rng.integers(1, 6))
        assert pyttp.satisfies_stand_limits(schedule, max_k) == reference_stand_limits(
            schedule, max_k
        )


def test_stand_limits_regression_short_windows():
    # Three away games at the very end: the old Cython version missed this.
    schedule = pyttp.canonical_schedule(4).copy()
    schedule[0] = [1, 1, 1, -1, -1, -1]
    assert not pyttp.satisfies_stand_limits(schedule, 2)
    assert pyttp.satisfies_stand_limits(schedule, 3)


@pytest.mark.parametrize("n", [4, 6, 8, 12, 20, 30])
def test_canonical_schedule_is_valid(n):
    assert pyttp.is_valid_schedule(pyttp.canonical_schedule(n))


@pytest.mark.parametrize("n", [0, 2, 3, 5, 7])
def test_canonical_schedule_rejects_bad_sizes(n):
    with pytest.raises(ValueError, match="even"):
        pyttp.canonical_schedule(n)


def test_is_valid_schedule_detects_corruption():
    schedule = pyttp.canonical_schedule(6)
    assert pyttp.is_valid_schedule(schedule)
    broken = schedule.copy()
    broken[0, 0], broken[0, 1] = broken[0, 1], broken[0, 0]
    assert not pyttp.is_valid_schedule(broken)
    wrong_venue = schedule.copy()
    wrong_venue[0, 0] = -wrong_venue[0, 0]
    assert not pyttp.is_valid_schedule(wrong_venue)
    assert not pyttp.is_valid_schedule(schedule[:, :-1])


def test_input_validation():
    with pytest.raises(ValueError, match="shape"):
        pyttp.objective(np.zeros((4, 5), dtype=np.int32), NL4_DISTANCES)
    with pytest.raises(ValueError, match="shape"):
        pyttp.objective(NL4_SCHEDULE, np.zeros((3, 3), dtype=np.int32))


def test_non_contiguous_input_is_accepted():
    d = np.asfortranarray(NL4_DISTANCES)
    s = np.asfortranarray(NL4_SCHEDULE)
    assert pyttp.objective(s, d) == 8276


@pytest.mark.parametrize("bad_entry", [0, 5, -5, 2**31 - 1, -(2**31)])
def test_out_of_range_entries_are_rejected_instead_of_crashing(bad_entry):
    schedule = pyttp.canonical_schedule(4)
    schedule[1, 2] = bad_entry
    assert not pyttp.is_valid_schedule(schedule)
    for call in (
        lambda: pyttp.objective(schedule, NL4_DISTANCES),
        lambda: pyttp.satisfies_stand_limits(schedule, 3),
        lambda: neighborhoods.swap_teams(schedule, 0, 1),
        lambda: neighborhoods.partial_swap_rounds(schedule, 0, 0, 1),
    ):
        with pytest.raises(ValueError, match="entries"):
            call()
