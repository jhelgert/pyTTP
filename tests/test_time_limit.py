"""The wall-clock budget of `solve` and the robustness of the MIP phase."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from mip import OptimizationStatus

import pyttp
from helpers import random_distances
from pyttp import _mip, solver
from pyttp._deadline import Deadline


def test_deadline_without_limit_never_expires():
    deadline = Deadline(None)
    assert not deadline.expired
    assert deadline.remaining() == math.inf


def test_deadline_expires_and_never_goes_negative():
    deadline = Deadline(0.05)
    assert not deadline.expired
    assert 0 < deadline.remaining() <= 0.05 + 1e-6  # tolerate float rounding
    time.sleep(0.08)
    assert deadline.expired
    assert deadline.remaining() == 0.0
    assert Deadline(0.0).expired


@pytest.mark.parametrize("limit", [-1.0, float("nan")])
def test_invalid_time_limits_are_rejected(limit):
    with pytest.raises(ValueError, match="time_limit"):
        pyttp.solve(np.ones((4, 4)), time_limit=limit)


def test_zero_time_limit_returns_the_canonical_solution_without_building_the_mip(rng, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the MIP model must not be built when there is no time left")

    monkeypatch.setattr(solver, "TTPMip", forbidden)
    d = random_distances(rng, 12, symmetric=True)
    solution = pyttp.solve(d, max_k=3, time_limit=0.0)
    canonical = pyttp.canonical_schedule(12)
    assert solution.objective == pyttp.objective(canonical, d)
    np.testing.assert_array_equal(solution.schedule, canonical)


@pytest.mark.slow
@pytest.mark.parametrize("phase2", ["lns", "mip"])
def test_time_limit_is_respected(rng, phase2):
    # With the global MIP this instance takes about a minute without a limit.
    d = random_distances(rng, 8, symmetric=True)
    start = time.monotonic()
    solution = pyttp.solve(d, max_k=3, time_limit=3.0, phase2=phase2)
    elapsed = time.monotonic() - start
    assert elapsed < 12.0
    assert pyttp.is_feasible(solution.schedule, 3)
    assert pyttp.objective(solution.schedule, d) == solution.objective
    assert solution.objective <= pyttp.objective(pyttp.canonical_schedule(8), d)


@pytest.mark.slow
@pytest.mark.parametrize("phase2", ["lns", "mip"])
def test_a_generous_limit_gives_the_same_result_as_no_limit(rng, phase2):
    d = random_distances(rng, 6, symmetric=True)
    options = {"max_k": 3, "phase2": phase2, "seed": 0, "lns_patience": 4}
    unlimited = pyttp.solve(d, **options)
    limited = pyttp.solve(d, time_limit=600.0, **options)
    assert limited.objective == unlimited.objective
    np.testing.assert_array_equal(limited.schedule, unlimited.schedule)


def test_expired_deadline_stops_the_local_search(rng):
    d = random_distances(rng, 8)
    schedule = pyttp.canonical_schedule(8)
    start = pyttp.Solution(pyttp.objective(schedule, d), schedule)
    assert solver._local_search(start, d, 3, 10, Deadline(0.0)) == []
    assert solver._local_search(start, d, 3, 10, Deadline(None))


def test_improve_with_an_expired_deadline_does_nothing(rng):
    d = random_distances(rng, 6, symmetric=True)
    schedule = pyttp.canonical_schedule(6)
    mip = _mip.TTPMip(d, 0.05, 3)
    assert mip.improve(pyttp.objective(schedule, d), schedule, Deadline(0.0)) is None


@pytest.mark.slow
def test_improve_returns_feasible_schedules_from_a_phase_one_start(rng):
    d = random_distances(rng, 6, symmetric=True)
    canonical = pyttp.canonical_schedule(6)
    start = pyttp.Solution(pyttp.objective(canonical, d), canonical)
    pool = solver._local_search(start, d, 3, 10)
    assert pool
    mip = _mip.TTPMip(d, 0.0, 3)
    for candidate in pool:
        result = mip.improve(candidate.objective, candidate.schedule)
        if result is not None:
            assert result.objective < candidate.objective
            assert pyttp.is_feasible(result.schedule, 3)
            assert pyttp.objective(result.schedule, d) == result.objective


@pytest.mark.slow
def test_improve_still_tries_the_second_neighborhood_after_a_failure(rng, monkeypatch):
    d = random_distances(rng, 6, symmetric=True)
    canonical = pyttp.canonical_schedule(6)
    mip = _mip.TTPMip(d, 0.0, 3)
    real_optimize = mip.model.optimize
    calls: list[int] = []

    def flaky_optimize(*args, **kwargs):
        calls.append(len(calls))
        if len(calls) == 1:
            return OptimizationStatus.INFEASIBLE  # e.g. a neighborhood without a solution
        return real_optimize(*args, **kwargs)

    monkeypatch.setattr(mip.model, "optimize", flaky_optimize)
    mip.improve(pyttp.objective(canonical, d), canonical)
    assert len(calls) >= 2, "a failing neighborhood must not abort the other one"


def test_driver_never_trades_a_better_pool_entry_for_a_worse_mip_result(monkeypatch):
    """Regression: improving a weaker pool entry used to overwrite the best schedule."""
    d = np.ones((4, 4), dtype=np.int32) - np.eye(4, dtype=np.int32)
    canonical = pyttp.canonical_schedule(4)
    best_entry = pyttp.Solution(100, canonical)
    weak_entry = pyttp.Solution(110, canonical)
    weak_result = pyttp.Solution(105, canonical)
    pools = iter([[best_entry, weak_entry], []])

    class FakeMip:
        def __init__(self, *args):
            pass

        def improve(self, objective, schedule, deadline=None):
            return weak_result if objective == 110 else None

    monkeypatch.setattr(solver, "TTPMip", FakeMip)
    monkeypatch.setattr(solver, "_local_search", lambda *a, **k: next(pools))
    assert pyttp.solve(d, max_k=3, phase2="mip").objective == 100
