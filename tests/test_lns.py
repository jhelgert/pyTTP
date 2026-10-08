"""Large neighborhood search (phase II)."""

from __future__ import annotations

import itertools
import logging
import time

import numpy as np
import pytest
from mip import OptimizationStatus

import pyttp
from helpers import NL4_DISTANCES, random_distances
from pyttp import _core, _lns, solver
from pyttp._deadline import Deadline


def feasible_start(rng: np.random.Generator, n: int):
    """A feasible, non-trivial schedule: the result of phase I on a random instance."""
    d = random_distances(rng, n, symmetric=True)
    canonical = pyttp.canonical_schedule(n)
    start = pyttp.Solution(pyttp.objective(canonical, d), canonical)
    pool = solver._local_search(start, d, 3, 10)
    return d, (pool[0] if pool else start)


# -- masks ---------------------------------------------------------------------------------------


def test_window_mask_frees_whole_rounds():
    free = _lns.window_mask(6, 10, 3, 4)
    assert free.shape == (6, 10)
    assert free[:, 3:7].all()
    assert not free[:, :3].any()
    assert not free[:, 7:].any()


def test_subset_mask_frees_the_games_among_the_subset_only():
    schedule = pyttp.canonical_schedule(8)
    subset = [1, 4, 5]
    free = _lns.subset_mask(schedule, subset)
    for t in range(8):
        for r in range(schedule.shape[1]):
            expected = t in subset and abs(int(schedule[t, r])) - 1 in subset
            assert free[t, r] == expected
    assert free.sum() == len(subset) * (len(subset) - 1) * 2  # each pair meets twice, both rows


def test_masks_are_always_symmetric_between_the_two_teams_of_a_game(rng):
    d, start = feasible_start(rng, 8)
    for _ in range(10):
        free = _lns.subset_mask(start.schedule, list(rng.choice(8, 4, replace=False)))
        _lns.FreeCellModel(start.schedule, d, 3, free)  # raises if not symmetric


def test_model_rejects_a_mask_that_splits_a_game():
    schedule = pyttp.canonical_schedule(6)
    free = np.zeros(schedule.shape, dtype=bool)
    free[0, 0] = True  # team 0 free, its opponent in that round is not
    with pytest.raises(ValueError, match="both teams"):
        _lns.FreeCellModel(schedule, np.ones((6, 6), dtype=np.int32), 3, free)


# -- the free-cell model -------------------------------------------------------------------------


@pytest.mark.parametrize("n", [6, 8])
def test_model_objective_is_exact_and_results_are_feasible(rng, n):
    """The predicted improvement equals the real change of the objective, for random masks."""
    d = random_distances(rng, n, symmetric=True)
    schedule = pyttp.canonical_schedule(n)  # a poor start: plenty to improve
    start = pyttp.Solution(pyttp.objective(schedule, d), schedule)
    rounds = 2 * n - 2
    improving = 0
    for trial in range(8):
        if trial % 2 == 0:
            width = int(rng.integers(2, 4))
            free = _lns.window_mask(n, rounds, int(rng.integers(0, rounds - width + 1)), width)
        else:
            size = int(rng.integers(3, 5))
            subset = [int(t) for t in rng.choice(n, size, replace=False)]
            free = _lns.subset_mask(start.schedule, subset)
        model = _lns.FreeCellModel(start.schedule, d, 3, free)
        assert model.solve(5) in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE)
        candidate = model.new_schedule()
        optimum = model.objective_value
        model.fix_to_start()
        assert model.solve(5) == OptimizationStatus.OPTIMAL
        predicted = model.objective_value - optimum  # model(start) - model(optimum)

        real = start.objective - pyttp.objective(candidate, d)
        assert real == pytest.approx(predicted, abs=1e-4)
        assert real >= 0  # warm start: never worse than the start
        assert pyttp.is_feasible(candidate, 3)
        assert (candidate[~free] == start.schedule[~free]).all(), "fixed cells must not change"
        improving += real > 0
    assert improving > 0, "some neighborhood must improve the canonical schedule"


def test_model_with_everything_free_is_the_whole_problem():
    n = 4
    schedule = pyttp.canonical_schedule(n)
    model = _lns.FreeCellModel(schedule, NL4_DISTANCES, 3, np.ones(schedule.shape, dtype=bool))
    assert model.num_games == n * (n - 1)
    assert model.solve(60) == OptimizationStatus.OPTIMAL
    assert pyttp.objective(model.new_schedule(), NL4_DISTANCES) == 8276  # proven optimum of NL4


def test_model_respects_the_stand_limit_inside_windows(rng):
    d = random_distances(rng, 8, symmetric=True)
    for max_k in (2, 3):
        base = pyttp.canonical_schedule(8)
        if not pyttp.satisfies_stand_limits(base, max_k):
            continue
        free = _lns.window_mask(8, 14, 4, 4)
        model = _lns.FreeCellModel(base, d, max_k, free)
        model.solve(30)
        assert pyttp.is_feasible(model.new_schedule(), max_k)


# -- adaptive neighborhood sizes -----------------------------------------------------------------


def test_neighborhood_grows_after_a_quick_proof_and_shrinks_after_a_timeout():
    kind = _lns._Neighborhood("window", size=4, low=2, high=6)
    kind.adapt(solved_to_optimality=True, seconds=0.01, limit=2.0)
    assert kind.size == 5
    kind.adapt(solved_to_optimality=True, seconds=1.5, limit=2.0)  # optimal, but slow: stay
    assert kind.size == 5
    kind.adapt(solved_to_optimality=False, seconds=2.0, limit=2.0)
    assert kind.size == 4


def test_neighborhood_size_is_clamped():
    kind = _lns._Neighborhood("subset", size=5, low=3, high=6)
    for _ in range(5):
        kind.adapt(True, 0.0, 2.0)
    assert kind.size == 6
    for _ in range(10):
        kind.adapt(False, 2.0, 2.0)
    assert kind.size == 3


def test_initial_neighborhoods_fit_tiny_instances():
    window, subset = _lns.make_neighborhoods(teams=4, rounds=6)
    assert (window.size, window.high) == (3, 6)
    assert (subset.size, subset.high) == (4, 4)


# -- the driver ----------------------------------------------------------------------------------


def identity_polish(solution, deadline):
    return solution


def run_improve(start, d, **kwargs):
    defaults = {
        "max_k": 3,
        "deadline": Deadline(None),
        "rng": np.random.default_rng(0),
        "polish": identity_polish,
        "patience": 5,
    }
    defaults.update(kwargs)
    return _lns.improve(start, d, **defaults)


def test_improve_returns_a_feasible_solution_that_is_not_worse(rng):
    d, start = feasible_start(rng, 8)
    result = run_improve(start, d, deadline=Deadline(2.0), patience=20)
    assert result.objective <= start.objective
    assert pyttp.is_feasible(result.schedule, 3)
    assert pyttp.objective(result.schedule, d) == result.objective


def test_improve_stops_after_the_patience_is_used_up(monkeypatch, rng):
    d, start = feasible_start(rng, 6)
    calls = []

    def never_solves(self, time_limit):
        calls.append(time_limit)
        return OptimizationStatus.INFEASIBLE

    monkeypatch.setattr(_lns.FreeCellModel, "solve", never_solves)
    assert run_improve(start, d, patience=4) is start
    assert len(calls) == 4


def test_improve_with_an_expired_deadline_does_nothing(rng):
    d, start = feasible_start(rng, 8)
    assert run_improve(start, d, deadline=Deadline(0.0)) is start


def test_improve_never_returns_an_infeasible_schedule(monkeypatch, rng, caplog):
    d, start = feasible_start(rng, 6)

    def broken(self):
        bad = self.schedule.copy()
        bad[0, 0] = -bad[0, 0]  # inconsistent venues
        return bad

    monkeypatch.setattr(_lns.FreeCellModel, "new_schedule", broken)
    # Pretend that every candidate improves, so only the feasibility guard can stop it.
    monkeypatch.setattr(_core, "objective", lambda *a: 0)
    monkeypatch.setattr(_lns, "_core", _FakeCore())
    with caplog.at_level(logging.WARNING, logger="pyttp"):
        result = run_improve(start, d, patience=3)
    assert result is start
    assert any("infeasible" in record.message for record in caplog.records)


class _FakeCore:
    @staticmethod
    def objective(*args):
        return 0


def test_improve_proves_optimality_when_every_game_is_free():
    start_schedule = pyttp.canonical_schedule(4)
    start = pyttp.Solution(pyttp.objective(start_schedule, NL4_DISTANCES), start_schedule)
    began = time.monotonic()
    result = run_improve(start, NL4_DISTANCES, patience=10_000)  # only the proof can stop it
    assert result.objective == 8276
    assert time.monotonic() - began < 60


def test_polish_is_called_and_its_improvement_is_kept(monkeypatch, rng):
    d, start = feasible_start(rng, 6)
    better = pyttp.Solution(start.objective - 1, start.schedule)
    seen = []

    def polish(solution, deadline):
        seen.append(solution.objective)
        return better

    monkeypatch.setattr(_lns.FreeCellModel, "solve", lambda self, t: OptimizationStatus.INFEASIBLE)
    result = run_improve(start, d, polish=polish, polish_every=1, patience=3)
    assert seen, "polish must be called"
    assert result.objective == better.objective


# -- solve() -------------------------------------------------------------------------------------


def test_solve_with_lns_finds_the_nl4_optimum_and_stops_by_itself():
    began = time.monotonic()
    solution = pyttp.solve(NL4_DISTANCES, max_k=3, phase2="lns", seed=0)
    assert solution.objective == 8276
    assert pyttp.is_feasible(solution.schedule, 3)
    assert time.monotonic() - began < 60


@pytest.mark.parametrize("n", [8, 12, 16])
def test_solve_with_lns_improves_on_the_start_within_the_time_limit(rng, n):
    d = random_distances(rng, n, symmetric=True)
    started = time.monotonic()
    solution = pyttp.solve(d, max_k=3, time_limit=2.0, seed=1)
    assert time.monotonic() - started < 8.0
    assert pyttp.is_feasible(solution.schedule, 3)
    assert pyttp.objective(solution.schedule, d) == solution.objective
    assert solution.objective < pyttp.objective(pyttp.canonical_schedule(n), d)


def test_solve_with_lns_is_reproducible_for_a_fixed_seed(rng):
    d = random_distances(rng, 6, symmetric=True)
    first = pyttp.solve(d, max_k=3, seed=7, lns_patience=1)
    second = pyttp.solve(d, max_k=3, seed=7, lns_patience=1)
    assert first.objective == second.objective
    np.testing.assert_array_equal(first.schedule, second.schedule)


def test_solve_with_lns_handles_a_tight_stand_limit(rng):
    d = random_distances(rng, 8, symmetric=True)
    solution = pyttp.solve(d, max_k=2, time_limit=3.0, seed=0)
    assert pyttp.is_feasible(solution.schedule, 2)


def test_solve_rejects_unknown_modes_and_bad_patience():
    with pytest.raises(ValueError, match="phase2"):
        pyttp.solve(NL4_DISTANCES, phase2="magic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lns_patience"):
        pyttp.solve(NL4_DISTANCES, lns_patience=0)


def test_solve_with_the_mip_mode_still_works():
    solution = pyttp.solve(NL4_DISTANCES, max_k=3, phase2="mip", max_mip_gap=0.0)
    assert solution.objective == 8276
    assert pyttp.is_feasible(solution.schedule, 3)


def test_the_mip_mode_does_not_build_the_lns_models(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("LNS must not run in mip mode")

    monkeypatch.setattr(_lns, "improve", forbidden)
    pyttp.solve(NL4_DISTANCES, max_k=3, phase2="mip", time_limit=20)


# -- the no-repeater rule inside the model -------------------------------------------------------


def best_reordering(schedule, d, first, width, max_k, *, require_feasible):
    """Brute force: cheapest permutation of the rounds of a window (no other changes)."""
    best = None
    for order in itertools.permutations(range(width)):
        candidate = np.concatenate(
            [
                schedule[:, :first],
                schedule[:, first : first + width][:, list(order)],
                schedule[:, first + width :],
            ],
            axis=1,
        )
        ok = (
            pyttp.is_feasible(candidate, max_k)
            if require_feasible
            else (
                pyttp.is_valid_schedule(candidate)
                and pyttp.satisfies_stand_limits(candidate, max_k)
            )
        )
        if ok:
            value = pyttp.objective(candidate, d)
            best = value if best is None else min(best, value)
    return best


@pytest.mark.parametrize("seed", [10, 11, 14])
def test_repeater_rule_binds_inside_a_window_and_the_model_respects_it(seed):
    """The rule can only bind in a window of at least five rounds, because a game and its mirror
    are five rounds apart in the canonical schedule of six teams. Find distances for which ignoring
    the rule would be cheaper, then require the model to stay on the feasible side."""
    n, width = 6, 5
    schedule = pyttp.canonical_schedule(n)
    rng = np.random.default_rng(seed)
    for _ in range(400):
        d = rng.integers(1, 60, (n, n)).astype(np.int32)
        d = ((d + d.T) // 2).astype(np.int32)
        np.fill_diagonal(d, 0)
        first = int(rng.integers(1, 2 * n - 2 - width))
        ignoring = best_reordering(schedule, d, first, width, 3, require_feasible=False)
        respecting = best_reordering(schedule, d, first, width, 3, require_feasible=True)
        if ignoring is not None and respecting is not None and ignoring < respecting:
            break
    else:
        pytest.fail("no distance matrix found for which the repeater rule binds")

    model = _lns.FreeCellModel(schedule, d, 3, _lns.window_mask(n, 2 * n - 2, first, width))
    assert model.solve(30) == OptimizationStatus.OPTIMAL
    candidate = model.new_schedule()
    assert not pyttp.has_repeaters(candidate)
    assert pyttp.is_feasible(candidate, 3)
    # Re-placing games may do better than reordering rounds, but never better than the free
    # optimum that respects the rule, and it must not exploit the forbidden repeater.
    assert pyttp.objective(candidate, d) <= respecting
    assert pyttp.objective(candidate, d) >= 0.9 * ignoring  # sanity: not wildly different


def test_model_never_creates_repeaters_when_everything_is_free():
    n = 4
    d = np.full((n, n), 100, dtype=np.int32)
    d[0, 1] = d[1, 0] = 1  # teams 0 and 1 are close: tempting to schedule them back to back
    np.fill_diagonal(d, 0)
    schedule = pyttp.canonical_schedule(n)
    model = _lns.FreeCellModel(schedule, d, 3, np.ones(schedule.shape, dtype=bool))
    assert model.solve(20) == OptimizationStatus.OPTIMAL
    assert pyttp.is_feasible(model.new_schedule(), 3)
