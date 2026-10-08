from __future__ import annotations

import logging

import numpy as np
import pytest

import pyttp
from helpers import NL4_DISTANCES, random_distances


def check_solution(solution, d, max_k):
    assert pyttp.is_feasible(solution.schedule, max_k)
    assert pyttp.objective(solution.schedule, d) == solution.objective


@pytest.mark.slow
@pytest.mark.parametrize("phase2", ["lns", "mip"])
def test_solve_nl4_finds_the_optimum(caplog, phase2):
    with caplog.at_level(logging.INFO, logger="pyttp"):
        solution = pyttp.solve(NL4_DISTANCES, max_k=3, max_mip_gap=0.0, phase2=phase2)
    check_solution(solution, NL4_DISTANCES, 3)
    assert solution.objective == 8276
    assert any("finished" in record.message for record in caplog.records)


@pytest.mark.slow
def test_solve_never_does_worse_than_the_canonical_schedule(rng):
    d = random_distances(rng, 6, symmetric=True)
    start = pyttp.objective(pyttp.canonical_schedule(6), d)
    solution = pyttp.solve(d, max_k=3, max_phase1_runs=3, time_limit=3.0, seed=0)
    check_solution(solution, d, 3)
    assert solution.objective <= start


@pytest.mark.slow
def test_solve_accepts_nested_lists():
    solution = pyttp.solve(NL4_DISTANCES.tolist(), max_k=3)
    check_solution(solution, NL4_DISTANCES, 3)


def test_solve_does_not_print(capsys):
    pyttp.solve(NL4_DISTANCES, max_k=3)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "bad", [np.zeros((4, 5)), np.zeros(4), np.zeros((2, 2, 2))], ids=["rect", "1d", "3d"]
)
def test_solve_rejects_bad_matrices(bad):
    with pytest.raises(ValueError, match="square"):
        pyttp.solve(bad)


def test_solve_rejects_odd_team_counts():
    with pytest.raises(ValueError, match="even"):
        pyttp.solve(np.ones((5, 5)))


def test_print_schedule_formats_a_table(capsys):
    pyttp.print_schedule(pyttp.canonical_schedule(4), ["ATL", "NYM", "PHI", "MON"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["Slot", "ATL", "NYM", "PHI", "MON"]
    assert len(lines) == 1 + 6
    with pytest.raises(ValueError, match="team names"):
        pyttp.format_schedule(pyttp.canonical_schedule(4), ["a", "b"])
