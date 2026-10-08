from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

import pyttp
from helpers import NL4_SCHEDULE, random_schedule
from pyttp import neighborhoods as nb

REFERENCE = np.load(Path(__file__).parent / "data" / "cython_reference_n6.npz")
MOVES = {
    "swap_homes": nb.swap_homes,
    "swap_rounds": nb.swap_rounds,
    "swap_teams": nb.swap_teams,
    "partial_swap_rounds": nb.partial_swap_rounds,
}


@pytest.mark.parametrize("move", MOVES)
def test_matches_original_cython_implementation(move):
    schedule = REFERENCE["schedule"]
    args = REFERENCE[f"args_{move}"]
    expected = REFERENCE[f"out_{move}"]
    assert len(args) > 0
    for arguments, want in zip(args, expected, strict=True):
        got = MOVES[move](schedule, *map(int, arguments))
        np.testing.assert_array_equal(got, want)


def all_moves(schedule: np.ndarray):
    n, rounds = schedule.shape
    for i, j in itertools.combinations(range(n), 2):
        yield "swap_homes", nb.swap_homes(schedule, i, j)
        yield "swap_teams", nb.swap_teams(schedule, i, j)
        for r in range(rounds):
            yield "partial_swap_teams", nb.partial_swap_teams(schedule, i, j, r)
    for r1, r2 in itertools.combinations(range(rounds), 2):
        yield "swap_rounds", nb.swap_rounds(schedule, r1, r2)
        for t in range(n):
            yield "partial_swap_rounds", nb.partial_swap_rounds(schedule, t, r1, r2)


@pytest.mark.parametrize("n", [4, 6, 8])
def test_every_move_keeps_the_schedule_valid(rng, n):
    for schedule in [pyttp.canonical_schedule(n), *(random_schedule(rng, n) for _ in range(3))]:
        for name, result in all_moves(schedule):
            assert pyttp.is_valid_schedule(result), name


def test_moves_do_not_modify_their_input():
    schedule = NL4_SCHEDULE.copy()
    for _, _ in all_moves(schedule):
        pass
    np.testing.assert_array_equal(schedule, NL4_SCHEDULE)


def test_swap_homes_flips_both_games_of_the_pair():
    result = nb.swap_homes(pyttp.canonical_schedule(4), 0, 1)
    original = pyttp.canonical_schedule(4)
    changed = np.argwhere(result != original)
    assert len(changed) == 4  # two games, two rows each
    assert (result[changed[:, 0], changed[:, 1]] == -original[changed[:, 0], changed[:, 1]]).all()


def test_swap_rounds_is_an_involution(rng):
    schedule = random_schedule(rng, 6)
    twice = nb.swap_rounds(nb.swap_rounds(schedule, 1, 4), 1, 4)
    np.testing.assert_array_equal(twice, schedule)


def test_partial_swap_teams_depends_on_the_round(rng):
    # The original Cython version ignored the round argument entirely.
    schedule = random_schedule(rng, 8)
    outputs = {nb.partial_swap_teams(schedule, 0, 1, r).tobytes() for r in range(14)}
    assert len(outputs) > 1


def test_partial_swap_teams_is_a_no_op_when_the_pair_meets():
    schedule = pyttp.canonical_schedule(6)
    meeting = next(r for r in range(10) if abs(schedule[0, r]) == 2)
    np.testing.assert_array_equal(nb.partial_swap_teams(schedule, 0, 1, meeting), schedule)


def test_index_errors():
    schedule = pyttp.canonical_schedule(4)
    with pytest.raises(IndexError):
        nb.swap_homes(schedule, 0, 4)
    with pytest.raises(IndexError):
        nb.swap_rounds(schedule, 0, 6)
    with pytest.raises(IndexError):
        nb.partial_swap_teams(schedule, 0, 1, 6)
    with pytest.raises(TypeError):
        nb.swap_teams(schedule, -1, 1)


def test_rejects_malformed_schedule_shape():
    with pytest.raises(ValueError, match="shape"):
        nb.swap_rounds(np.zeros((4, 4), dtype=np.int32), 0, 1)
