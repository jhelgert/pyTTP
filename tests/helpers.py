from __future__ import annotations

import numpy as np

import pyttp
from pyttp import neighborhoods

NL4_DISTANCES = np.array(
    [[0, 745, 665, 929], [745, 0, 80, 337], [665, 80, 0, 380], [929, 337, 380, 0]],
    dtype=np.int32,
)
# Optimal NL4 schedule from the README (objective 8276), 0-based team indices.
NL4_SCHEDULE = np.array(
    [[-4, -2, -3, 2, 4, 3], [3, 1, -4, -1, -3, 4], [-2, -4, 1, 4, 2, -1], [1, 3, 2, -3, -1, -2]],
    dtype=np.int32,
)


def random_distances(rng: np.random.Generator, n: int, symmetric: bool = False) -> np.ndarray:
    d = rng.integers(1, 500, (n, n)).astype(np.int32)
    if symmetric:
        d = ((d + d.T) // 2).astype(np.int32)
    np.fill_diagonal(d, 0)
    return d


def random_schedule(rng: np.random.Generator, n: int, steps: int = 12) -> np.ndarray:
    """A random valid schedule, reached by a random walk of neighborhood moves."""
    schedule = pyttp.canonical_schedule(n)
    rounds = 2 * n - 2
    for _ in range(steps):
        i, j = rng.choice(n, size=2, replace=False)
        r1, r2 = rng.choice(rounds, size=2, replace=False)
        match rng.integers(5):
            case 0:
                schedule = neighborhoods.swap_homes(schedule, int(i), int(j))
            case 1:
                schedule = neighborhoods.swap_rounds(schedule, int(r1), int(r2))
            case 2:
                schedule = neighborhoods.swap_teams(schedule, int(i), int(j))
            case 3:
                schedule = neighborhoods.partial_swap_rounds(schedule, int(i), int(r1), int(r2))
            case _:
                schedule = neighborhoods.partial_swap_teams(schedule, int(i), int(j), int(r1))
    return schedule
