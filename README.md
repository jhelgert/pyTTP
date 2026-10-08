# pyttp

🏈 A hybrid local search and MIP heuristic for the **t**raveling **t**ournament **p**roblem.

Given `n` teams (`n` even) and a distance matrix `D`, where `D[i, j]` is the distance from team
`i` to team `j`, find a
[double round robin tournament](https://en.wikipedia.org/wiki/Round-robin_tournament) such that

- no team plays more than `max_k` consecutive home games or more than `max_k` consecutive away games,
- no two teams play each other in two consecutive rounds, and
- the total distance traveled by all teams is minimal. Every team starts and ends at home.

The algorithm is a hybrid. Starting from a canonical schedule, a greedy local search in C++
(phase I) improves the solution. Phase II then improves the result further and can be chosen with
`phase2`:

- **`"lns"` (default): large neighborhood search.** It repeatedly frees a few rounds or the games
  among a few teams and re-optimizes just those games exactly with a small MIP. The model size
  does not depend on the number of teams, so it works up to 30+ teams. Neighborhood sizes adapt
  to the solve times; the local search polishes the result between rounds.
- **`"mip"`: global MIP.** The approach of
  [M. Goerigk and S. Westphal](https://link.springer.com/article/10.1007%2Fs10479-014-1586-6):
  alternately optimize the pairings and the home-away pattern with one MIP over the whole
  problem. It only fits small instances, the model has millions of rows for 32 teams.

The local search is implemented in C++23 and exposed with [nanobind](https://github.com/wjakob/nanobind).
The MIPs are solved by [HiGHS](https://highs.dev) through [python-mip](https://github.com/coin-or/python-mip),
so no commercial solver is needed.

### Solution quality

Measured with `benchmarks/bench_quality.py` on the classic benchmark instances (see
`benchmarks/instances`), `max_k=3`, seed 0, one run each. The gap is relative to the best known
upper bound of the [RobinX repository](https://www.sportscheduling.ugent.be/RobinX/travelRepo.php).

| Instance | Teams | Time limit | Objective | Best known | Gap |
|---|---:|---:|---:|---:|---:|
| NL4 | 4 | none | 8,276 | 8,276 | 0% (proven optimal) |
| NL12 | 12 | 30 s | 133,682 | 110,729 | 20.7% |
| NL16 | 16 | 30 s | 346,768 | 261,687 | 32.5% |
| NFL16 | 16 | 60 s | 297,529 | 231,483 | 28.5% |
| NFL24 | 24 | 60 s | 570,044 | 463,657 | 22.9% |
| NFL32 | 32 | 60 s | 1,281,078 | 914,620 | 40.1% |

This is a practical heuristic, not a state-of-the-art solver: the best known values come from
specialised algorithms that run for hours. On small instances the search stalls in a local optimum
(it only accepts improvements) and further time does not help. For comparison, the global MIP
mode (`phase2="mip"`) ends 46.2% above the best known on NFL16 in 60 s (LNS: 28.5%), and its model
is too large to be useful for 32 teams.

## Install

```bash
pip install pyttp
```

Wheels are published for CPython 3.11 to 3.14 on Linux (glibc, x86_64 and aarch64), macOS
(x86_64 and arm64) and Windows (x86_64).

## Example

```python
import logging

import numpy as np
import pyttp

logging.basicConfig(level=logging.INFO)  # progress is reported through the "pyttp" logger

team_names = ["ATL", "NYM", "PHI", "MON"]
distances = np.array(
    [[0, 745, 665, 929], [745, 0, 80, 337], [665, 80, 0, 380], [929, 337, 380, 0]],
    dtype=np.int32,
)

solution = pyttp.solve(distances, max_k=3, time_limit=60)  # time_limit is optional
print(solution.objective)  # 8276
pyttp.print_schedule(solution.schedule, team_names)
```

```none
Slot   ATL  NYM  PHI  MON
   0  @MON  PHI @NYM  ATL
   1  @NYM  ATL @MON  PHI
   2  @PHI @MON  ATL  NYM
   3   MON @PHI  NYM @ATL
   4   NYM @ATL  MON @PHI
   5   PHI  MON @ATL @NYM
```

A schedule is an `int32` array of shape `(n, 2n - 2)`. Entry `[t, r]` describes the game of team
`t` in round `r` (both 0-based): `abs(entry) - 1` is the opponent, a positive sign is a home game
and a negative sign an away game.

### API

| | |
|---|---|
| `pyttp.solve(distances, max_k=3, max_phase1_runs=10, max_mip_gap=0.05, time_limit=None, phase2="lns", seed=None, lns_patience=100)` | Run the hybrid heuristic, returns a `Solution(objective, schedule)`. `time_limit` (seconds): return the best schedule found when the budget is used up. `phase2`: `"lns"` or `"mip"`. `seed`: random neighborhood choice of `"lns"`. `lns_patience`: without a time limit, stop after this many fruitless neighborhoods. `max_mip_gap` only affects `"mip"`. |
| `pyttp.objective(schedule, distances)` | Total travel distance of a schedule. |
| `pyttp.satisfies_stand_limits(schedule, max_k)` | Checks the home stand and road trip limit. |
| `pyttp.has_repeaters(schedule)` | True if some pair of teams plays in two consecutive rounds (forbidden). |
| `pyttp.is_feasible(schedule, max_k)` | All rules at once: valid tournament, stand limits, no repeaters. |
| `pyttp.is_valid_schedule(schedule)` | Checks that a schedule is a valid double round robin. |
| `pyttp.canonical_schedule(n)` | The canonical start schedule. |
| `pyttp.neighborhoods` | `swap_homes`, `swap_rounds`, `swap_teams`, `partial_swap_rounds`, `partial_swap_teams`. |

## Development

```bash
uv sync                  # builds the extension in editable mode (needs a C++23 compiler)
uv run pytest
uv run ruff check . && uv run ruff format .
uv run pyrefly check
clang-format -i src/cpp/*.cpp src/cpp/*.hpp
uv build                 # sdist + wheel in dist/
```

### Benchmarks

`benchmarks/bench_core.py` times the native core (objective, every neighborhood move, local search
steps and descents; no MIP) on fixed random instances:

```bash
uv run python benchmarks/bench_core.py --output result.json
```

The `Benchmark` workflow builds the base and the head of every pull request on the same runner,
runs the script alternately against both and posts a comparison table (`benchmarks/compare.py`) to
the job summary. Slowdowns above 10% show up as warnings on the PR. Run-to-run noise is around
3-4%. Pass `--fail-above PERCENT` to `compare.py` to turn regressions into failures.

Local builds are compiled with `-march=native` (`/arch:AVX2` on MSVC). Release wheels turn this
off (`-DPYTTP_NATIVE=OFF`, see `[tool.cibuildwheel]` in `pyproject.toml`), because they have to run
on every CPU. To opt out locally, build with `CMAKE_ARGS="-DPYTTP_NATIVE=OFF" uv build`.

The version is derived from git tags (`git tag v0.1.0`). Pushing a `v*` tag builds all wheels with
cibuildwheel and uploads them to TestPyPI via Trusted Publishing.
