"""Micro-benchmarks for the native core (no MIP).

Usage:
    uv run python benchmarks/bench_core.py [--output result.json] [--repeat 7]

The script only uses APIs that exist in both old and new versions of the core,
so it can be run against any build of ``pyttp`` to compare commits.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

import pyttp
from pyttp import _core

SEED = 20240601
MAX_K = 3


@dataclass(frozen=True)
class Case:
    name: str
    run: Callable[[], object]
    number: int  # calls per timed repetition; results are reported per call


def make_instance(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + n)
    d = rng.integers(1, 1000, (n, n)).astype(np.int32)
    d = ((d + d.T) // 2).astype(np.int32)
    np.fill_diagonal(d, 0)
    return pyttp.canonical_schedule(n), d


def descend(schedule: np.ndarray, d: np.ndarray, max_steps: int = 50) -> int:
    """Best-improvement local search until a local optimum, returns the number of steps."""
    for step in range(max_steps):
        found = _core.local_search_step(schedule, d, MAX_K)
        if not found:
            return step
        schedule = min(found, key=lambda c: c[1])[2]
    return max_steps


def build_cases() -> list[Case]:
    cases: list[Case] = []

    s16, d16 = make_instance(16)
    cases += [
        Case("objective/n16", lambda: _core.objective(s16, d16), 20_000),
        Case(
            "satisfies_stand_limits/n16", lambda: _core.satisfies_stand_limits(s16, MAX_K), 20_000
        ),
        Case("is_valid_schedule/n16", lambda: _core.is_valid_schedule(s16), 20_000),
        Case("move/swap_homes/n16", lambda: _core.swap_homes(s16, 2, 9), 20_000),
        Case("move/swap_rounds/n16", lambda: _core.swap_rounds(s16, 3, 20), 20_000),
        Case("move/swap_teams/n16", lambda: _core.swap_teams(s16, 2, 9), 20_000),
        Case(
            "move/partial_swap_rounds/n16", lambda: _core.partial_swap_rounds(s16, 4, 3, 20), 20_000
        ),
        Case("move/partial_swap_teams/n16", lambda: _core.partial_swap_teams(s16, 2, 9, 5), 20_000),
    ]

    for n, number in ((8, 200), (12, 20), (16, 5), (20, 2)):
        s, d = make_instance(n)
        cases.append(
            Case(
                f"local_search_step/n{n}",
                lambda s=s, d=d: _core.local_search_step(s, d, MAX_K),
                number,
            )
        )

    for n in (10, 14):
        s, d = make_instance(n)
        cases.append(Case(f"local_search_descent/n{n}", lambda s=s, d=d: descend(s, d), 1))

    return cases


def measure(case: Case, repeat: int) -> dict[str, float]:
    case.run()  # warm-up
    per_call = []
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(case.number):
            case.run()
        per_call.append((time.perf_counter() - start) / case.number)
    return {
        "min_us": min(per_call) * 1e6,
        "median_us": statistics.median(per_call) * 1e6,
        "number": case.number,
        "repeat": repeat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="write results as JSON to this file")
    parser.add_argument("--repeat", type=int, default=7, help="timed repetitions per case")
    parser.add_argument("--filter", default="", help="only run cases whose name contains this")
    args = parser.parse_args()

    results = {}
    for case in build_cases():
        if args.filter not in case.name:
            continue
        results[case.name] = measure(case, args.repeat)
        r = results[case.name]
        print(f"{case.name:<36} min {r['min_us']:>12.2f} us   median {r['median_us']:>12.2f} us")

    if args.output:
        meta = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "pyttp": pyttp.__version__,
        }
        with open(args.output, "w") as f:
            json.dump({"meta": meta, "cases": results}, f, indent=2)


if __name__ == "__main__":
    main()
