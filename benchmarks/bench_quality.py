"""Solution quality on the classic TTP benchmark instances (see benchmarks/instances).

Usage:
    uv run python benchmarks/bench_quality.py nl8 nfl16 nfl32 --time-limit 60
    uv run python benchmarks/bench_quality.py --family nfl --time-limit 120 --output quality.json

Each run is limited by `--time-limit` seconds (wall clock) and reports the gap to the best
known upper bound of the RobinX repository. These runs take as long as the limit, so they are
not part of the regular CI.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import pyttp

INSTANCES = Path(__file__).parent / "instances"


def load(name: str) -> np.ndarray:
    rows = [
        list(map(int, line.split()))
        for line in (INSTANCES / f"{name}.txt").read_text().splitlines()
        if line.strip()
    ]
    return np.array(rows, dtype=np.int32)


def run(name: str, reference: dict, time_limit: float, phase2: str, seed: int) -> dict:
    d = load(name)
    started = time.monotonic()
    solution = pyttp.solve(d, max_k=3, time_limit=time_limit, phase2=phase2, seed=seed)  # type: ignore[arg-type]
    elapsed = time.monotonic() - started
    assert pyttp.is_feasible(solution.schedule, 3), f"{name}: infeasible schedule"
    best_known, lower = reference["best_known"], reference["lower_bound"]
    return {
        "instance": name,
        "teams": reference["teams"],
        "phase2": phase2,
        "seed": seed,
        "seconds": round(elapsed, 1),
        "objective": solution.objective,
        "best_known": best_known,
        "gap_to_best_known_percent": round(100 * (solution.objective - best_known) / best_known, 2),
        "gap_to_lower_bound_percent": round(100 * (solution.objective - lower) / lower, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", nargs="*", help="instance names, e.g. nl8 nfl32")
    parser.add_argument("--family", choices=["nl", "circ", "nfl"], help="run a whole family")
    parser.add_argument("--time-limit", type=float, default=60.0, help="seconds per instance")
    parser.add_argument("--phase2", choices=["lns", "mip"], default="lns")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="write the results as JSON")
    args = parser.parse_args()

    references = json.loads((INSTANCES / "reference.json").read_text())
    names = list(args.instances)
    if args.family:
        names += sorted(
            (k for k in references if k.startswith(args.family)),
            key=lambda k: references[k]["teams"],
        )
    if not names:
        parser.error("name at least one instance or use --family")
    unknown = [n for n in names if n not in references]
    if unknown:
        parser.error(f"unknown instances {unknown}; available: {sorted(references)}")

    results = []
    print(f"{'instance':<9}{'n':>3}{'objective':>12}{'best known':>12}{'gap':>9}{'time':>8}")
    for name in names:
        r = run(name, references[name], args.time_limit, args.phase2, args.seed)
        results.append(r)
        print(
            f"{name:<9}{r['teams']:>3}{r['objective']:>12}{r['best_known']:>12}"
            f"{r['gap_to_best_known_percent']:>8.1f}%{r['seconds']:>7.0f}s",
            flush=True,
        )
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
