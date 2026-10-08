"""Compare benchmark results of two builds and render a markdown table.

Usage:
    python benchmarks/compare.py --base base1.json [base2.json ...] \
        --head head1.json [head2.json ...] [--threshold 10] [--fail-above 25]

Several files per side are merged by taking the minimum per benchmark, which
filters out noise from other processes on the machine.
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def load(paths: list[str]) -> dict[str, float]:
    best: dict[str, float] = {}
    for path in paths:
        with open(path) as f:
            for name, result in json.load(f)["cases"].items():
                best[name] = min(best.get(name, math.inf), result["min_us"])
    return best


def fmt_time(us: float) -> str:
    if us >= 1000:
        return f"{us / 1000:.2f} ms"
    return f"{us:.2f} µs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", nargs="+", required=True)
    parser.add_argument("--head", nargs="+", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="percent change that counts as faster/slower (default 10)",
    )
    parser.add_argument(
        "--fail-above",
        type=float,
        default=None,
        help="exit with status 1 if any benchmark is slower by more than this percent",
    )
    parser.add_argument(
        "--annotations",
        action="store_true",
        help="emit GitHub ::warning:: annotations for regressions",
    )
    args = parser.parse_args()

    base, head = load(args.base), load(args.head)
    names = [n for n in head if n in base]
    rows, ratios, worst = [], [], 0.0
    for name in names:
        ratio = head[name] / base[name]
        change = (ratio - 1) * 100
        ratios.append(ratio)
        worst = max(worst, change)
        if change > args.threshold:
            mark = "🔴 slower"
            if args.annotations:
                print(f"::warning title=Benchmark regression::{name} is {change:+.1f}% slower")
        elif change < -args.threshold:
            mark = "🟢 faster"
        else:
            mark = "⚪ unchanged"
        cells = [f"`{name}`", fmt_time(base[name]), fmt_time(head[name]), f"{change:+.1f}%", mark]
        rows.append("| " + " | ".join(cells) + " |")

    lines = [
        "### Core benchmark: base vs. head",
        "",
        "| Benchmark | base | head | change | |",
        "|---|---:|---:|---:|---|",
        *rows,
    ]
    if ratios:
        geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        lines += [
            "",
            f"**Geometric mean:** {(geomean - 1) * 100:+.1f}% (negative is faster). "
            f"Threshold for the markers: ±{args.threshold:g}%.",
        ]
    only_head = sorted(set(head) - set(base))
    if only_head:
        lines += [
            "",
            "New benchmarks without a baseline: " + ", ".join(f"`{n}`" for n in only_head),
        ]
    print("\n".join(lines))

    if args.fail_above is not None and worst > args.fail_above:
        print(
            f"\nFAIL: worst regression {worst:+.1f}% exceeds {args.fail_above:g}%", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
