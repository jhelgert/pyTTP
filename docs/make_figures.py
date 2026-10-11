"""Regenerate the figures of the README section "How the LNS works".

The pictures are drawn with the package's own neighborhood code, so they show what the
implementation really does. matplotlib is not a dependency of pyttp; run

    uv run --with matplotlib python docs/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

import pyttp
from pyttp import _lns

OUT = Path(__file__).parent / "images"
matplotlib.rcParams["svg.fonttype"] = "path"  # text as paths: identical on every viewer
matplotlib.rcParams["font.family"] = "DejaVu Sans"

FIXED_FILL, FIXED_TEXT = "#f1f3f5", "#868e96"
WINDOW_FILL, SUBSET_FILL = "#ffd8a8", "#a5d8ff"
EDGE = "#ced4da"


def cell_label(entry: int) -> str:
    """Opponent in Trick's notation: the number of the opponent, '@' for an away game."""
    return f"@{-entry}" if entry < 0 else f"{entry}"


def draw_schedule(ax, schedule, free, fill, title):
    n, rounds = schedule.shape
    for t in range(n):
        for r in range(rounds):
            is_free = bool(free[t, r])
            ax.add_patch(
                Rectangle(
                    (r, n - 1 - t),
                    1,
                    1,
                    facecolor=fill if is_free else FIXED_FILL,
                    edgecolor="white",
                    linewidth=1.5,
                )
            )
            ax.text(
                r + 0.5,
                n - 1 - t + 0.5,
                cell_label(int(schedule[t, r])),
                ha="center",
                va="center",
                fontsize=8.5,
                color="black" if is_free else FIXED_TEXT,
                fontweight="bold" if is_free else "normal",
            )
    ax.set_xlim(0, rounds)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(rounds) + 0.5, [str(r + 1) for r in range(rounds)], fontsize=8)
    ax.set_yticks(np.arange(n) + 0.5, [f"team {t + 1}" for t in reversed(range(n))], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_xlabel("round", fontsize=9)
    ax.xaxis.set_label_position("top")
    ax.set_title(title, fontsize=10.5, pad=24, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")


def figure_neighborhoods() -> None:
    n = 8
    schedule = pyttp.canonical_schedule(n)
    rounds = schedule.shape[1]
    window = _lns.window_mask(n, rounds, 5, 4)  # rounds 6..9 (1-based)
    subset_teams = [1, 4, 6]  # teams 2, 5, 7 (1-based)
    subset = _lns.subset_mask(schedule, subset_teams)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 9.6))
    draw_schedule(
        axes[0],
        schedule,
        window,
        WINDOW_FILL,
        "Window neighborhood: all games in w = 4 consecutive rounds are free",
    )
    draw_schedule(
        axes[1],
        schedule,
        subset,
        SUBSET_FILL,
        "Subset neighborhood: the games among s = 3 teams (2, 5, 7) are free",
    )
    fig.text(
        0.5,
        0.012,
        "Entry = opponent, '@' = away game. Grey cells stay fixed; coloured cells are re-planned "
        "by one small MIP.\nCanonical schedule of n = 8 teams (R = 14 rounds).",
        ha="center",
        fontsize=8.5,
        color="#495057",
    )
    fig.subplots_adjust(top=0.9, bottom=0.07, hspace=0.45, left=0.09, right=0.98)
    fig.savefig(OUT / "lns_neighborhoods.svg")
    plt.close(fig)


def figure_legs() -> None:
    """The trip of one team: only the legs that touch a free cell appear in the objective."""
    n = 8
    schedule = pyttp.canonical_schedule(n)
    rounds = schedule.shape[1]
    team = 0
    first, width = 5, 4  # free rounds 6..9 (1-based)
    free_rounds = set(range(first, first + width))

    venues = ["home"]  # round 0 of the trip: the start at home
    for r in range(rounds):
        entry = int(schedule[team, r])
        venues.append("home" if entry > 0 else f"{-entry}")
    venues.append("home")  # back home after the last round
    labels = [f"{v}" if v != "home" else "H" for v in venues]
    positions = list(range(len(venues)))

    fig, ax = plt.subplots(figsize=(11, 3.7))
    for i, label in enumerate(labels):
        round_index = i - 1  # position 0 = start, 1..R = rounds, R+1 = end
        is_free = round_index in free_rounds
        ax.add_patch(
            Rectangle(
                (i, 0),
                0.82,
                0.82,
                facecolor=WINDOW_FILL if is_free else FIXED_FILL,
                edgecolor=EDGE,
            )
        )
        ax.text(
            i + 0.41,
            0.41,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold" if is_free else "normal",
            color="black" if is_free else FIXED_TEXT,
        )
        if 1 <= i <= rounds:
            ax.text(i + 0.41, -0.32, str(i), ha="center", fontsize=8, color="#495057")
    ax.text(-0.15, -0.32, "round", ha="right", fontsize=8, color="#495057")

    for i in range(len(positions) - 1):
        left_free = (i - 1) in free_rounds
        right_free = i in free_rounds
        touches = left_free or right_free
        arrow = FancyArrowPatch(
            (i + 0.41, 0.86),
            (i + 1.41, 0.86),
            connectionstyle="arc3,rad=-0.55",
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=2.0 if touches else 1.0,
            color="#e8590c" if touches else "#adb5bd",
        )
        ax.add_patch(arrow)

    # The five legs next to the free rounds span the boxes first .. first + width + 1.
    centre = first + 1 + (width + 1) / 2 + 0.41
    ax.annotate(
        "legs touching a free cell:\ncost $D_{uv}\\,y$, decided by the MIP",
        xy=(centre, 1.08),
        xytext=(centre, 1.85),
        ha="center",
        fontsize=9.5,
        color="#e8590c",
        arrowprops={"arrowstyle": "-", "color": "#e8590c", "linewidth": 0.8},
    )
    ax.text(
        1.0,
        1.55,
        "all other legs: constants,\nnot part of the objective",
        ha="left",
        fontsize=9.5,
        color="#868e96",
    )
    ax.set_xlim(-1.3, len(positions) + 0.2)
    ax.set_ylim(-0.7, 2.5)
    ax.axis("off")
    ax.set_title(
        "Trip of team 1: H = home, a number = visited venue. Free rounds 6 to 9 are highlighted",
        fontsize=10.5,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "lns_travel_legs.svg")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure_neighborhoods()
    figure_legs()
    print("written to", OUT)
