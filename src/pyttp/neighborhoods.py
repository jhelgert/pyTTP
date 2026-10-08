"""Neighborhood moves on schedules.

All functions return a new schedule and leave their input untouched. Team and
round indices are 0-based. Every move maps a valid schedule to a valid
schedule; they do not check the home-away stand limits.
"""

from ._core import (
    partial_swap_rounds,
    partial_swap_teams,
    swap_homes,
    swap_rounds,
    swap_teams,
)

__all__ = [
    "partial_swap_rounds",
    "partial_swap_teams",
    "swap_homes",
    "swap_rounds",
    "swap_teams",
]
