"""pyttp: a hybrid local search and MIP heuristic for the traveling tournament problem."""

from importlib.metadata import PackageNotFoundError, version

from ._core import has_repeaters, is_valid_schedule, objective, satisfies_stand_limits
from .schedule import Solution, canonical_schedule, format_schedule, is_feasible, print_schedule
from .solver import solve

try:
    __version__ = version("pyttp")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "Solution",
    "__version__",
    "canonical_schedule",
    "format_schedule",
    "has_repeaters",
    "is_feasible",
    "is_valid_schedule",
    "objective",
    "print_schedule",
    "satisfies_stand_limits",
    "solve",
]
