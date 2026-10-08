#pragma once

#include <cstddef>

#include "schedule.hpp"

namespace pyttp {

// All moves modify `schedule` in place and use 0-based team and round indices.
// Every move maps a valid double round robin schedule to another valid one.
// Moves never check the home/away stand limits; that is up to the caller.

/// Swap the venues of the two games between teams `i` and `j`.
void swap_homes(ScheduleView schedule, std::size_t i, std::size_t j);

/// Swap the two rounds `r1` and `r2`.
void swap_rounds(ScheduleView schedule, std::size_t r1, std::size_t r2);

/// Swap the schedules of teams `i` and `j` in all rounds except the ones in
/// which they play each other.
void swap_teams(ScheduleView schedule, std::size_t i, std::size_t j);

/// Swap the games of team `team` in rounds `r1` and `r2`. The swap ripples to
/// every other team whose games are affected, so the result stays valid.
void partial_swap_rounds(ScheduleView schedule, std::size_t team, std::size_t r1, std::size_t r2);

/// Swap the games of teams `i` and `j` in round `round`. The swap ripples to
/// all further rounds that are needed to keep the schedule valid. Does nothing
/// if `i` and `j` play each other in `round`.
void partial_swap_teams(ScheduleView schedule, std::size_t i, std::size_t j, std::size_t round);

}  // namespace pyttp
