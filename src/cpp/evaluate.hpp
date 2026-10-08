#pragma once

#include <cstdint>

#include "schedule.hpp"

namespace pyttp {

/// Total travel distance of all teams. Every team starts and ends at home.
[[nodiscard]] std::int64_t objective(ConstScheduleView schedule, ConstDistanceView distances);

/// True if no team has more than `max_k` consecutive home games or more than
/// `max_k` consecutive away games.
[[nodiscard]] bool satisfies_stand_limits(ConstScheduleView schedule, int max_k) noexcept;

/// True if some team plays the same opponent in two consecutive rounds ("repeater"), which the
/// TTP forbids. Requires valid entries (see `has_valid_entries`).
[[nodiscard]] bool has_repeaters(ConstScheduleView schedule) noexcept;

/// True if the schedule has the shape (n, 2n - 2) and every entry is a non-zero
/// team number in [-n, n]. This is the precondition for all other functions to
/// access memory safely; it says nothing about the schedule being a tournament.
[[nodiscard]] bool has_valid_entries(ConstScheduleView schedule) noexcept;

/// True if the schedule is a valid double round robin tournament: every round
/// is a perfect matching with consistent venues, and every ordered pair of
/// teams meets exactly once.
[[nodiscard]] bool is_valid_schedule(ConstScheduleView schedule);

}  // namespace pyttp
