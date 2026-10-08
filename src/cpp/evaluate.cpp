#include "evaluate.hpp"

#include <algorithm>
#include <vector>

namespace pyttp {

std::int64_t objective(ConstScheduleView schedule, ConstDistanceView distances) {
  std::int64_t total = 0;
  for (std::size_t team = 0; team < schedule.rows; ++team) {
    std::size_t location = team;
    for (std::size_t round = 0; round < schedule.cols; ++round) {
      const Entry entry = schedule(team, round);
      const std::size_t next = entry < 0 ? opponent(entry) : team;
      total += distances(location, next);
      location = next;
    }
    total += distances(location, team);
  }
  return total;
}

bool satisfies_stand_limits(ConstScheduleView schedule, int max_k) noexcept {
  for (std::size_t team = 0; team < schedule.rows; ++team) {
    int run = 0;
    bool previous_home = false;
    for (std::size_t round = 0; round < schedule.cols; ++round) {
      const bool home = schedule(team, round) > 0;
      run = (round > 0 && home == previous_home) ? run + 1 : 1;
      if (run > max_k) {
        return false;
      }
      previous_home = home;
    }
  }
  return true;
}

bool has_repeaters(ConstScheduleView schedule) noexcept {
  for (std::size_t team = 0; team < schedule.rows; ++team) {
    for (std::size_t round = 1; round < schedule.cols; ++round) {
      if (opponent(schedule(team, round)) == opponent(schedule(team, round - 1))) {
        return true;
      }
    }
  }
  return false;
}

bool has_valid_entries(ConstScheduleView schedule) noexcept {
  const std::size_t n = schedule.rows;
  if (n < 2 || schedule.cols != 2 * n - 2) {
    return false;
  }
  const auto limit = static_cast<Entry>(n);
  return std::ranges::all_of(schedule.flat(),
                             [limit](Entry e) { return e != 0 && e >= -limit && e <= limit; });
}

bool is_valid_schedule(ConstScheduleView schedule) {
  if (!has_valid_entries(schedule)) {
    return false;
  }
  const std::size_t n = schedule.rows;
  std::vector<bool> seen(n * n, false);
  for (std::size_t team = 0; team < n; ++team) {
    for (std::size_t round = 0; round < schedule.cols; ++round) {
      const Entry entry = schedule(team, round);
      const std::size_t other = opponent(entry);
      if (other == team || schedule(other, round) != -with_sign_of(entry, team)) {
        return false;
      }
      if (entry > 0) {
        // `team` hosts `other`: each ordered pair must occur exactly once.
        if (seen[team * n + other]) {
          return false;
        }
        seen[team * n + other] = true;
      }
    }
  }
  return true;
}

}  // namespace pyttp
