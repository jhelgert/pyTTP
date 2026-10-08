#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "schedule.hpp"

namespace pyttp {

enum class Neighborhood : std::uint8_t {
  SwapHomes,
  SwapRounds,
  SwapTeams,
  PartialSwapRounds,
  PartialSwapTeams,
};

inline constexpr std::size_t kNeighborhoodCount = 5;

[[nodiscard]] constexpr std::string_view name(Neighborhood n) noexcept {
  switch (n) {
    case Neighborhood::SwapHomes:
      return "swap_homes";
    case Neighborhood::SwapRounds:
      return "swap_rounds";
    case Neighborhood::SwapTeams:
      return "swap_teams";
    case Neighborhood::PartialSwapRounds:
      return "partial_swap_rounds";
    case Neighborhood::PartialSwapTeams:
      return "partial_swap_teams";
  }
  return {};
}

/// Best improving neighbor found in one neighborhood.
struct Candidate {
  Neighborhood neighborhood;
  std::int64_t objective;
  std::vector<Entry> schedule;  // row-major, same shape as the input
};

/// Scan every neighborhood completely and return, per neighborhood, the best
/// neighbor that is strictly better than `schedule` and feasible, i.e. satisfies the
/// stand limits and has no repeaters. Neighborhoods without such a neighbor contribute no
/// candidate. Candidates are ordered by neighborhood.
[[nodiscard]] std::vector<Candidate> local_search_step(ConstScheduleView schedule,
                                                       ConstDistanceView distances, int max_k);

}  // namespace pyttp
