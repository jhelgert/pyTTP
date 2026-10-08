#include "local_search.hpp"

#include <algorithm>
#include <array>
#include <optional>

#include "evaluate.hpp"
#include "neighborhoods.hpp"

namespace pyttp {

namespace {

class Scanner {
 public:
  Scanner(ConstScheduleView start, ConstDistanceView distances, int max_k)
      : start_(start),
        distances_(distances),
        max_k_(max_k),
        scratch_(start.flat().begin(), start.flat().end()),
        current_objective_(objective(start, distances)) {
    best_objective_.fill(current_objective_);
  }

  /// Apply `move` to a fresh copy of the start schedule and remember the
  /// result if it is the best feasible improvement of `which` so far.
  template <typename Move>
  void try_move(Neighborhood which, Move&& move) {
    std::ranges::copy(start_.flat(), scratch_.begin());
    const ScheduleView view{.data = scratch_.data(), .rows = start_.rows, .cols = start_.cols};
    move(view);
    const ConstScheduleView const_view{
        .data = scratch_.data(), .rows = start_.rows, .cols = start_.cols};
    const auto idx = static_cast<std::size_t>(which);
    const std::int64_t value = objective(const_view, distances_);
    if (value >= best_objective_[idx] || !satisfies_stand_limits(const_view, max_k_)) {
      return;
    }
    best_objective_[idx] = value;
    best_schedule_[idx] = scratch_;
  }

  [[nodiscard]] std::vector<Candidate> finish() {
    std::vector<Candidate> result;
    for (std::size_t idx = 0; idx < kNeighborhoodCount; ++idx) {
      if (auto& schedule = best_schedule_[idx]) {
        result.push_back({.neighborhood = static_cast<Neighborhood>(idx),
                          .objective = best_objective_[idx],
                          .schedule = std::move(*schedule)});
      }
    }
    return result;
  }

 private:
  ConstScheduleView start_;
  ConstDistanceView distances_;
  int max_k_;
  std::vector<Entry> scratch_;
  std::int64_t current_objective_;
  std::array<std::int64_t, kNeighborhoodCount> best_objective_{};
  std::array<std::optional<std::vector<Entry>>, kNeighborhoodCount> best_schedule_{};
};

}  // namespace

std::vector<Candidate> local_search_step(ConstScheduleView schedule, ConstDistanceView distances,
                                         int max_k) {
  const std::size_t teams = schedule.rows;
  const std::size_t rounds = schedule.cols;
  Scanner scan(schedule, distances, max_k);

  for (std::size_t i = 0; i < teams; ++i) {
    for (std::size_t j = i + 1; j < teams; ++j) {
      scan.try_move(Neighborhood::SwapHomes, [&](ScheduleView s) { swap_homes(s, i, j); });
      scan.try_move(Neighborhood::SwapTeams, [&](ScheduleView s) { swap_teams(s, i, j); });
      for (std::size_t r = 0; r < rounds; ++r) {
        scan.try_move(Neighborhood::PartialSwapTeams,
                      [&](ScheduleView s) { partial_swap_teams(s, i, j, r); });
      }
    }
  }
  for (std::size_t r1 = 0; r1 < rounds; ++r1) {
    for (std::size_t r2 = r1 + 1; r2 < rounds; ++r2) {
      scan.try_move(Neighborhood::SwapRounds, [&](ScheduleView s) { swap_rounds(s, r1, r2); });
      for (std::size_t t = 0; t < teams; ++t) {
        scan.try_move(Neighborhood::PartialSwapRounds,
                      [&](ScheduleView s) { partial_swap_rounds(s, t, r1, r2); });
      }
    }
  }
  return scan.finish();
}

}  // namespace pyttp
