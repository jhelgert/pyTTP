#include "neighborhoods.hpp"

#include <utility>
#include <vector>

namespace pyttp {

namespace {

/// Swap the entries of teams `i` and `j` in a single round and re-point their
/// opponents. Requires that `i` and `j` do not play each other in `round`.
void swap_teams_in_round(ScheduleView s, std::size_t i, std::size_t j, std::size_t round) {
  const std::size_t a = opponent(s(i, round));
  const std::size_t b = opponent(s(j, round));
  std::swap(s(i, round), s(j, round));
  s(a, round) = with_sign_of(s(a, round), j);
  s(b, round) = with_sign_of(s(b, round), i);
}

}  // namespace

void swap_homes(ScheduleView s, std::size_t i, std::size_t j) {
  for (std::size_t r = 0; r < s.cols; ++r) {
    if (opponent(s(i, r)) == j) {
      s(i, r) = -s(i, r);
      s(j, r) = -s(j, r);
    }
  }
}

void swap_rounds(ScheduleView s, std::size_t r1, std::size_t r2) {
  for (std::size_t t = 0; t < s.rows; ++t) {
    std::swap(s(t, r1), s(t, r2));
  }
}

void swap_teams(ScheduleView s, std::size_t i, std::size_t j) {
  for (std::size_t r = 0; r < s.cols; ++r) {
    if (opponent(s(i, r)) != j) {
      swap_teams_in_round(s, i, j, r);
    }
  }
}

void partial_swap_rounds(ScheduleView s, std::size_t team, std::size_t r1, std::size_t r2) {
  if (r1 == r2) {
    return;
  }
  // The teams whose entries in r1 and r2 have to be exchanged form the
  // connected component of `team` in the graph whose edges are the games of r1
  // and r2.
  std::vector<bool> in_component(s.rows, false);
  std::vector<std::size_t> queue{team};
  in_component[team] = true;
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const std::size_t t = queue[head];
    for (const std::size_t r : {r1, r2}) {
      const std::size_t o = opponent(s(t, r));
      if (!in_component[o]) {
        in_component[o] = true;
        queue.push_back(o);
      }
    }
  }
  for (const std::size_t t : queue) {
    std::swap(s(t, r1), s(t, r2));
  }
}

void partial_swap_teams(ScheduleView s, std::size_t i, std::size_t j, std::size_t round) {
  if (opponent(s(i, round)) == j) {
    return;
  }
  // Closure: once the teams `a` and `b` (opponents of i and j in a swapped
  // round) are involved, all games of i and j against them must be swapped.
  std::vector<bool> round_selected(s.cols, false);
  std::vector<bool> team_touched(s.rows, false);
  std::vector<std::size_t> rounds{round};
  round_selected[round] = true;
  for (std::size_t head = 0; head < rounds.size(); ++head) {
    const std::size_t r = rounds[head];
    for (const std::size_t x : {opponent(s(i, r)), opponent(s(j, r))}) {
      if (team_touched[x]) {
        continue;
      }
      team_touched[x] = true;
      for (std::size_t q = 0; q < s.cols; ++q) {
        if (!round_selected[q] && (opponent(s(i, q)) == x || opponent(s(j, q)) == x)) {
          round_selected[q] = true;
          rounds.push_back(q);
        }
      }
    }
  }
  for (const std::size_t r : rounds) {
    swap_teams_in_round(s, i, j, r);
  }
}

}  // namespace pyttp
