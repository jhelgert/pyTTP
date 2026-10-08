#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "evaluate.hpp"
#include "local_search.hpp"
#include "neighborhoods.hpp"
#include "schedule.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using ScheduleIn = nb::ndarray<const pyttp::Entry, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using DistancesIn = ScheduleIn;  // both are C-contiguous int32 matrices
using ScheduleOut = nb::ndarray<nb::numpy, pyttp::Entry, nb::ndim<2>>;

pyttp::ConstScheduleView view_of(const ScheduleIn& a) {
  return {.data = a.data(), .rows = a.shape(0), .cols = a.shape(1)};
}

/// Moves a vector onto the heap and hands its ownership to a numpy array.
ScheduleOut to_numpy(std::vector<pyttp::Entry> values, std::size_t rows, std::size_t cols) {
  auto* owner = new std::vector<pyttp::Entry>(std::move(values));
  const nb::capsule capsule(
      owner, [](void* p) noexcept { delete static_cast<std::vector<pyttp::Entry>*>(p); });
  return ScheduleOut(owner->data(), {rows, cols}, capsule);
}

void check_schedule(const ScheduleIn& s) {
  const std::size_t n = s.shape(0);
  if (n < 2 || n % 2 != 0 || s.shape(1) != 2 * n - 2) {
    throw std::invalid_argument(
        "schedule must have shape (n, 2n - 2) with an even number of teams n >= 2");
  }
  if (!pyttp::has_valid_entries(view_of(s))) {
    throw std::invalid_argument("schedule entries must be non-zero team numbers in [-n, n]");
  }
}

void check_distances(const DistancesIn& d, std::size_t teams) {
  if (d.shape(0) != teams || d.shape(1) != teams) {
    throw std::invalid_argument("distance matrix must have shape (n, n)");
  }
}

void check_index(std::size_t value, std::size_t bound, const char* what) {
  if (value >= bound) {
    throw nb::index_error((std::string(what) + " index out of range").c_str());
  }
}

/// Copy the input, apply `move` to the copy and return it as a numpy array.
template <typename Move>
ScheduleOut apply_move(const ScheduleIn& schedule, Move&& move) {
  check_schedule(schedule);
  std::vector<pyttp::Entry> copy(schedule.data(), schedule.data() + schedule.size());
  move(pyttp::ScheduleView{
      .data = copy.data(), .rows = schedule.shape(0), .cols = schedule.shape(1)});
  return to_numpy(std::move(copy), schedule.shape(0), schedule.shape(1));
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "Native core of pyttp: schedule evaluation, neighborhoods and local search.";

  m.def(
      "objective",
      [](const ScheduleIn& schedule, const DistancesIn& distances) {
        check_schedule(schedule);
        check_distances(distances, schedule.shape(0));
        return pyttp::objective(view_of(schedule), view_of(distances));
      },
      "schedule"_a, "distances"_a);

  m.def(
      "satisfies_stand_limits",
      [](const ScheduleIn& schedule, int max_k) {
        check_schedule(schedule);
        return pyttp::satisfies_stand_limits(view_of(schedule), max_k);
      },
      "schedule"_a, "max_k"_a);

  m.def(
      "has_repeaters",
      [](const ScheduleIn& schedule) {
        check_schedule(schedule);
        return pyttp::has_repeaters(view_of(schedule));
      },
      "schedule"_a);

  m.def(
      "is_valid_schedule",
      [](const ScheduleIn& schedule) { return pyttp::is_valid_schedule(view_of(schedule)); },
      "schedule"_a);

  m.def(
      "swap_homes",
      [](const ScheduleIn& schedule, std::size_t i, std::size_t j) {
        check_schedule(schedule);
        check_index(i, schedule.shape(0), "team");
        check_index(j, schedule.shape(0), "team");
        return apply_move(schedule, [&](pyttp::ScheduleView s) { pyttp::swap_homes(s, i, j); });
      },
      "schedule"_a, "i"_a, "j"_a);

  m.def(
      "swap_rounds",
      [](const ScheduleIn& schedule, std::size_t r1, std::size_t r2) {
        check_schedule(schedule);
        check_index(r1, schedule.shape(1), "round");
        check_index(r2, schedule.shape(1), "round");
        return apply_move(schedule, [&](pyttp::ScheduleView s) { pyttp::swap_rounds(s, r1, r2); });
      },
      "schedule"_a, "r1"_a, "r2"_a);

  m.def(
      "swap_teams",
      [](const ScheduleIn& schedule, std::size_t i, std::size_t j) {
        check_schedule(schedule);
        check_index(i, schedule.shape(0), "team");
        check_index(j, schedule.shape(0), "team");
        return apply_move(schedule, [&](pyttp::ScheduleView s) { pyttp::swap_teams(s, i, j); });
      },
      "schedule"_a, "i"_a, "j"_a);

  m.def(
      "partial_swap_rounds",
      [](const ScheduleIn& schedule, std::size_t team, std::size_t r1, std::size_t r2) {
        check_schedule(schedule);
        check_index(team, schedule.shape(0), "team");
        check_index(r1, schedule.shape(1), "round");
        check_index(r2, schedule.shape(1), "round");
        return apply_move(
            schedule, [&](pyttp::ScheduleView s) { pyttp::partial_swap_rounds(s, team, r1, r2); });
      },
      "schedule"_a, "team"_a, "r1"_a, "r2"_a);

  m.def(
      "partial_swap_teams",
      [](const ScheduleIn& schedule, std::size_t i, std::size_t j, std::size_t round) {
        check_schedule(schedule);
        check_index(i, schedule.shape(0), "team");
        check_index(j, schedule.shape(0), "team");
        check_index(round, schedule.shape(1), "round");
        return apply_move(
            schedule, [&](pyttp::ScheduleView s) { pyttp::partial_swap_teams(s, i, j, round); });
      },
      "schedule"_a, "i"_a, "j"_a, "round"_a);

  m.def(
      "local_search_step",
      [](const ScheduleIn& schedule, const DistancesIn& distances, int max_k) {
        check_schedule(schedule);
        check_distances(distances, schedule.shape(0));
        std::vector<pyttp::Candidate> candidates;
        {
          const nb::gil_scoped_release release;
          candidates = pyttp::local_search_step(view_of(schedule), view_of(distances), max_k);
        }
        std::vector<std::tuple<std::string, std::int64_t, nb::object>> result;
        result.reserve(candidates.size());
        for (auto& c : candidates) {
          result.emplace_back(
              std::string(pyttp::name(c.neighborhood)), c.objective,
              nb::cast(to_numpy(std::move(c.schedule), schedule.shape(0), schedule.shape(1))));
        }
        return result;
      },
      "schedule"_a, "distances"_a, "max_k"_a,
      "Scan all neighborhoods once. Returns a list of (neighborhood, objective, schedule) "
      "tuples, one per neighborhood that contains a feasible improving neighbor.");
}
