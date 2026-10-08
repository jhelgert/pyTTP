#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

namespace pyttp {

/// Value type of a schedule entry.
///
/// Entry `S(t, r)` describes the game of team `t` (0-based) in round `r`
/// (0-based). Its absolute value minus one is the 0-based index of the
/// opponent; a positive sign means a home game, a negative sign an away game.
using Entry = std::int32_t;

/// Non-owning, row-major view of a two-dimensional array.
template <typename T>
struct MatrixView {
  T* data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;

  [[nodiscard]] constexpr T& operator()(std::size_t r, std::size_t c) const noexcept {
    return data[r * cols + c];
  }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return rows * cols; }
  [[nodiscard]] constexpr std::span<T> flat() const noexcept { return {data, size()}; }
};

using ScheduleView = MatrixView<Entry>;
using ConstScheduleView = MatrixView<const Entry>;
using ConstDistanceView = MatrixView<const std::int32_t>;

/// 0-based index of the opponent encoded in a schedule entry.
[[nodiscard]] constexpr std::size_t opponent(Entry e) noexcept {
  return static_cast<std::size_t>(std::abs(e)) - 1;
}

/// Entry value that refers to `team` (0-based) with the sign of `like`.
[[nodiscard]] constexpr Entry with_sign_of(Entry like, std::size_t team) noexcept {
  const auto value = static_cast<Entry>(team + 1);
  return like < 0 ? -value : value;
}

}  // namespace pyttp
