#ifndef ENG_PERFT_HPP
#define ENG_PERFT_HPP

#include "eng/move.hpp"
#include "eng/position.hpp"

#include <cstdint>
#include <vector>

namespace eng {

using PerftNodeCount = std::uint64_t;

struct PerftDivideEntry {
    Move move;
    PerftNodeCount nodes{0};

    friend bool operator==(const PerftDivideEntry& lhs,
                           const PerftDivideEntry& rhs) noexcept = default;
};

[[nodiscard]] PerftNodeCount perft(Position& position, int depth) noexcept;
[[nodiscard]] std::vector<PerftDivideEntry> divide(Position& position, int depth);

} // namespace eng

#endif // ENG_PERFT_HPP
