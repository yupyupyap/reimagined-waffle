#ifndef ENG_SEE_HPP
#define ENG_SEE_HPP

#include "eng/move.hpp"
#include "eng/position.hpp"
#include "eng/types.hpp"

namespace eng {

[[nodiscard]] Score static_exchange_score(const Position& position, Move move) noexcept;
[[nodiscard]] bool static_exchange_ge(const Position& position, Move move,
                                      Score threshold = Score::centipawns(0)) noexcept;

} // namespace eng

#endif // ENG_SEE_HPP
