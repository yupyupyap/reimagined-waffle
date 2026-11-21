#ifndef ENG_RULES_HPP
#define ENG_RULES_HPP

#include "eng/position.hpp"

#include <span>

namespace eng {

[[nodiscard]] bool is_fifty_move_rule_draw(const Position& position) noexcept;
[[nodiscard]] bool is_insufficient_material(const Position& position) noexcept;
[[nodiscard]] bool is_threefold_repetition(const Position& position,
                                           std::span<const ZobristKey> previous_keys) noexcept;

} // namespace eng

#endif // ENG_RULES_HPP
