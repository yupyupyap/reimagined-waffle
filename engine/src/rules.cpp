#include <eng/rules.hpp>

#include <eng/bitboard.hpp>

#include <algorithm>
#include <cstddef>

namespace eng {
namespace {

constexpr int fifty_move_rule_halfmove_threshold = 100;

[[nodiscard]] bool has_only_kings_and_minors(const Position& position) noexcept {
    return position.pieces(Color::White, PieceType::Pawn) == empty_bitboard &&
           position.pieces(Color::Black, PieceType::Pawn) == empty_bitboard &&
           position.pieces(Color::White, PieceType::Rook) == empty_bitboard &&
           position.pieces(Color::Black, PieceType::Rook) == empty_bitboard &&
           position.pieces(Color::White, PieceType::Queen) == empty_bitboard &&
           position.pieces(Color::Black, PieceType::Queen) == empty_bitboard;
}

[[nodiscard]] bool is_dark_square(Square square) noexcept {
    return ((square.file().index() + square.rank().index()) % 2) != 0;
}

[[nodiscard]] bool all_bishops_on_one_square_color(Bitboard bishops) noexcept {
    if (bishops == empty_bitboard) {
        return false;
    }

    const auto first = lsb(bishops);
    const auto dark = is_dark_square(first);
    while (bishops != empty_bitboard) {
        if (is_dark_square(pop_lsb(bishops)) != dark) {
            return false;
        }
    }
    return true;
}

} // namespace

bool is_fifty_move_rule_draw(const Position& position) noexcept {
    return position.halfmove_clock() >= fifty_move_rule_halfmove_threshold;
}

bool is_insufficient_material(const Position& position) noexcept {
    if (!has_only_kings_and_minors(position)) {
        return false;
    }

    const auto bishops = position.pieces(Color::White, PieceType::Bishop) |
                         position.pieces(Color::Black, PieceType::Bishop);
    const auto knights = position.pieces(Color::White, PieceType::Knight) |
                         position.pieces(Color::Black, PieceType::Knight);
    const auto bishop_count = popcount(bishops);
    const auto knight_count = popcount(knights);
    const auto minor_count = bishop_count + knight_count;

    if (minor_count == 0) {
        return true;
    }
    if (minor_count == 1) {
        return true;
    }
    if (knight_count == 0 && all_bishops_on_one_square_color(bishops)) {
        return true;
    }

    return false;
}

bool is_threefold_repetition(const Position& position,
                             std::span<const ZobristKey> previous_keys) noexcept {
    const auto limit =
        std::min(previous_keys.size(), static_cast<std::size_t>(position.halfmove_clock()));
    std::size_t matches = 0;
    for (std::size_t offset = 0; offset < limit; ++offset) {
        const auto key = previous_keys[previous_keys.size() - 1U - offset];
        if (key == position.zobrist_key()) {
            ++matches;
            if (matches >= 2U) {
                return true;
            }
        }
    }

    return false;
}

} // namespace eng
