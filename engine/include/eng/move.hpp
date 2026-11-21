#ifndef ENG_MOVE_HPP
#define ENG_MOVE_HPP

#include "eng/types.hpp"

#include <cstdint>

namespace eng {

enum class MoveKind : std::uint8_t { Normal = 0, Promotion = 1, EnPassant = 2, Castling = 3 };

class Move {
  public:
    using Raw = std::uint16_t;

    static constexpr int to_shift = 6;
    static constexpr int kind_shift = 12;
    static constexpr int promotion_shift = 14;
    static constexpr Raw square_mask = 0x3F;
    static constexpr Raw kind_mask = 0x03;
    static constexpr Raw promotion_mask = 0x03;
    static constexpr Raw none_value = 0;
    static constexpr Raw null_value = Raw{1} | (Raw{1} << to_shift);

    constexpr Move() noexcept = default;

    [[nodiscard]] static constexpr Move none() noexcept { return Move{none_value}; }
    [[nodiscard]] static constexpr Move null() noexcept { return Move{null_value}; }
    [[nodiscard]] static constexpr Move from_raw(Raw value) noexcept { return Move{value}; }

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    [[nodiscard]] static constexpr Move normal(Square from_square, Square to_square) noexcept {
        if (!is_valid_endpoint_pair(from_square, to_square)) {
            return none();
        }
        return encode(from_square, to_square, MoveKind::Normal, PieceType::Knight);
    }

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    [[nodiscard]] static constexpr Move promotion(Square from_square, Square to_square,
                                                  PieceType promotion_piece) noexcept {
        if (!is_valid_endpoint_pair(from_square, to_square) ||
            !is_valid_promotion_piece(promotion_piece)) {
            return none();
        }
        return encode(from_square, to_square, MoveKind::Promotion, promotion_piece);
    }

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    [[nodiscard]] static constexpr Move en_passant(Square from_square, Square to_square) noexcept {
        if (!is_valid_endpoint_pair(from_square, to_square)) {
            return none();
        }
        return encode(from_square, to_square, MoveKind::EnPassant, PieceType::Knight);
    }

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    [[nodiscard]] static constexpr Move castling(Square from_square, Square to_square) noexcept {
        if (!is_valid_endpoint_pair(from_square, to_square)) {
            return none();
        }
        return encode(from_square, to_square, MoveKind::Castling, PieceType::Knight);
    }

    [[nodiscard]] constexpr Raw raw() const noexcept { return value_; }

    [[nodiscard]] constexpr bool is_none() const noexcept { return value_ == none_value; }
    [[nodiscard]] constexpr bool is_null() const noexcept { return value_ == null_value; }
    [[nodiscard]] constexpr bool is_valid() const noexcept { return !is_none(); }

    [[nodiscard]] constexpr Square from() const noexcept {
        if (is_none()) {
            return Square{};
        }
        return Square::from_index(value_ & square_mask);
    }

    [[nodiscard]] constexpr Square to() const noexcept {
        if (is_none()) {
            return Square{};
        }
        return Square::from_index((value_ >> to_shift) & square_mask);
    }

    [[nodiscard]] constexpr MoveKind kind() const noexcept {
        return static_cast<MoveKind>((value_ >> kind_shift) & kind_mask);
    }

    [[nodiscard]] constexpr bool is_promotion() const noexcept {
        return !is_none() && kind() == MoveKind::Promotion;
    }

    [[nodiscard]] constexpr bool is_en_passant() const noexcept {
        return !is_none() && kind() == MoveKind::EnPassant;
    }

    [[nodiscard]] constexpr bool is_castling() const noexcept {
        return !is_none() && kind() == MoveKind::Castling;
    }

    [[nodiscard]] constexpr PieceType promotion_piece() const noexcept {
        if (!is_promotion()) {
            return PieceType::None;
        }
        return promotion_piece_from_bits((value_ >> promotion_shift) & promotion_mask);
    }

    friend constexpr bool operator==(Move lhs, Move rhs) noexcept = default;

  private:
    constexpr explicit Move(Raw value) noexcept : value_{value} {}

    [[nodiscard]] static constexpr bool is_valid_endpoint_pair(Square from_square,
                                                               Square to_square) noexcept {
        return from_square.is_valid() && to_square.is_valid() && from_square != to_square;
    }

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    [[nodiscard]] static constexpr Move encode(Square from_square, Square to_square, MoveKind kind,
                                               PieceType promotion_piece) noexcept {
        const auto promotion_bits = promotion_piece_to_bits(promotion_piece);
        const auto raw_value =
            static_cast<Raw>(static_cast<Raw>(from_square.index()) |
                             static_cast<Raw>(to_square.index() << to_shift) |
                             static_cast<Raw>(static_cast<int>(kind) << kind_shift) |
                             static_cast<Raw>(promotion_bits << promotion_shift));
        return Move{raw_value};
    }

    [[nodiscard]] static constexpr int promotion_piece_to_bits(PieceType promotion_piece) noexcept {
        if (promotion_piece == PieceType::Bishop) {
            return 1;
        }
        if (promotion_piece == PieceType::Rook) {
            return 2;
        }
        if (promotion_piece == PieceType::Queen) {
            return 3;
        }
        return 0;
    }

    [[nodiscard]] static constexpr PieceType
    promotion_piece_from_bits(Raw promotion_bits) noexcept {
        if (promotion_bits == 1) {
            return PieceType::Bishop;
        }
        if (promotion_bits == 2) {
            return PieceType::Rook;
        }
        if (promotion_bits == 3) {
            return PieceType::Queen;
        }
        return PieceType::Knight;
    }

    Raw value_{none_value};
};

static_assert(sizeof(Move) == sizeof(Move::Raw));

} // namespace eng

#endif // ENG_MOVE_HPP
