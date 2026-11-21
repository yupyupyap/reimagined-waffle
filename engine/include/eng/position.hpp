#ifndef ENG_POSITION_HPP
#define ENG_POSITION_HPP

#include "eng/move.hpp"
#include "eng/types.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

namespace eng {

using ZobristKey = std::uint64_t;

enum class CastlingSide : std::uint8_t { KingSide = 0, QueenSide = 1 };

class CastlingRights {
  public:
    using Raw = std::uint8_t;

    static constexpr Raw white_king_side_mask = 0x01;
    static constexpr Raw white_queen_side_mask = 0x02;
    static constexpr Raw black_king_side_mask = 0x04;
    static constexpr Raw black_queen_side_mask = 0x08;
    static constexpr Raw all_mask =
        white_king_side_mask | white_queen_side_mask | black_king_side_mask | black_queen_side_mask;

    constexpr CastlingRights() noexcept = default;

    [[nodiscard]] static constexpr CastlingRights none() noexcept { return CastlingRights{}; }

    [[nodiscard]] static constexpr CastlingRights all() noexcept {
        return CastlingRights{all_mask};
    }

    [[nodiscard]] static constexpr CastlingRights from_raw(Raw raw) noexcept {
        if ((raw & ~all_mask) != 0U) {
            return none();
        }
        return CastlingRights{raw};
    }

    [[nodiscard]] constexpr Raw raw() const noexcept { return mask_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return mask_ == 0U; }

    [[nodiscard]] constexpr bool has(Color color, CastlingSide side) const noexcept {
        return (mask_ & mask_for(color, side)) != 0U;
    }

    [[nodiscard]] constexpr bool white_king_side() const noexcept {
        return has(Color::White, CastlingSide::KingSide);
    }

    [[nodiscard]] constexpr bool white_queen_side() const noexcept {
        return has(Color::White, CastlingSide::QueenSide);
    }

    [[nodiscard]] constexpr bool black_king_side() const noexcept {
        return has(Color::Black, CastlingSide::KingSide);
    }

    [[nodiscard]] constexpr bool black_queen_side() const noexcept {
        return has(Color::Black, CastlingSide::QueenSide);
    }

    constexpr void add(Color color, CastlingSide side) noexcept { mask_ |= mask_for(color, side); }
    constexpr void remove(Color color, CastlingSide side) noexcept {
        mask_ = static_cast<Raw>(mask_ & ~mask_for(color, side));
    }

    constexpr void remove(Color color) noexcept {
        remove(color, CastlingSide::KingSide);
        remove(color, CastlingSide::QueenSide);
    }

    friend constexpr bool operator==(CastlingRights lhs, CastlingRights rhs) noexcept = default;

  private:
    constexpr explicit CastlingRights(Raw mask) noexcept : mask_{mask} {}

    [[nodiscard]] static constexpr Raw mask_for(Color color, CastlingSide side) noexcept {
        if (color == Color::White && side == CastlingSide::KingSide) {
            return white_king_side_mask;
        }
        if (color == Color::White && side == CastlingSide::QueenSide) {
            return white_queen_side_mask;
        }
        if (color == Color::Black && side == CastlingSide::KingSide) {
            return black_king_side_mask;
        }
        if (color == Color::Black && side == CastlingSide::QueenSide) {
            return black_queen_side_mask;
        }
        return 0;
    }

    Raw mask_{0};
};

struct StateInfo {
    Piece moved_piece;
    Piece captured_piece;
    Square captured_square;
    CastlingRights castling_rights;
    Square en_passant_square;
    int halfmove_clock{0};
    int fullmove_number{1};
    ZobristKey zobrist_key{0};
};

enum class FenErrorCode : std::uint8_t {
    MissingField = 0,
    TooManyFields = 1,
    InvalidPiecePlacement = 2,
    InvalidSideToMove = 3,
    InvalidCastlingRights = 4,
    InvalidEnPassantSquare = 5,
    InvalidHalfmoveClock = 6,
    InvalidFullmoveNumber = 7,
    InvalidBoardState = 8
};

struct FenError {
    FenErrorCode code{FenErrorCode::InvalidBoardState};
    std::string message;

    friend bool operator==(const FenError& lhs, const FenError& rhs) = default;
};

class Position {
  public:
    using PieceBitboards = std::array<Bitboard, Piece::count>;
    using ColorOccupancies = std::array<Bitboard, 2>;
    using Mailbox = std::array<Piece, Square::count>;
    using FenResult = std::expected<Position, FenError>;

    static constexpr std::string_view startpos_fen =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    Position() noexcept;

    [[nodiscard]] static Position startpos();
    [[nodiscard]] static FenResult from_fen(std::string_view fen);

    [[nodiscard]] std::string to_fen() const;

    [[nodiscard]] Piece piece_on(Square square) const noexcept;
    [[nodiscard]] Bitboard pieces(Piece piece) const noexcept;
    [[nodiscard]] Bitboard pieces(Color color, PieceType type) const noexcept;
    [[nodiscard]] Bitboard occupancy(Color color) const noexcept;
    [[nodiscard]] Bitboard occupancy() const noexcept { return occupancy_; }
    [[nodiscard]] const Mailbox& mailbox() const noexcept { return mailbox_; }

    [[nodiscard]] Color side_to_move() const noexcept { return side_to_move_; }
    [[nodiscard]] CastlingRights castling_rights() const noexcept { return castling_rights_; }
    [[nodiscard]] Square en_passant_square() const noexcept { return en_passant_square_; }
    [[nodiscard]] int halfmove_clock() const noexcept { return halfmove_clock_; }
    [[nodiscard]] int fullmove_number() const noexcept { return fullmove_number_; }
    [[nodiscard]] ZobristKey zobrist_key() const noexcept { return zobrist_key_; }

    [[nodiscard]] bool has_consistent_occupancies() const noexcept;
    [[nodiscard]] ZobristKey recompute_zobrist_key() const noexcept;
    [[nodiscard]] bool make_move(Move move, StateInfo& state) noexcept;
    void unmake_move(Move move, const StateInfo& state) noexcept;
    [[nodiscard]] bool make_null_move(StateInfo& state) noexcept;
    void unmake_null_move(const StateInfo& state) noexcept;

  private:
    friend class PositionBuilder;

    void clear() noexcept;
    void put_piece(Piece piece, Square square) noexcept;
    void remove_piece(Piece piece, Square square) noexcept;
    void move_piece(Piece piece, Square from_square, Square to_square) noexcept;
    void refresh_derived_state() noexcept;

    PieceBitboards piece_bitboards_{};
    ColorOccupancies occupancies_{};
    Bitboard occupancy_{0};
    Mailbox mailbox_{};
    Color side_to_move_{Color::White};
    CastlingRights castling_rights_;
    Square en_passant_square_;
    int halfmove_clock_{0};
    int fullmove_number_{1};
    ZobristKey zobrist_key_{0};
};

static_assert(sizeof(CastlingRights) == sizeof(CastlingRights::Raw));

} // namespace eng

#endif // ENG_POSITION_HPP
