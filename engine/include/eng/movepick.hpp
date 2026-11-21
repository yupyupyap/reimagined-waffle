#ifndef ENG_MOVEPICK_HPP
#define ENG_MOVEPICK_HPP

#include "eng/move.hpp"
#include "eng/movegen.hpp"
#include "eng/position.hpp"
#include "eng/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace eng {

inline constexpr int move_ordering_max_ply = 64;
inline constexpr int move_ordering_killer_slots = 2;
inline constexpr int move_ordering_history_limit = 16384;
inline constexpr std::size_t move_ordering_piece_square_count =
    static_cast<std::size_t>(Piece::count) * static_cast<std::size_t>(Square::count);
inline constexpr std::size_t move_ordering_continuation_count =
    move_ordering_piece_square_count * move_ordering_piece_square_count;

struct MoveContinuationKey {
    Piece piece;
    Square to;

    [[nodiscard]] constexpr bool is_valid() const noexcept {
        return piece.is_valid() && to.is_valid();
    }

    friend constexpr bool operator==(MoveContinuationKey lhs,
                                     MoveContinuationKey rhs) noexcept = default;
};

[[nodiscard]] int move_ordering_bonus(Depth depth) noexcept;
[[nodiscard]] bool is_quiet_ordering_move(const Position& position, Move move) noexcept;
[[nodiscard]] MoveContinuationKey continuation_key_for_move(const Position& position,
                                                            Move move) noexcept;

class MoveOrderingState {
  public:
    MoveOrderingState() noexcept = default;

    void clear() noexcept;

    void add_killer(Ply ply, Move move) noexcept;
    [[nodiscard]] Move killer(Ply ply, std::size_t slot) const noexcept;
    [[nodiscard]] bool is_killer(Ply ply, Move move) const noexcept;

    void update_quiet(Piece piece, Square target, MoveContinuationKey previous, int bonus) noexcept;
    void update_quiet_move(const Position& position, Move move, MoveContinuationKey previous,
                           int bonus) noexcept;

    [[nodiscard]] int history_score(Piece piece, Square target) const noexcept;
    [[nodiscard]] int continuation_history_score(MoveContinuationKey previous, Piece piece,
                                                 Square target) const noexcept;
    [[nodiscard]] int quiet_score(Piece piece, Square target,
                                  MoveContinuationKey previous) const noexcept;
    [[nodiscard]] bool has_any_killer() const noexcept;
    [[nodiscard]] bool has_any_history() const noexcept;

  private:
    using KillerTable =
        std::array<std::array<Move, move_ordering_killer_slots>, move_ordering_max_ply>;
    using PieceSquareTable = std::array<std::int16_t, move_ordering_piece_square_count>;
    using ContinuationTable = std::array<std::int16_t, move_ordering_continuation_count>;

    [[nodiscard]] static std::size_t killer_index(Ply ply) noexcept;
    [[nodiscard]] static std::size_t piece_square_index(Piece piece, Square square) noexcept;
    [[nodiscard]] static std::size_t continuation_index(MoveContinuationKey previous, Piece piece,
                                                        Square square) noexcept;
    static void update_history_value(std::int16_t& value, int bonus) noexcept;

    KillerTable killers_{};
    PieceSquareTable history_{};
    ContinuationTable continuation_history_{};
};

struct MovePickerContext {
    Move tt_move{Move::none()};
    MoveOrderingState* ordering{nullptr};
    Ply ply;
    MoveContinuationKey previous;
    bool score_captures_with_see{false};
};

class MovePicker {
  public:
    MovePicker(const Position& position, const MoveList& moves, MovePickerContext context) noexcept;
    MovePicker(const Position& position, std::span<const Move> moves,
               MovePickerContext context) noexcept;

    [[nodiscard]] Move next() noexcept;
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

  private:
    struct ScoredMove {
        Move move{Move::none()};
        int score{0};
        std::size_t generation_index{0};
    };

    void add_move(const Position& position, Move move, MovePickerContext context,
                  std::size_t generation_index) noexcept;
    void sort_moves() noexcept;

    std::array<ScoredMove, max_pseudo_legal_moves> moves_{};
    std::size_t size_{0};
    std::size_t next_{0};
};

} // namespace eng

#endif // ENG_MOVEPICK_HPP
