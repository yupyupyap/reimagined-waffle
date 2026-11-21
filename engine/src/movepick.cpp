#include <eng/movepick.hpp>

#include <eng/see.hpp>

#include <algorithm>
#include <cstdlib>

namespace eng {
namespace {

constexpr int history_bonus_min = 32;
constexpr int history_bonus_max = 2000;
constexpr int history_bonus_depth_scale = 300;
constexpr int history_bonus_offset = 250;
constexpr int tt_move_score = 2'000'000;
constexpr int capture_stage_score = 1'000'000;
constexpr int winning_see_capture_score = 1'100'000;
constexpr int equal_see_capture_score = 1'000'000;
constexpr int losing_see_capture_score = 100'000;
constexpr int first_killer_score = 800'000;
constexpr int second_killer_score = 790'000;
constexpr int quiet_stage_score = 400'000;
constexpr int promotion_order_bonus = 900;
constexpr int capture_order_scale = 16;
constexpr int see_order_scale = 64;
constexpr int see_order_min = -2000;
constexpr int see_order_max = 2000;
constexpr std::array<int, Piece::type_count> move_order_piece_values{100, 320, 330, 500, 900, 0};

[[nodiscard]] constexpr std::size_t piece_type_index(PieceType type) noexcept {
    return static_cast<std::size_t>(type);
}

[[nodiscard]] int piece_order_value(PieceType type) noexcept {
    if (!is_valid_piece_type(type)) {
        return 0;
    }
    return move_order_piece_values[piece_type_index(type)];
}

[[nodiscard]] Piece captured_piece_for_move(const Position& position, Move move) noexcept {
    if (move.is_en_passant()) {
        return Piece::make(opposite(position.side_to_move()), PieceType::Pawn);
    }
    return position.piece_on(move.to());
}

[[nodiscard]] bool move_is_capture(const Position& position, Move move) noexcept {
    return move.is_valid() && (move.is_en_passant() || position.piece_on(move.to()).is_valid());
}

[[nodiscard]] int capture_order_score(const Position& position, Move move) noexcept {
    const auto attacker = position.piece_on(move.from());
    const auto captured = captured_piece_for_move(position, move);

    int score = 0;
    if (captured.is_valid()) {
        score += (piece_order_value(captured.type()) * capture_order_scale) -
                 piece_order_value(attacker.type());
    }
    if (move.is_promotion()) {
        score += promotion_order_bonus + piece_order_value(move.promotion_piece());
    }
    return score;
}

[[nodiscard]] int see_capture_order_score(const Position& position, Move move) noexcept {
    const auto see_score = static_exchange_score(position, move).value();
    int stage_score = equal_see_capture_score;
    if (see_score > 0) {
        stage_score = winning_see_capture_score;
    } else if (see_score < 0) {
        stage_score = losing_see_capture_score;
    }

    const auto scaled_see = std::clamp(see_score, see_order_min, see_order_max) * see_order_scale;
    return stage_score + scaled_see + capture_order_score(position, move);
}

[[nodiscard]] Piece placed_piece_for_move(const Position& position, Move move) noexcept {
    auto piece = position.piece_on(move.from());
    if (!piece.is_valid()) {
        return {};
    }
    if (move.is_promotion() && is_valid_promotion_piece(move.promotion_piece())) {
        piece = Piece::make(piece.color(), move.promotion_piece());
    }
    return piece;
}

[[nodiscard]] int quiet_order_score(const Position& position, Move move,
                                    MovePickerContext context) noexcept {
    const auto piece = position.piece_on(move.from());
    if (!piece.is_valid()) {
        return quiet_stage_score;
    }

    if (context.ordering != nullptr) {
        const auto first_killer = context.ordering->killer(context.ply, 0);
        if (first_killer.is_valid() && move == first_killer) {
            return first_killer_score;
        }

        const auto second_killer = context.ordering->killer(context.ply, 1);
        if (second_killer.is_valid() && move == second_killer) {
            return second_killer_score;
        }

        return quiet_stage_score +
               context.ordering->quiet_score(piece, move.to(), context.previous);
    }

    return quiet_stage_score;
}

[[nodiscard]] int move_order_score(const Position& position, Move move,
                                   MovePickerContext context) noexcept {
    if (context.tt_move.is_valid() && move == context.tt_move) {
        return tt_move_score;
    }
    if (move_is_capture(position, move) || move.is_promotion()) {
        if (context.score_captures_with_see) {
            return see_capture_order_score(position, move);
        }
        return capture_stage_score + capture_order_score(position, move);
    }
    return quiet_order_score(position, move, context);
}

} // namespace

int move_ordering_bonus(Depth depth) noexcept {
    const auto scaled = (history_bonus_depth_scale * depth.value()) - history_bonus_offset;
    return std::clamp(scaled, history_bonus_min, history_bonus_max);
}

bool is_quiet_ordering_move(const Position& position, Move move) noexcept {
    return move.is_valid() && !move.is_null() && !move.is_promotion() &&
           !move_is_capture(position, move);
}

MoveContinuationKey continuation_key_for_move(const Position& position, Move move) noexcept {
    if (!move.is_valid() || move.is_null()) {
        return {};
    }

    const auto piece = placed_piece_for_move(position, move);
    if (!piece.is_valid()) {
        return {};
    }

    return MoveContinuationKey{.piece = piece, .to = move.to()};
}

void MoveOrderingState::clear() noexcept {
    killers_ = {};
    history_ = {};
    continuation_history_ = {};
}

void MoveOrderingState::add_killer(Ply ply, Move move) noexcept {
    const auto index = killer_index(ply);
    if (index >= killers_.size() || !move.is_valid() || move.is_null()) {
        return;
    }

    auto& killers = killers_[index];
    if (killers[0] == move) {
        return;
    }
    if (killers[1] == move) {
        std::swap(killers[0], killers[1]);
        return;
    }

    killers[1] = killers[0];
    killers[0] = move;
}

Move MoveOrderingState::killer(Ply ply, std::size_t slot) const noexcept {
    const auto index = killer_index(ply);
    if (index >= killers_.size() || slot >= move_ordering_killer_slots) {
        return Move::none();
    }
    return killers_[index][slot];
}

bool MoveOrderingState::is_killer(Ply ply, Move move) const noexcept {
    return move.is_valid() && (killer(ply, 0) == move || killer(ply, 1) == move);
}

void MoveOrderingState::update_quiet(Piece piece, Square target, MoveContinuationKey previous,
                                     int bonus) noexcept {
    if (!piece.is_valid() || !target.is_valid()) {
        return;
    }

    update_history_value(history_[piece_square_index(piece, target)], bonus);
    if (previous.is_valid()) {
        update_history_value(continuation_history_[continuation_index(previous, piece, target)],
                             bonus);
    }
}

void MoveOrderingState::update_quiet_move(const Position& position, Move move,
                                          MoveContinuationKey previous, int bonus) noexcept {
    if (!is_quiet_ordering_move(position, move)) {
        return;
    }

    update_quiet(position.piece_on(move.from()), move.to(), previous, bonus);
}

int MoveOrderingState::history_score(Piece piece, Square target) const noexcept {
    if (!piece.is_valid() || !target.is_valid()) {
        return 0;
    }
    return history_[piece_square_index(piece, target)];
}

int MoveOrderingState::continuation_history_score(MoveContinuationKey previous, Piece piece,
                                                  Square target) const noexcept {
    if (!previous.is_valid() || !piece.is_valid() || !target.is_valid()) {
        return 0;
    }
    return continuation_history_[continuation_index(previous, piece, target)];
}

int MoveOrderingState::quiet_score(Piece piece, Square target,
                                   MoveContinuationKey previous) const noexcept {
    return history_score(piece, target) + continuation_history_score(previous, piece, target);
}

bool MoveOrderingState::has_any_killer() const noexcept {
    return std::ranges::any_of(killers_, [](const auto& killers) {
        return killers[0].is_valid() || killers[1].is_valid();
    });
}

bool MoveOrderingState::has_any_history() const noexcept {
    return std::ranges::any_of(history_, [](std::int16_t value) { return value != 0; }) ||
           std::ranges::any_of(continuation_history_,
                               [](std::int16_t value) { return value != 0; });
}

std::size_t MoveOrderingState::killer_index(Ply ply) noexcept {
    if (ply.value() < 0 || ply.value() >= move_ordering_max_ply) {
        return move_ordering_max_ply;
    }
    return static_cast<std::size_t>(ply.value());
}

std::size_t MoveOrderingState::piece_square_index(Piece piece, Square square) noexcept {
    return (static_cast<std::size_t>(piece.index()) * Square::count) +
           static_cast<std::size_t>(square.index());
}

std::size_t MoveOrderingState::continuation_index(MoveContinuationKey previous, Piece piece,
                                                  Square square) noexcept {
    return (((static_cast<std::size_t>(previous.piece.index()) * Square::count) +
             static_cast<std::size_t>(previous.to.index())) *
            Piece::count * Square::count) +
           piece_square_index(piece, square);
}

void MoveOrderingState::update_history_value(std::int16_t& value, int bonus) noexcept {
    const auto clamped_bonus =
        std::clamp(bonus, -move_ordering_history_limit, move_ordering_history_limit);
    const auto current = static_cast<int>(value);
    const auto updated = current + clamped_bonus -
                         ((current * std::abs(clamped_bonus)) / move_ordering_history_limit);
    value = static_cast<std::int16_t>(
        std::clamp(updated, -move_ordering_history_limit, move_ordering_history_limit));
}

MovePicker::MovePicker(const Position& position, const MoveList& moves,
                       MovePickerContext context) noexcept {
    for (std::size_t index = 0; index < moves.size(); ++index) {
        add_move(position, moves[index], context, index);
    }
    sort_moves();
}

MovePicker::MovePicker(const Position& position, std::span<const Move> moves,
                       MovePickerContext context) noexcept {
    const auto count = std::min(moves.size(), moves_.size());
    for (std::size_t index = 0; index < count; ++index) {
        add_move(position, moves[index], context, index);
    }
    sort_moves();
}

Move MovePicker::next() noexcept {
    if (next_ >= size_) {
        return Move::none();
    }

    const auto move = moves_[next_].move;
    ++next_;
    return move;
}

void MovePicker::add_move(const Position& position, Move move, MovePickerContext context,
                          std::size_t generation_index) noexcept {
    if (!move.is_valid() || size_ >= moves_.size()) {
        return;
    }

    moves_[size_] = ScoredMove{.move = move,
                               .score = move_order_score(position, move, context),
                               .generation_index = generation_index};
    ++size_;
}

void MovePicker::sort_moves() noexcept {
    std::sort(moves_.begin(), moves_.begin() + static_cast<std::ptrdiff_t>(size_),
              [](ScoredMove lhs, ScoredMove rhs) {
                  if (lhs.score != rhs.score) {
                      return lhs.score > rhs.score;
                  }
                  return lhs.generation_index < rhs.generation_index;
              });
}

} // namespace eng
