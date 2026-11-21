#include <eng/movegen.hpp>

#include <array>
#include <cstddef>

// Square names such as e1 and h8 are clearer than expanded identifiers in chess code.
// NOLINTBEGIN(readability-identifier-length)

namespace eng {
namespace {

constexpr int file_a = 0;
constexpr int file_b = 1;
constexpr int file_c = 2;
constexpr int file_d = 3;
constexpr int file_e = 4;
constexpr int file_f = 5;
constexpr int file_g = 6;
constexpr int file_h = 7;
constexpr int rank_1 = 0;
constexpr int rank_2 = 1;
constexpr int rank_4 = 3;
constexpr int rank_5 = 4;
constexpr int rank_7 = 6;
constexpr int rank_8 = 7;

inline constexpr std::array<PieceType, 4> promotion_pieces{
    PieceType::Knight,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Queen,
};

[[nodiscard]] constexpr Square square(int file_index, int rank_index) noexcept {
    return Square::from_file_rank(File::from_index(file_index), Rank::from_index(rank_index));
}

[[nodiscard]] constexpr int forward_delta(Color color) noexcept {
    return color == Color::White ? 1 : -1;
}

[[nodiscard]] constexpr int start_rank(Color color) noexcept {
    return color == Color::White ? rank_2 : rank_7;
}

[[nodiscard]] constexpr int promotion_from_rank(Color color) noexcept {
    return color == Color::White ? rank_7 : rank_2;
}

[[nodiscard]] constexpr int en_passant_captured_rank(Color color) noexcept {
    return color == Color::White ? rank_5 : rank_4;
}

[[nodiscard]] constexpr Square offset(Square from_square, int file_delta, int rank_delta) noexcept {
    return bitboard_detail::offset_square(from_square, file_delta, rank_delta);
}

[[nodiscard]] Square king_square(const Position& position, Color color) noexcept {
    return lsb(position.pieces(color, PieceType::King));
}

[[nodiscard]] bool is_enemy_piece(const Position& position, Square target, Color us) noexcept {
    const auto piece = position.piece_on(target);
    return piece.is_valid() && piece.color() == opposite(us);
}

[[nodiscard]] Bitboard attackers_to_with_occupancy(const Position& position, Square target,
                                                   Color attacker_color,
                                                   Bitboard occupancy) noexcept {
    if (!target.is_valid() || !is_valid_color(attacker_color)) {
        return empty_bitboard;
    }

    const auto pawns = pawn_attacks(opposite(attacker_color), target) &
                       position.pieces(attacker_color, PieceType::Pawn);
    const auto knights =
        knight_attacks(target) & position.pieces(attacker_color, PieceType::Knight);
    const auto bishops_and_queens =
        (bishop_attacks(target, occupancy) & (position.pieces(attacker_color, PieceType::Bishop) |
                                              position.pieces(attacker_color, PieceType::Queen)));
    const auto rooks_and_queens =
        (rook_attacks(target, occupancy) & (position.pieces(attacker_color, PieceType::Rook) |
                                            position.pieces(attacker_color, PieceType::Queen)));
    const auto kings = king_attacks(target) & position.pieces(attacker_color, PieceType::King);

    return pawns | knights | bishops_and_queens | rooks_and_queens | kings;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] bool is_king_destination_safe(const Position& position, Square from_square,
                                            Square target, Color us) noexcept {
    const auto occupancy_without_king = position.occupancy() & ~square_bb(from_square);
    return attackers_to_with_occupancy(position, target, opposite(us), occupancy_without_king) ==
           empty_bitboard;
}

[[nodiscard]] bool should_include_captures(MoveGenStage stage) noexcept {
    return stage == MoveGenStage::All || stage == MoveGenStage::Captures;
}

[[nodiscard]] bool should_include_quiets(MoveGenStage stage) noexcept {
    return stage == MoveGenStage::All || stage == MoveGenStage::Quiets;
}

} // namespace

class MoveListBuilder {
  public:
    explicit MoveListBuilder(MoveList& moves) noexcept : moves_{&moves} {}

    void push(Move move) noexcept { static_cast<void>(moves_->push(move)); }

    void normal(Square from_square, Square to_square) noexcept {
        static_cast<void>(moves_->push(Move::normal(from_square, to_square)));
    }

    void en_passant(Square from_square, Square to_square) noexcept {
        static_cast<void>(moves_->push(Move::en_passant(from_square, to_square)));
    }

    void castling(Square from_square, Square to_square) noexcept {
        static_cast<void>(moves_->push(Move::castling(from_square, to_square)));
    }

    void promotions(Square from_square, Square to_square) noexcept {
        for (const auto promotion_piece : promotion_pieces) {
            static_cast<void>(
                moves_->push(Move::promotion(from_square, to_square, promotion_piece)));
        }
    }

  private:
    MoveList* moves_;
};

namespace {

void add_target_moves(MoveListBuilder& builder, Square from_square, Bitboard targets) noexcept {
    while (targets != empty_bitboard) {
        builder.normal(from_square, pop_lsb(targets));
    }
}

void add_pawn_move_or_promotions(MoveListBuilder& builder, Color us, Square from,
                                 Square target) noexcept {
    if (from.rank().index() == promotion_from_rank(us)) {
        builder.promotions(from, target);
        return;
    }

    builder.normal(from, target);
}

void generate_pawn_quiets_for_pawn(const Position& position, MoveListBuilder& builder, Color us,
                                   Square from, Bitboard target_filter) noexcept {
    const auto forward = forward_delta(us);
    const auto one_ahead = offset(from, 0, forward);
    if (!one_ahead.is_valid() || position.piece_on(one_ahead).is_valid()) {
        return;
    }

    if (has_square(target_filter, one_ahead)) {
        add_pawn_move_or_promotions(builder, us, from, one_ahead);
    }

    if (from.rank().index() == promotion_from_rank(us)) {
        return;
    }

    const auto two_ahead = offset(one_ahead, 0, forward);
    if (from.rank().index() == start_rank(us) && two_ahead.is_valid() &&
        !position.piece_on(two_ahead).is_valid() && has_square(target_filter, two_ahead)) {
        builder.normal(from, two_ahead);
    }
}

void generate_pawn_capture_target(const Position& position, MoveListBuilder& builder, Color us,
                                  Square from, Square target, Bitboard target_filter) noexcept {
    const auto them = opposite(us);
    if (!target.is_valid()) {
        return;
    }

    if (is_enemy_piece(position, target, us) && has_square(target_filter, target)) {
        add_pawn_move_or_promotions(builder, us, from, target);
    }

    if (target == position.en_passant_square() && has_square(target_filter, target)) {
        const auto captured_square = square(target.file().index(), en_passant_captured_rank(us));
        if (position.piece_on(captured_square) == Piece::make(them, PieceType::Pawn)) {
            builder.en_passant(from, target);
        }
    }
}

void generate_pawn_captures_for_pawn(const Position& position, MoveListBuilder& builder, Color us,
                                     Square from, Bitboard target_filter) noexcept {
    const auto forward = forward_delta(us);
    generate_pawn_capture_target(position, builder, us, from, offset(from, -1, forward),
                                 target_filter);
    generate_pawn_capture_target(position, builder, us, from, offset(from, 1, forward),
                                 target_filter);
}

void generate_pawn_moves(const Position& position, MoveListBuilder& builder, Color us,
                         MoveGenStage stage, Bitboard target_filter) noexcept {
    const auto include_captures = should_include_captures(stage);
    const auto include_quiets = should_include_quiets(stage);
    auto pawns = position.pieces(us, PieceType::Pawn);

    while (pawns != empty_bitboard) {
        const auto from = pop_lsb(pawns);
        if (include_quiets) {
            generate_pawn_quiets_for_pawn(position, builder, us, from, target_filter);
        }
        if (include_captures) {
            generate_pawn_captures_for_pawn(position, builder, us, from, target_filter);
        }
    }
}

void generate_leaper_moves(const Position& position, MoveListBuilder& builder, Color us,
                           PieceType type, MoveGenStage stage, Bitboard target_filter) noexcept {
    const auto own = position.occupancy(us);
    const auto enemy = position.occupancy(opposite(us));
    auto pieces = position.pieces(us, type);

    while (pieces != empty_bitboard) {
        const auto from = pop_lsb(pieces);
        const auto attacks = type == PieceType::Knight ? knight_attacks(from) : king_attacks(from);
        const auto targets = attacks & ~own & target_filter;

        if (should_include_captures(stage)) {
            add_target_moves(builder, from, targets & enemy);
        }
        if (should_include_quiets(stage)) {
            add_target_moves(builder, from, targets & ~position.occupancy());
        }
    }
}

void generate_slider_moves(const Position& position, MoveListBuilder& builder, Color us,
                           PieceType type, MoveGenStage stage, Bitboard target_filter) noexcept {
    const auto own = position.occupancy(us);
    const auto enemy = position.occupancy(opposite(us));
    auto pieces = position.pieces(us, type);

    while (pieces != empty_bitboard) {
        const auto from = pop_lsb(pieces);
        Bitboard attacks = empty_bitboard;
        if (type == PieceType::Bishop) {
            attacks = bishop_attacks(from, position.occupancy());
        } else if (type == PieceType::Rook) {
            attacks = rook_attacks(from, position.occupancy());
        } else {
            attacks = queen_attacks(from, position.occupancy());
        }

        const auto targets = attacks & ~own & target_filter;
        if (should_include_captures(stage)) {
            add_target_moves(builder, from, targets & enemy);
        }
        if (should_include_quiets(stage)) {
            add_target_moves(builder, from, targets & ~position.occupancy());
        }
    }
}

void generate_castling_candidates(const Position& position, MoveListBuilder& builder,
                                  Color us) noexcept {
    const auto king_from = square(file_e, us == Color::White ? rank_1 : rank_8);
    if (position.piece_on(king_from) != Piece::make(us, PieceType::King)) {
        return;
    }

    const auto rights = position.castling_rights();
    const auto rank = us == Color::White ? rank_1 : rank_8;

    if (rights.has(us, CastlingSide::KingSide) &&
        position.piece_on(square(file_h, rank)) == Piece::make(us, PieceType::Rook) &&
        !position.piece_on(square(file_f, rank)).is_valid() &&
        !position.piece_on(square(file_g, rank)).is_valid()) {
        builder.castling(king_from, square(file_g, rank));
    }

    if (rights.has(us, CastlingSide::QueenSide) &&
        position.piece_on(square(file_a, rank)) == Piece::make(us, PieceType::Rook) &&
        !position.piece_on(square(file_b, rank)).is_valid() &&
        !position.piece_on(square(file_c, rank)).is_valid() &&
        !position.piece_on(square(file_d, rank)).is_valid()) {
        builder.castling(king_from, square(file_c, rank));
    }
}

void generate_standard_moves(const Position& position, MoveListBuilder& builder, MoveGenStage stage,
                             Bitboard target_filter) noexcept {
    const auto us = position.side_to_move();
    generate_pawn_moves(position, builder, us, stage, target_filter);
    generate_leaper_moves(position, builder, us, PieceType::Knight, stage, target_filter);
    generate_slider_moves(position, builder, us, PieceType::Bishop, stage, target_filter);
    generate_slider_moves(position, builder, us, PieceType::Rook, stage, target_filter);
    generate_slider_moves(position, builder, us, PieceType::Queen, stage, target_filter);
    generate_leaper_moves(position, builder, us, PieceType::King, stage, target_filter);

    if (should_include_quiets(stage) && target_filter == full_bitboard) {
        generate_castling_candidates(position, builder, us);
    }
}

void generate_safe_king_moves(const Position& position, MoveListBuilder& builder, Color us,
                              Square from) noexcept {
    auto targets = king_attacks(from) & ~position.occupancy(us);
    while (targets != empty_bitboard) {
        const auto target = pop_lsb(targets);
        if (is_king_destination_safe(position, from, target, us)) {
            builder.normal(from, target);
        }
    }
}

void generate_en_passant_evasions(const Position& position, MoveListBuilder& builder, Color us,
                                  Square checker, Bitboard evasion_targets) noexcept {
    const auto target = position.en_passant_square();
    if (!target.is_valid()) {
        return;
    }

    const auto captured_square = square(target.file().index(), en_passant_captured_rank(us));
    if (captured_square != checker || has_square(evasion_targets, target)) {
        return;
    }

    auto pawns = position.pieces(us, PieceType::Pawn) & pawn_attacks(opposite(us), target);
    while (pawns != empty_bitboard) {
        const auto from = pop_lsb(pawns);
        builder.en_passant(from, target);
    }
}

} // namespace

MoveList generate_moves(const Position& position, MoveGenStage stage) {
    if (stage == MoveGenStage::Evasions) {
        return generate_evasions(position);
    }

    MoveList moves;
    MoveListBuilder builder{moves};
    generate_standard_moves(position, builder, stage, full_bitboard);
    return moves;
}

MoveList generate_moves(const Position& position) {
    return side_to_move_in_check(position) ? generate_evasions(position)
                                           : generate_moves(position, MoveGenStage::All);
}

MoveList generate_captures(const Position& position) {
    return generate_moves(position, MoveGenStage::Captures);
}

MoveList generate_quiets(const Position& position) {
    return generate_moves(position, MoveGenStage::Quiets);
}

MoveList generate_evasions(const Position& position) {
    MoveList moves;
    MoveListBuilder builder{moves};

    const auto us = position.side_to_move();
    const auto king = king_square(position, us);
    if (!king.is_valid()) {
        return moves;
    }

    auto checkers = attackers_to(position, king, opposite(us));
    const auto checker_count = popcount(checkers);
    if (checker_count == 0) {
        return moves;
    }

    generate_safe_king_moves(position, builder, us, king);
    if (checker_count > 1) {
        return moves;
    }

    const auto checker = lsb(checkers);
    const auto checker_piece = position.piece_on(checker);
    auto evasion_targets = square_bb(checker);
    if (checker_piece.type() == PieceType::Bishop || checker_piece.type() == PieceType::Rook ||
        checker_piece.type() == PieceType::Queen) {
        evasion_targets |= between(king, checker);
    }

    generate_pawn_moves(position, builder, us, MoveGenStage::All, evasion_targets);
    generate_leaper_moves(position, builder, us, PieceType::Knight, MoveGenStage::All,
                          evasion_targets);
    generate_slider_moves(position, builder, us, PieceType::Bishop, MoveGenStage::All,
                          evasion_targets);
    generate_slider_moves(position, builder, us, PieceType::Rook, MoveGenStage::All,
                          evasion_targets);
    generate_slider_moves(position, builder, us, PieceType::Queen, MoveGenStage::All,
                          evasion_targets);
    generate_en_passant_evasions(position, builder, us, checker, evasion_targets);

    return moves;
}

MoveList generate_legal_moves(const Position& position) {
    const auto pseudo_legal_moves = generate_moves(position);
    MoveList legal_moves;
    MoveListBuilder builder{legal_moves};

    for (const auto move : pseudo_legal_moves) {
        auto candidate = position;
        StateInfo state;
        if (candidate.make_move(move, state)) {
            builder.push(move);
        }
    }

    return legal_moves;
}

bool has_legal_moves(const Position& position) {
    const auto pseudo_legal_moves = generate_moves(position);
    for (const auto move : pseudo_legal_moves) {
        auto candidate = position;
        StateInfo state;
        if (candidate.make_move(move, state)) {
            return true;
        }
    }

    return false;
}

bool is_checkmate(const Position& position) {
    return side_to_move_in_check(position) && !has_legal_moves(position);
}

bool is_stalemate(const Position& position) {
    return !side_to_move_in_check(position) && !has_legal_moves(position);
}

Bitboard attackers_to(const Position& position, Square target, Color attacker_color) noexcept {
    return attackers_to(position, target, attacker_color, position.occupancy());
}

Bitboard attackers_to(const Position& position, Square target, Color attacker_color,
                      Bitboard occupancy) noexcept {
    return attackers_to_with_occupancy(position, target, attacker_color, occupancy);
}

bool is_square_attacked(const Position& position, Square target, Color attacker_color) noexcept {
    return attackers_to(position, target, attacker_color) != empty_bitboard;
}

bool is_square_attacked(const Position& position, Square target, Color attacker_color,
                        Bitboard occupancy) noexcept {
    return attackers_to(position, target, attacker_color, occupancy) != empty_bitboard;
}

bool in_check(const Position& position, Color color) noexcept {
    const auto king = king_square(position, color);
    return king.is_valid() && is_square_attacked(position, king, opposite(color));
}

bool side_to_move_in_check(const Position& position) noexcept {
    return in_check(position, position.side_to_move());
}

} // namespace eng

// NOLINTEND(readability-identifier-length)
