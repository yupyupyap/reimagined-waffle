#include <eng/see.hpp>

#include <eng/bitboard.hpp>

#include <algorithm>
#include <array>
#include <cstddef>

namespace eng {
namespace {

constexpr int see_max_exchanges = 32;
constexpr std::array<int, Piece::type_count> see_piece_values{100, 320, 330, 500, 900, 30000};
constexpr std::array<PieceType, Piece::type_count> least_valuable_piece_order{
    PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
    PieceType::Rook, PieceType::Queen,  PieceType::King,
};

struct LocalBoard {
    Position::PieceBitboards pieces{};
    Position::ColorOccupancies occupancies{};
    Bitboard occupancy{empty_bitboard};

    [[nodiscard]] static LocalBoard from_position(const Position& position) noexcept {
        LocalBoard board;
        for (int color = 0; color < 2; ++color) {
            const auto side = static_cast<Color>(color);
            for (int type = 0; type < Piece::type_count; ++type) {
                const auto piece_type = static_cast<PieceType>(type);
                const auto piece = Piece::make(side, piece_type);
                board.pieces[static_cast<std::size_t>(piece.index())] =
                    position.pieces(side, piece_type);
            }
            board.occupancies[static_cast<std::size_t>(color)] = position.occupancy(side);
        }
        board.occupancy = position.occupancy();
        return board;
    }

    [[nodiscard]] Bitboard pieces_for(Color color, PieceType type) const noexcept {
        if (!is_valid_color(color) || !is_valid_piece_type(type)) {
            return empty_bitboard;
        }
        return pieces[static_cast<std::size_t>(Piece::make(color, type).index())];
    }

    [[nodiscard]] Piece piece_on(Square square) const noexcept {
        if (!square.is_valid()) {
            return {};
        }

        const auto square_mask = square_bb(square);
        for (int color = 0; color < 2; ++color) {
            const auto side = static_cast<Color>(color);
            for (int type = 0; type < Piece::type_count; ++type) {
                const auto piece_type = static_cast<PieceType>(type);
                if ((pieces_for(side, piece_type) & square_mask) != empty_bitboard) {
                    return Piece::make(side, piece_type);
                }
            }
        }
        return {};
    }

    void remove_piece(Piece piece, Square square) noexcept {
        if (!piece.is_valid() || !square.is_valid()) {
            return;
        }

        const auto mask = square_bb(square);
        pieces[static_cast<std::size_t>(piece.index())] &= ~mask;
        occupancies[static_cast<std::size_t>(piece.color())] &= ~mask;
        occupancy &= ~mask;
    }

    void put_piece(Piece piece, Square square) noexcept {
        if (!piece.is_valid() || !square.is_valid()) {
            return;
        }

        const auto mask = square_bb(square);
        pieces[static_cast<std::size_t>(piece.index())] |= mask;
        occupancies[static_cast<std::size_t>(piece.color())] |= mask;
        occupancy |= mask;
    }
};

struct CaptureInfo {
    Piece moved_piece;
    Piece placed_piece;
    Piece captured_piece;
    Square captured_square;
    int score{0};
};

struct Attacker {
    Piece piece;
    Square square;
};

struct CaptureSquares {
    Square from;
    Square target;
};

[[nodiscard]] constexpr int piece_type_index(PieceType type) noexcept {
    return static_cast<int>(type);
}

[[nodiscard]] int piece_value(PieceType type) noexcept {
    if (!is_valid_piece_type(type)) {
        return 0;
    }
    return see_piece_values[static_cast<std::size_t>(piece_type_index(type))];
}

[[nodiscard]] Square square(int file, int rank) noexcept {
    return Square::from_file_rank(File::from_index(file), Rank::from_index(rank));
}

[[nodiscard]] Square en_passant_capture_square(Color capturing_side, Square target) noexcept {
    const auto rank_delta = capturing_side == Color::White ? -1 : 1;
    return square(target.file().index(), target.rank().index() + rank_delta);
}

[[nodiscard]] Bitboard attackers_to(const LocalBoard& board, Square target,
                                    Color attacker_color) noexcept {
    if (!target.is_valid() || !is_valid_color(attacker_color)) {
        return empty_bitboard;
    }

    const auto pawns = pawn_attacks(opposite(attacker_color), target) &
                       board.pieces_for(attacker_color, PieceType::Pawn);
    const auto knights =
        knight_attacks(target) & board.pieces_for(attacker_color, PieceType::Knight);
    const auto bishops_and_queens = bishop_attacks(target, board.occupancy) &
                                    (board.pieces_for(attacker_color, PieceType::Bishop) |
                                     board.pieces_for(attacker_color, PieceType::Queen));
    const auto rooks_and_queens = rook_attacks(target, board.occupancy) &
                                  (board.pieces_for(attacker_color, PieceType::Rook) |
                                   board.pieces_for(attacker_color, PieceType::Queen));
    const auto kings = king_attacks(target) & board.pieces_for(attacker_color, PieceType::King);

    return pawns | knights | bishops_and_queens | rooks_and_queens | kings;
}

[[nodiscard]] bool is_square_attacked(const LocalBoard& board, Square target,
                                      Color attacker_color) noexcept {
    return attackers_to(board, target, attacker_color) != empty_bitboard;
}

[[nodiscard]] Square king_square(const LocalBoard& board, Color color) noexcept {
    return lsb(board.pieces_for(color, PieceType::King));
}

void apply_capture(LocalBoard& board, CaptureSquares squares, Piece placed_piece) noexcept {
    const auto captured_piece = board.piece_on(squares.target);
    const auto moved_piece = board.piece_on(squares.from);
    board.remove_piece(captured_piece, squares.target);
    board.remove_piece(moved_piece, squares.from);
    board.put_piece(placed_piece, squares.target);
}

[[nodiscard]] bool capture_is_legalish(const LocalBoard& board, CaptureSquares squares,
                                       Color side_to_move) noexcept {
    const auto piece = board.piece_on(squares.from);
    const auto captured = board.piece_on(squares.target);
    if (!piece.is_valid() || piece.color() != side_to_move || !captured.is_valid() ||
        captured.color() != opposite(side_to_move) || captured.type() == PieceType::King) {
        return false;
    }

    auto trial = board;
    apply_capture(trial, squares, piece);
    if (piece.type() == PieceType::King) {
        return !is_square_attacked(trial, squares.target, opposite(side_to_move));
    }

    const auto own_king = king_square(trial, side_to_move);
    return !own_king.is_valid() || !is_square_attacked(trial, own_king, opposite(side_to_move));
}

[[nodiscard]] Attacker least_valuable_attacker(const LocalBoard& board, Square target,
                                               Color side) noexcept {
    for (const auto type : least_valuable_piece_order) {
        auto candidates = attackers_to(board, target, side) & board.pieces_for(side, type);
        while (candidates != empty_bitboard) {
            const auto from = pop_lsb(candidates);
            if (capture_is_legalish(board, {.from = from, .target = target}, side)) {
                return {.piece = Piece::make(side, type), .square = from};
            }
        }
    }
    return {};
}

[[nodiscard]] CaptureInfo describe_capture(const Position& position, Move move) noexcept {
    if (!move.is_valid() || move.is_null() || move.is_castling() || !move.from().is_valid() ||
        !move.to().is_valid() || move.from() == move.to()) {
        return {};
    }

    const auto moved_piece = position.piece_on(move.from());
    if (!moved_piece.is_valid() || moved_piece.color() != position.side_to_move()) {
        return {};
    }

    CaptureInfo info{.moved_piece = moved_piece,
                     .placed_piece = moved_piece,
                     .captured_piece = position.piece_on(move.to()),
                     .captured_square = move.to()};

    if (move.is_en_passant()) {
        if (moved_piece.type() != PieceType::Pawn || move.to() != position.en_passant_square() ||
            position.piece_on(move.to()).is_valid()) {
            return {};
        }
        info.captured_square = en_passant_capture_square(moved_piece.color(), move.to());
        info.captured_piece = position.piece_on(info.captured_square);
    }

    if (!info.captured_piece.is_valid() ||
        info.captured_piece.color() != opposite(moved_piece.color()) ||
        info.captured_piece.type() == PieceType::King) {
        return {};
    }

    info.score = piece_value(info.captured_piece.type());
    if (move.is_promotion()) {
        if (moved_piece.type() != PieceType::Pawn ||
            !is_valid_promotion_piece(move.promotion_piece())) {
            return {};
        }
        info.placed_piece = Piece::make(moved_piece.color(), move.promotion_piece());
        info.score += piece_value(move.promotion_piece()) - piece_value(PieceType::Pawn);
    }

    return info;
}

void apply_initial_capture(LocalBoard& board, Move move, const CaptureInfo& info) noexcept {
    board.remove_piece(info.captured_piece, info.captured_square);
    board.remove_piece(info.moved_piece, move.from());
    board.put_piece(info.placed_piece, move.to());
}

} // namespace

Score static_exchange_score(const Position& position, Move move) noexcept {
    const auto info = describe_capture(position, move);
    if (!info.captured_piece.is_valid()) {
        return Score::centipawns(0);
    }

    auto board = LocalBoard::from_position(position);
    apply_initial_capture(board, move, info);

    std::array<int, see_max_exchanges> gains{};
    gains[0] = info.score;

    auto side = opposite(info.moved_piece.color());
    auto victim_type = info.placed_piece.type();
    int depth = 0;

    while (depth + 1 < see_max_exchanges) {
        const auto attacker = least_valuable_attacker(board, move.to(), side);
        if (!attacker.piece.is_valid()) {
            break;
        }

        ++depth;
        gains[static_cast<std::size_t>(depth)] =
            piece_value(victim_type) - gains[static_cast<std::size_t>(depth - 1)];
        apply_capture(board, {.from = attacker.square, .target = move.to()}, attacker.piece);
        victim_type = attacker.piece.type();
        side = opposite(side);
    }

    while (depth > 0) {
        const auto current = static_cast<std::size_t>(depth);
        const auto previous = static_cast<std::size_t>(depth - 1);
        gains[previous] = -std::max(-gains[previous], gains[current]);
        --depth;
    }

    return Score::centipawns(gains[0]);
}

bool static_exchange_ge(const Position& position, Move move, Score threshold) noexcept {
    return static_exchange_score(position, move).value() >= threshold.value();
}

} // namespace eng
