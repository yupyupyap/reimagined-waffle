#include <eng/position.hpp>

#include <eng/bitboard.hpp>
#include <eng/movegen.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

// Square names such as e1 and a8 are clearer than expanded identifiers in chess code.
// NOLINTBEGIN(readability-identifier-length)

namespace eng {
namespace {

constexpr int fen_field_count = 6;
constexpr int piece_placement_field = 0;
constexpr int side_to_move_field = 1;
constexpr int castling_field = 2;
constexpr int en_passant_field = 3;
constexpr int halfmove_field = 4;
constexpr int fullmove_field = 5;
constexpr int decimal_base = 10;
constexpr int max_side_piece_count = 16;
constexpr std::size_t fen_reserve_size = 80;
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
constexpr int rank_3 = 2;
constexpr int rank_4 = 3;
constexpr int rank_5 = 4;
constexpr int rank_7 = 6;
constexpr int rank_8 = 7;
constexpr std::string_view castling_order = "KQkq";
constexpr std::uint64_t zobrist_seed = 0x4C6F63616C436831ULL;
constexpr std::uint64_t splitmix_increment = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t splitmix_multiplier_a = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t splitmix_multiplier_b = 0x94D049BB133111EBULL;
constexpr unsigned splitmix_shift_a = 30U;
constexpr unsigned splitmix_shift_b = 27U;
constexpr unsigned splitmix_shift_c = 31U;

using FenFields = std::array<std::string_view, fen_field_count>;

class SplitMix64 {
  public:
    constexpr explicit SplitMix64(std::uint64_t seed) noexcept : state_{seed} {}

    [[nodiscard]] constexpr std::uint64_t next() noexcept {
        state_ += splitmix_increment;
        auto value = state_;
        value = (value ^ (value >> splitmix_shift_a)) * splitmix_multiplier_a;
        value = (value ^ (value >> splitmix_shift_b)) * splitmix_multiplier_b;
        return value ^ (value >> splitmix_shift_c);
    }

  private:
    std::uint64_t state_;
};

struct ZobristTables {
    ZobristTables() noexcept {
        auto random = SplitMix64{zobrist_seed};
        for (auto& piece_keys : pieces) {
            for (auto& key : piece_keys) {
                key = random.next();
            }
        }
        side_to_move = random.next();
        for (auto& key : castling) {
            key = random.next();
        }
        for (auto& key : en_passant_file) {
            key = random.next();
        }
    }

    std::array<std::array<ZobristKey, Square::count>, Piece::count> pieces{};
    ZobristKey side_to_move{0};
    std::array<ZobristKey, CastlingRights::all_mask + 1> castling{};
    std::array<ZobristKey, File::count> en_passant_file{};
};

[[nodiscard]] const ZobristTables& zobrist_tables() noexcept {
    static const ZobristTables tables{};
    return tables;
}

[[nodiscard]] ZobristKey piece_zobrist(Piece piece, Square square_value) noexcept {
    if (!piece.is_valid() || !square_value.is_valid()) {
        return 0;
    }
    const auto& tables = zobrist_tables();
    return tables.pieces[static_cast<std::size_t>(piece.index())]
                        [static_cast<std::size_t>(square_value.index())];
}

[[nodiscard]] ZobristKey castling_zobrist(CastlingRights rights) noexcept {
    return zobrist_tables().castling[rights.raw()];
}

[[nodiscard]] ZobristKey en_passant_zobrist(Square square_value) noexcept {
    if (!square_value.is_valid()) {
        return 0;
    }
    return zobrist_tables().en_passant_file[static_cast<std::size_t>(square_value.file().index())];
}

[[nodiscard]] ZobristKey side_to_move_zobrist() noexcept { return zobrist_tables().side_to_move; }

[[nodiscard]] FenError make_error(FenErrorCode code, std::string message) {
    return FenError{.code = code, .message = std::move(message)};
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] constexpr Square square(int file_index, int rank_index) noexcept {
    return Square::from_file_rank(File::from_index(file_index), Rank::from_index(rank_index));
}

[[nodiscard]] constexpr char file_char(File file) noexcept {
    return static_cast<char>('a' + file.index());
}

[[nodiscard]] constexpr char rank_char(Rank rank) noexcept {
    return static_cast<char>('1' + rank.index());
}

[[nodiscard]] std::optional<Piece> piece_from_fen_char(char symbol) noexcept {
    switch (symbol) {
    case 'P':
        return Piece::make(Color::White, PieceType::Pawn);
    case 'N':
        return Piece::make(Color::White, PieceType::Knight);
    case 'B':
        return Piece::make(Color::White, PieceType::Bishop);
    case 'R':
        return Piece::make(Color::White, PieceType::Rook);
    case 'Q':
        return Piece::make(Color::White, PieceType::Queen);
    case 'K':
        return Piece::make(Color::White, PieceType::King);
    case 'p':
        return Piece::make(Color::Black, PieceType::Pawn);
    case 'n':
        return Piece::make(Color::Black, PieceType::Knight);
    case 'b':
        return Piece::make(Color::Black, PieceType::Bishop);
    case 'r':
        return Piece::make(Color::Black, PieceType::Rook);
    case 'q':
        return Piece::make(Color::Black, PieceType::Queen);
    case 'k':
        return Piece::make(Color::Black, PieceType::King);
    default:
        return std::nullopt;
    }
}

[[nodiscard]] char piece_to_fen_char(Piece piece) noexcept {
    if (piece.color() == Color::White) {
        switch (piece.type()) {
        case PieceType::Pawn:
            return 'P';
        case PieceType::Knight:
            return 'N';
        case PieceType::Bishop:
            return 'B';
        case PieceType::Rook:
            return 'R';
        case PieceType::Queen:
            return 'Q';
        case PieceType::King:
            return 'K';
        case PieceType::None:
            break;
        }
    }
    if (piece.color() == Color::Black) {
        switch (piece.type()) {
        case PieceType::Pawn:
            return 'p';
        case PieceType::Knight:
            return 'n';
        case PieceType::Bishop:
            return 'b';
        case PieceType::Rook:
            return 'r';
        case PieceType::Queen:
            return 'q';
        case PieceType::King:
            return 'k';
        case PieceType::None:
            break;
        }
    }
    return '\0';
}

[[nodiscard]] bool is_digit(char value) noexcept { return value >= '0' && value <= '9'; }

[[nodiscard]] std::expected<FenFields, FenError> split_fen_fields(std::string_view fen) {
    FenFields fields{};
    std::size_t field_index = 0;
    std::size_t start = 0;

    while (start <= fen.size()) {
        if (field_index >= fields.size()) {
            return std::unexpected(
                make_error(FenErrorCode::TooManyFields, "FEN has more than six fields"));
        }

        const auto separator = fen.find(' ', start);
        const auto end = separator == std::string_view::npos ? fen.size() : separator;
        if (end == start) {
            return std::unexpected(
                make_error(FenErrorCode::MissingField, "FEN contains an empty field"));
        }

        fields[field_index] = fen.substr(start, end - start);
        ++field_index;

        if (separator == std::string_view::npos) {
            break;
        }
        start = separator + 1U;
    }

    if (field_index < fields.size()) {
        return std::unexpected(
            make_error(FenErrorCode::MissingField, "FEN has fewer than six fields"));
    }

    return fields;
}

[[nodiscard]] std::expected<int, FenError>
parse_non_negative_int(std::string_view field, FenErrorCode error_code, std::string_view name) {
    if (field.empty()) {
        return std::unexpected(make_error(error_code, std::string{name} + " is empty"));
    }

    int value = 0;
    for (const auto symbol : field) {
        if (!is_digit(symbol)) {
            return std::unexpected(
                make_error(error_code, std::string{name} + " contains a non-digit"));
        }

        const auto digit = symbol - '0';
        if (value > (std::numeric_limits<int>::max() - digit) / decimal_base) {
            return std::unexpected(make_error(error_code, std::string{name} + " overflows int"));
        }
        value = (value * decimal_base) + digit;
    }

    return value;
}

[[nodiscard]] bool has_piece(const Position& position, Color color, PieceType type,
                             Square piece_square) noexcept {
    return position.piece_on(piece_square) == Piece::make(color, type);
}

[[nodiscard]] constexpr int abs_value(int value) noexcept { return value < 0 ? -value : value; }

[[nodiscard]] constexpr int forward_delta(Color color) noexcept {
    return color == Color::White ? 1 : -1;
}

[[nodiscard]] constexpr int start_rank(Color color) noexcept {
    return color == Color::White ? rank_2 : rank_7;
}

[[nodiscard]] constexpr int promotion_from_rank(Color color) noexcept {
    return color == Color::White ? rank_7 : rank_2;
}

[[nodiscard]] constexpr int promotion_to_rank(Color color) noexcept {
    return color == Color::White ? rank_8 : rank_1;
}

[[nodiscard]] constexpr int en_passant_captured_rank(Color color) noexcept {
    return color == Color::White ? rank_5 : rank_4;
}

[[nodiscard]] constexpr Square offset(Square from_square, int file_delta, int rank_delta) noexcept {
    return bitboard_detail::offset_square(from_square, file_delta, rank_delta);
}

[[nodiscard]] bool is_enemy_piece(Piece piece, Color us) noexcept {
    return piece.is_valid() && piece.color() == opposite(us) && piece.type() != PieceType::King;
}

[[nodiscard]] bool move_is_pawn_double_push(Color color, Square from, Square to) noexcept {
    return from.file() == to.file() && from.rank().index() == start_rank(color) &&
           to.rank().index() == start_rank(color) + (2 * forward_delta(color));
}

[[nodiscard]] Square en_passant_target_after_double_push(Color color, Square from) noexcept {
    return offset(from, 0, forward_delta(color));
}

void remove_rook_castling_right(CastlingRights& rights, Square rook_square) noexcept {
    if (rook_square == square(file_h, rank_1)) {
        rights.remove(Color::White, CastlingSide::KingSide);
    } else if (rook_square == square(file_a, rank_1)) {
        rights.remove(Color::White, CastlingSide::QueenSide);
    } else if (rook_square == square(file_h, rank_8)) {
        rights.remove(Color::Black, CastlingSide::KingSide);
    } else if (rook_square == square(file_a, rank_8)) {
        rights.remove(Color::Black, CastlingSide::QueenSide);
    }
}

[[nodiscard]] CastlingRights castling_rights_after_move(CastlingRights rights, Piece moved_piece,
                                                        Square from, Piece captured_piece,
                                                        Square captured_square) noexcept {
    if (moved_piece.type() == PieceType::King) {
        rights.remove(moved_piece.color());
    } else if (moved_piece.type() == PieceType::Rook) {
        remove_rook_castling_right(rights, from);
    }

    if (captured_piece.type() == PieceType::Rook) {
        remove_rook_castling_right(rights, captured_square);
    }

    return rights;
}

struct CastlingRookMove {
    Square from;
    Square to;
};

[[nodiscard]] CastlingRookMove castling_rook_move(Color color, Square king_to) noexcept {
    const auto rank = color == Color::White ? rank_1 : rank_8;
    if (king_to == square(file_g, rank)) {
        return {.from = square(file_h, rank), .to = square(file_f, rank)};
    }
    if (king_to == square(file_c, rank)) {
        return {.from = square(file_a, rank), .to = square(file_d, rank)};
    }
    return {};
}

[[nodiscard]] Square castling_transit_square(Color color, Square king_to) noexcept {
    const auto rank = color == Color::White ? rank_1 : rank_8;
    if (king_to == square(file_g, rank)) {
        return square(file_f, rank);
    }
    if (king_to == square(file_c, rank)) {
        return square(file_d, rank);
    }
    return {};
}

[[nodiscard]] bool validate_pawn_normal_move(const Position& position, Move move, Color us,
                                             StateInfo& state) noexcept {
    const auto from = move.from();
    const auto to = move.to();
    const auto target_piece = position.piece_on(to);
    const auto file_delta = to.file().index() - from.file().index();
    const auto rank_delta = to.rank().index() - from.rank().index();
    const auto forward = forward_delta(us);

    if (to.rank().index() == promotion_to_rank(us)) {
        return false;
    }

    if (file_delta == 0) {
        if (target_piece.is_valid()) {
            return false;
        }
        if (rank_delta == forward) {
            return true;
        }
        if (rank_delta == (2 * forward) && from.rank().index() == start_rank(us)) {
            const auto intermediate = offset(from, 0, forward);
            return intermediate.is_valid() && !position.piece_on(intermediate).is_valid();
        }
        return false;
    }

    if (abs_value(file_delta) == 1 && rank_delta == forward && is_enemy_piece(target_piece, us)) {
        state.captured_piece = target_piece;
        state.captured_square = to;
        return true;
    }

    return false;
}

[[nodiscard]] bool validate_pawn_promotion_move(const Position& position, Move move, Color us,
                                                StateInfo& state) noexcept {
    const auto from = move.from();
    const auto to = move.to();
    if (!is_valid_promotion_piece(move.promotion_piece()) ||
        from.rank().index() != promotion_from_rank(us) ||
        to.rank().index() != promotion_to_rank(us)) {
        return false;
    }

    const auto target_piece = position.piece_on(to);
    const auto file_delta = to.file().index() - from.file().index();
    const auto rank_delta = to.rank().index() - from.rank().index();
    if (rank_delta != forward_delta(us)) {
        return false;
    }

    if (file_delta == 0) {
        return !target_piece.is_valid();
    }

    if (abs_value(file_delta) == 1 && is_enemy_piece(target_piece, us)) {
        state.captured_piece = target_piece;
        state.captured_square = to;
        return true;
    }

    return false;
}

[[nodiscard]] bool validate_en_passant_move(const Position& position, Move move, Piece moved_piece,
                                            StateInfo& state) noexcept {
    const auto us = moved_piece.color();
    const auto from = move.from();
    const auto to = move.to();
    if (moved_piece.type() != PieceType::Pawn || to != position.en_passant_square() ||
        position.piece_on(to).is_valid()) {
        return false;
    }

    const auto file_delta = to.file().index() - from.file().index();
    const auto rank_delta = to.rank().index() - from.rank().index();
    if (abs_value(file_delta) != 1 || rank_delta != forward_delta(us)) {
        return false;
    }

    const auto captured_square = square(to.file().index(), en_passant_captured_rank(us));
    const auto captured_piece = position.piece_on(captured_square);
    if (captured_piece != Piece::make(opposite(us), PieceType::Pawn)) {
        return false;
    }

    state.captured_piece = captured_piece;
    state.captured_square = captured_square;
    return true;
}

[[nodiscard]] bool validate_castling_move(const Position& position, Move move,
                                          Piece moved_piece) noexcept {
    const auto us = moved_piece.color();
    const auto them = opposite(us);
    const auto rank = us == Color::White ? rank_1 : rank_8;
    const auto from = move.from();
    const auto to = move.to();
    const auto king_from = square(file_e, rank);
    const auto rook_move = castling_rook_move(us, to);
    const auto transit = castling_transit_square(us, to);

    if (moved_piece.type() != PieceType::King || from != king_from || !rook_move.from.is_valid() ||
        !rook_move.to.is_valid() || !transit.is_valid() || position.piece_on(to).is_valid() ||
        position.piece_on(rook_move.to).is_valid() ||
        position.piece_on(rook_move.from) != Piece::make(us, PieceType::Rook) ||
        !position.castling_rights().has(
            us, to.file().index() == file_g ? CastlingSide::KingSide : CastlingSide::QueenSide)) {
        return false;
    }

    if (to.file().index() == file_c && (position.piece_on(square(file_b, rank)).is_valid() ||
                                        position.piece_on(square(file_c, rank)).is_valid() ||
                                        position.piece_on(square(file_d, rank)).is_valid())) {
        return false;
    }
    if (to.file().index() == file_g && (position.piece_on(square(file_f, rank)).is_valid() ||
                                        position.piece_on(square(file_g, rank)).is_valid())) {
        return false;
    }

    const auto occupancy_without_king = position.occupancy() & ~square_bb(from);
    return !is_square_attacked(position, from, them) &&
           !is_square_attacked(position, transit, them, occupancy_without_king) &&
           !is_square_attacked(position, to, them, occupancy_without_king);
}

[[nodiscard]] bool validate_non_pawn_normal_move(const Position& position, Move move,
                                                 Piece moved_piece, StateInfo& state) noexcept {
    const auto us = moved_piece.color();
    const auto from = move.from();
    const auto to = move.to();
    const auto target_piece = position.piece_on(to);
    if (target_piece.is_valid()) {
        if (!is_enemy_piece(target_piece, us)) {
            return false;
        }
        state.captured_piece = target_piece;
        state.captured_square = to;
    }

    switch (moved_piece.type()) {
    case PieceType::Knight:
        return has_square(knight_attacks(from), to);
    case PieceType::Bishop:
        return has_square(bishop_attacks(from, position.occupancy()), to);
    case PieceType::Rook:
        return has_square(rook_attacks(from, position.occupancy()), to);
    case PieceType::Queen:
        return has_square(queen_attacks(from, position.occupancy()), to);
    case PieceType::King:
        return has_square(king_attacks(from), to);
    case PieceType::Pawn:
    case PieceType::None:
        return false;
    }
    return false;
}

[[nodiscard]] bool validate_move_shape(const Position& position, Move move, Piece moved_piece,
                                       StateInfo& state) noexcept {
    if (!move.is_valid() || move.is_null() || !move.from().is_valid() || !move.to().is_valid() ||
        move.from() == move.to() || !moved_piece.is_valid() ||
        moved_piece.color() != position.side_to_move()) {
        return false;
    }

    if (move.is_en_passant()) {
        return validate_en_passant_move(position, move, moved_piece, state);
    }
    if (move.is_castling()) {
        return validate_castling_move(position, move, moved_piece);
    }
    if (move.is_promotion()) {
        if (moved_piece.type() != PieceType::Pawn) {
            return false;
        }
        return validate_pawn_promotion_move(position, move, moved_piece.color(), state);
    }
    if (moved_piece.type() == PieceType::Pawn) {
        return validate_pawn_normal_move(position, move, moved_piece.color(), state);
    }
    return validate_non_pawn_normal_move(position, move, moved_piece, state);
}

[[nodiscard]] std::expected<void, FenError>
validate_castling_piece_state(const Position& position) {
    const auto rights = position.castling_rights();
    if (rights.white_king_side() &&
        (!has_piece(position, Color::White, PieceType::King, square(file_e, rank_1)) ||
         !has_piece(position, Color::White, PieceType::Rook, square(file_h, rank_1)))) {
        return std::unexpected(make_error(FenErrorCode::InvalidCastlingRights,
                                          "White king-side castling lacks king or rook"));
    }
    if (rights.white_queen_side() &&
        (!has_piece(position, Color::White, PieceType::King, square(file_e, rank_1)) ||
         !has_piece(position, Color::White, PieceType::Rook, square(file_a, rank_1)))) {
        return std::unexpected(make_error(FenErrorCode::InvalidCastlingRights,
                                          "White queen-side castling lacks king or rook"));
    }
    if (rights.black_king_side() &&
        (!has_piece(position, Color::Black, PieceType::King, square(file_e, rank_8)) ||
         !has_piece(position, Color::Black, PieceType::Rook, square(file_h, rank_8)))) {
        return std::unexpected(make_error(FenErrorCode::InvalidCastlingRights,
                                          "Black king-side castling lacks king or rook"));
    }
    if (rights.black_queen_side() &&
        (!has_piece(position, Color::Black, PieceType::King, square(file_e, rank_8)) ||
         !has_piece(position, Color::Black, PieceType::Rook, square(file_a, rank_8)))) {
        return std::unexpected(make_error(FenErrorCode::InvalidCastlingRights,
                                          "Black queen-side castling lacks king or rook"));
    }

    return {};
}

[[nodiscard]] std::expected<void, FenError> validate_board_state(const Position& position) {
    if (popcount(position.pieces(Color::White, PieceType::King)) != 1 ||
        popcount(position.pieces(Color::Black, PieceType::King)) != 1) {
        return std::unexpected(make_error(FenErrorCode::InvalidBoardState,
                                          "FEN must contain exactly one king per side"));
    }

    const auto first_rank = bitboard_detail::make_ray(square(file_a, rank_1), Direction::East) |
                            square_bb(square(file_a, rank_1));
    const auto eighth_rank = bitboard_detail::make_ray(square(file_a, rank_8), Direction::East) |
                             square_bb(square(file_a, rank_8));
    const auto white_pawns = position.pieces(Color::White, PieceType::Pawn);
    const auto black_pawns = position.pieces(Color::Black, PieceType::Pawn);
    if (((white_pawns | black_pawns) & (first_rank | eighth_rank)) != empty_bitboard) {
        return std::unexpected(make_error(FenErrorCode::InvalidBoardState,
                                          "Pawns cannot appear on the first or eighth rank"));
    }

    if (popcount(position.occupancy(Color::White)) > max_side_piece_count ||
        popcount(position.occupancy(Color::Black)) > max_side_piece_count) {
        return std::unexpected(
            make_error(FenErrorCode::InvalidBoardState, "A side cannot have more than 16 pieces"));
    }

    if (!position.has_consistent_occupancies()) {
        return std::unexpected(make_error(FenErrorCode::InvalidBoardState,
                                          "Mailbox, bitboards and occupancies disagree"));
    }

    return {};
}

void append_piece_placement(const Position& position, std::string& result) {
    for (int rank = rank_8; rank >= rank_1; --rank) {
        int empty_count = 0;
        for (int file = file_a; file < File::count; ++file) {
            const auto piece = position.piece_on(square(file, rank));
            if (piece.is_none()) {
                ++empty_count;
                continue;
            }

            if (empty_count > 0) {
                result.push_back(static_cast<char>('0' + empty_count));
                empty_count = 0;
            }
            result.push_back(piece_to_fen_char(piece));
        }

        if (empty_count > 0) {
            result.push_back(static_cast<char>('0' + empty_count));
        }
        if (rank != rank_1) {
            result.push_back('/');
        }
    }
}

void append_castling_rights(CastlingRights castling_rights, std::string& result) {
    if (castling_rights.empty()) {
        result.push_back('-');
        return;
    }

    if (castling_rights.white_king_side()) {
        result.push_back('K');
    }
    if (castling_rights.white_queen_side()) {
        result.push_back('Q');
    }
    if (castling_rights.black_king_side()) {
        result.push_back('k');
    }
    if (castling_rights.black_queen_side()) {
        result.push_back('q');
    }
}

void append_en_passant(Square en_passant_square, std::string& result) {
    if (!en_passant_square.is_valid()) {
        result.push_back('-');
        return;
    }

    result.push_back(file_char(en_passant_square.file()));
    result.push_back(rank_char(en_passant_square.rank()));
}

} // namespace

class PositionBuilder {
  public:
    explicit PositionBuilder(Position& position) noexcept : position_{&position} {}

    [[nodiscard]] std::expected<void, FenError> parse_piece_placement(std::string_view field) {
        position_->clear();

        int rank = rank_8;
        int file = file_a;
        bool previous_was_digit = false;

        for (const auto symbol : field) {
            if (symbol == '/') {
                if (file != File::count || rank == rank_1) {
                    return std::unexpected(make_error(FenErrorCode::InvalidPiecePlacement,
                                                      "FEN rank does not contain eight files"));
                }
                --rank;
                file = file_a;
                previous_was_digit = false;
                continue;
            }

            if (rank < rank_1) {
                return std::unexpected(
                    make_error(FenErrorCode::InvalidPiecePlacement, "FEN has too many ranks"));
            }

            if (is_digit(symbol)) {
                if (symbol == '0' || previous_was_digit) {
                    return std::unexpected(make_error(FenErrorCode::InvalidPiecePlacement,
                                                      "FEN rank has an invalid empty count"));
                }
                file += symbol - '0';
                if (file > File::count) {
                    return std::unexpected(make_error(FenErrorCode::InvalidPiecePlacement,
                                                      "FEN rank exceeds eight files"));
                }
                previous_was_digit = true;
                continue;
            }

            const auto piece = piece_from_fen_char(symbol);
            if (!piece.has_value()) {
                return std::unexpected(make_error(FenErrorCode::InvalidPiecePlacement,
                                                  "FEN contains an invalid piece symbol"));
            }
            if (file >= File::count) {
                return std::unexpected(make_error(FenErrorCode::InvalidPiecePlacement,
                                                  "FEN rank exceeds eight files"));
            }

            position_->put_piece(*piece, square(file, rank));
            ++file;
            previous_was_digit = false;
        }

        if (rank != rank_1 || file != File::count) {
            return std::unexpected(make_error(FenErrorCode::InvalidPiecePlacement,
                                              "FEN does not contain eight ranks"));
        }

        return {};
    }

    [[nodiscard]] std::expected<void, FenError> parse_side_to_move(std::string_view field) {
        if (field == "w") {
            position_->side_to_move_ = Color::White;
            return {};
        }
        if (field == "b") {
            position_->side_to_move_ = Color::Black;
            return {};
        }
        return std::unexpected(
            make_error(FenErrorCode::InvalidSideToMove, "Side to move must be 'w' or 'b'"));
    }

    [[nodiscard]] std::expected<void, FenError> parse_castling_rights(std::string_view field) {
        position_->castling_rights_ = CastlingRights::none();
        if (field == "-") {
            return {};
        }
        if (field.empty() || field.contains('-')) {
            return std::unexpected(make_error(FenErrorCode::InvalidCastlingRights,
                                              "Castling rights must be '-' or KQkq subset"));
        }

        int last_order_index = -1;
        for (const auto symbol : field) {
            const auto order_index = castling_order.find(symbol);
            if (order_index == std::string_view::npos) {
                return std::unexpected(make_error(FenErrorCode::InvalidCastlingRights,
                                                  "Castling rights must be unique and canonical"));
            }
            const auto order_index_as_int = static_cast<int>(order_index);
            if (order_index_as_int <= last_order_index) {
                return std::unexpected(make_error(FenErrorCode::InvalidCastlingRights,
                                                  "Castling rights must be unique and canonical"));
            }
            last_order_index = order_index_as_int;

            if (symbol == 'K') {
                position_->castling_rights_.add(Color::White, CastlingSide::KingSide);
            } else if (symbol == 'Q') {
                position_->castling_rights_.add(Color::White, CastlingSide::QueenSide);
            } else if (symbol == 'k') {
                position_->castling_rights_.add(Color::Black, CastlingSide::KingSide);
            } else {
                position_->castling_rights_.add(Color::Black, CastlingSide::QueenSide);
            }
        }

        return {};
    }

    [[nodiscard]] std::expected<void, FenError> parse_en_passant(std::string_view field) {
        position_->en_passant_square_ = Square{};
        if (field == "-") {
            return {};
        }
        if (field.size() != 2U || field[0] < 'a' || field[0] > 'h' ||
            (field[1] != '3' && field[1] != '6')) {
            return std::unexpected(
                make_error(FenErrorCode::InvalidEnPassantSquare,
                           "En-passant square must be '-' or file plus rank 3/6"));
        }

        const auto file = field[0] - 'a';
        const auto rank = field[1] - '1';
        const auto target = square(file, rank);
        if (position_->piece_on(target).is_valid()) {
            return std::unexpected(make_error(FenErrorCode::InvalidEnPassantSquare,
                                              "En-passant target square must be empty"));
        }

        if (rank == rank_3) {
            if (position_->side_to_move_ != Color::Black ||
                !has_piece(*position_, Color::White, PieceType::Pawn, square(file, rank_4))) {
                return std::unexpected(make_error(FenErrorCode::InvalidEnPassantSquare,
                                                  "Rank-3 en-passant square requires a white pawn "
                                                  "on rank 4 and black to move"));
            }
        } else if (position_->side_to_move_ != Color::White ||
                   !has_piece(*position_, Color::Black, PieceType::Pawn, square(file, rank_5))) {
            return std::unexpected(make_error(FenErrorCode::InvalidEnPassantSquare,
                                              "Rank-6 en-passant square requires a black pawn on "
                                              "rank 5 and white to move"));
        }

        position_->en_passant_square_ = target;
        return {};
    }

    [[nodiscard]] std::expected<void, FenError> parse_halfmove_clock(std::string_view field) {
        const auto value =
            parse_non_negative_int(field, FenErrorCode::InvalidHalfmoveClock, "Halfmove clock");
        if (!value.has_value()) {
            return std::unexpected(value.error());
        }
        position_->halfmove_clock_ = *value;
        return {};
    }

    [[nodiscard]] std::expected<void, FenError> parse_fullmove_number(std::string_view field) {
        const auto value =
            parse_non_negative_int(field, FenErrorCode::InvalidFullmoveNumber, "Fullmove number");
        if (!value.has_value()) {
            return std::unexpected(value.error());
        }
        if (*value <= 0) {
            return std::unexpected(make_error(FenErrorCode::InvalidFullmoveNumber,
                                              "Fullmove number must be positive"));
        }
        position_->fullmove_number_ = *value;
        return {};
    }

  private:
    Position* position_;
};

Position::Position() noexcept { clear(); }

Position Position::startpos() {
    auto result = from_fen(startpos_fen);
    if (!result.has_value()) {
        return Position{};
    }
    return *result;
}

Position::FenResult Position::from_fen(std::string_view fen) {
    const auto fields = split_fen_fields(fen);
    if (!fields.has_value()) {
        return std::unexpected(fields.error());
    }

    Position position;
    PositionBuilder builder{position};

    if (auto parsed = builder.parse_piece_placement((*fields)[piece_placement_field]);
        !parsed.has_value()) {
        return std::unexpected(parsed.error());
    }
    if (auto parsed = builder.parse_side_to_move((*fields)[side_to_move_field]);
        !parsed.has_value()) {
        return std::unexpected(parsed.error());
    }
    if (auto parsed = builder.parse_castling_rights((*fields)[castling_field]);
        !parsed.has_value()) {
        return std::unexpected(parsed.error());
    }
    if (auto parsed = builder.parse_en_passant((*fields)[en_passant_field]); !parsed.has_value()) {
        return std::unexpected(parsed.error());
    }
    if (auto parsed = builder.parse_halfmove_clock((*fields)[halfmove_field]);
        !parsed.has_value()) {
        return std::unexpected(parsed.error());
    }
    if (auto parsed = builder.parse_fullmove_number((*fields)[fullmove_field]);
        !parsed.has_value()) {
        return std::unexpected(parsed.error());
    }

    position.refresh_derived_state();

    if (auto valid_castling = validate_castling_piece_state(position);
        !valid_castling.has_value()) {
        return std::unexpected(valid_castling.error());
    }
    if (auto valid_board = validate_board_state(position); !valid_board.has_value()) {
        return std::unexpected(valid_board.error());
    }

    return position;
}

std::string Position::to_fen() const {
    std::string result;
    result.reserve(fen_reserve_size);
    append_piece_placement(*this, result);

    result.push_back(' ');
    result.push_back(side_to_move_ == Color::White ? 'w' : 'b');
    result.push_back(' ');
    append_castling_rights(castling_rights_, result);
    result.push_back(' ');
    append_en_passant(en_passant_square_, result);
    result.push_back(' ');
    result += std::to_string(halfmove_clock_);
    result.push_back(' ');
    result += std::to_string(fullmove_number_);

    return result;
}

Piece Position::piece_on(Square square_value) const noexcept {
    if (!square_value.is_valid()) {
        return Piece{};
    }
    return mailbox_[static_cast<std::size_t>(square_value.index())];
}

Bitboard Position::pieces(Piece piece) const noexcept {
    if (!piece.is_valid()) {
        return empty_bitboard;
    }
    return piece_bitboards_[static_cast<std::size_t>(piece.index())];
}

Bitboard Position::pieces(Color color, PieceType type) const noexcept {
    return pieces(Piece::make(color, type));
}

Bitboard Position::occupancy(Color color) const noexcept {
    if (color == Color::White) {
        return occupancies_[0];
    }
    if (color == Color::Black) {
        return occupancies_[1];
    }
    return empty_bitboard;
}

bool Position::has_consistent_occupancies() const noexcept {
    PieceBitboards expected_piece_bitboards{};
    ColorOccupancies expected_occupancies{};
    Bitboard expected_occupancy = empty_bitboard;

    for (int index = 0; index < Square::count; ++index) {
        const auto piece = mailbox_[static_cast<std::size_t>(index)];
        if (!piece.is_valid()) {
            continue;
        }

        const auto square_bitboard = square_bb(Square::from_index(index));
        expected_piece_bitboards[static_cast<std::size_t>(piece.index())] |= square_bitboard;
        expected_occupancies[static_cast<std::size_t>(piece.color())] |= square_bitboard;
        expected_occupancy |= square_bitboard;
    }

    return piece_bitboards_ == expected_piece_bitboards && occupancies_ == expected_occupancies &&
           occupancy_ == expected_occupancy;
}

ZobristKey Position::recompute_zobrist_key() const noexcept {
    const auto& tables = zobrist_tables();
    ZobristKey key = 0;

    for (int piece_index = 0; piece_index < Piece::count; ++piece_index) {
        auto remaining = piece_bitboards_[static_cast<std::size_t>(piece_index)];
        while (remaining != empty_bitboard) {
            const auto current = pop_lsb(remaining);
            key ^= tables.pieces[static_cast<std::size_t>(piece_index)]
                                [static_cast<std::size_t>(current.index())];
        }
    }

    if (side_to_move_ == Color::Black) {
        key ^= tables.side_to_move;
    }

    key ^= tables.castling[castling_rights_.raw()];

    if (en_passant_square_.is_valid()) {
        key ^= tables.en_passant_file[static_cast<std::size_t>(en_passant_square_.file().index())];
    }

    return key;
}

bool Position::make_move(Move move, StateInfo& state) noexcept {
    state = StateInfo{
        .moved_piece = piece_on(move.from()),
        .captured_piece = Piece{},
        .captured_square = Square{},
        .castling_rights = castling_rights_,
        .en_passant_square = en_passant_square_,
        .halfmove_clock = halfmove_clock_,
        .fullmove_number = fullmove_number_,
        .zobrist_key = zobrist_key_,
    };

    const auto moved_piece = state.moved_piece;
    if (!validate_move_shape(*this, move, moved_piece, state)) {
        return false;
    }

    const auto us = side_to_move_;
    const auto them = opposite(us);
    const auto from = move.from();
    const auto to = move.to();

    if (move.is_castling()) {
        const auto rook_move = castling_rook_move(us, to);
        move_piece(moved_piece, from, to);
        move_piece(Piece::make(us, PieceType::Rook), rook_move.from, rook_move.to);
    } else if (move.is_en_passant()) {
        remove_piece(moved_piece, from);
        remove_piece(state.captured_piece, state.captured_square);
        put_piece(moved_piece, to);
    } else if (move.is_promotion()) {
        if (state.captured_piece.is_valid()) {
            remove_piece(state.captured_piece, state.captured_square);
        }
        remove_piece(moved_piece, from);
        put_piece(Piece::make(us, move.promotion_piece()), to);
    } else {
        if (state.captured_piece.is_valid()) {
            remove_piece(state.captured_piece, state.captured_square);
        }
        move_piece(moved_piece, from, to);
    }

    castling_rights_ = castling_rights_after_move(state.castling_rights, moved_piece, from,
                                                  state.captured_piece, state.captured_square);
    en_passant_square_ = Square{};
    if (moved_piece.type() == PieceType::Pawn && move_is_pawn_double_push(us, from, to)) {
        en_passant_square_ = en_passant_target_after_double_push(us, from);
    }

    halfmove_clock_ = moved_piece.type() == PieceType::Pawn || state.captured_piece.is_valid()
                          ? 0
                          : state.halfmove_clock + 1;
    fullmove_number_ = state.fullmove_number + (us == Color::Black ? 1 : 0);
    side_to_move_ = them;

    auto key = state.zobrist_key;
    key ^= castling_zobrist(state.castling_rights);
    key ^= en_passant_zobrist(state.en_passant_square);
    key ^= side_to_move_zobrist();
    key ^= piece_zobrist(moved_piece, from);
    if (state.captured_piece.is_valid()) {
        key ^= piece_zobrist(state.captured_piece, state.captured_square);
    }
    if (move.is_castling()) {
        const auto rook_move = castling_rook_move(us, to);
        const auto rook = Piece::make(us, PieceType::Rook);
        key ^= piece_zobrist(moved_piece, to);
        key ^= piece_zobrist(rook, rook_move.from);
        key ^= piece_zobrist(rook, rook_move.to);
    } else if (move.is_promotion()) {
        key ^= piece_zobrist(Piece::make(us, move.promotion_piece()), to);
    } else {
        key ^= piece_zobrist(moved_piece, to);
    }
    key ^= castling_zobrist(castling_rights_);
    key ^= en_passant_zobrist(en_passant_square_);
    zobrist_key_ = key;

    if (in_check(*this, us) || !has_consistent_occupancies()) {
        unmake_move(move, state);
        return false;
    }

    return true;
}

void Position::unmake_move(Move move, const StateInfo& state) noexcept {
    const auto moved_piece = state.moved_piece;
    if (!moved_piece.is_valid() || !move.from().is_valid() || !move.to().is_valid()) {
        return;
    }

    const auto us = moved_piece.color();
    const auto from = move.from();
    const auto to = move.to();

    if (move.is_castling()) {
        const auto rook_move = castling_rook_move(us, to);
        move_piece(Piece::make(us, PieceType::King), to, from);
        move_piece(Piece::make(us, PieceType::Rook), rook_move.to, rook_move.from);
    } else if (move.is_en_passant()) {
        remove_piece(moved_piece, to);
        put_piece(moved_piece, from);
        put_piece(state.captured_piece, state.captured_square);
    } else if (move.is_promotion()) {
        remove_piece(Piece::make(us, move.promotion_piece()), to);
        put_piece(moved_piece, from);
        if (state.captured_piece.is_valid()) {
            put_piece(state.captured_piece, state.captured_square);
        }
    } else {
        move_piece(moved_piece, to, from);
        if (state.captured_piece.is_valid()) {
            put_piece(state.captured_piece, state.captured_square);
        }
    }

    side_to_move_ = us;
    castling_rights_ = state.castling_rights;
    en_passant_square_ = state.en_passant_square;
    halfmove_clock_ = state.halfmove_clock;
    fullmove_number_ = state.fullmove_number;
    zobrist_key_ = state.zobrist_key;
}

bool Position::make_null_move(StateInfo& state) noexcept {
    if (!is_valid_color(side_to_move_)) {
        return false;
    }

    state = StateInfo{
        .moved_piece = Piece{},
        .captured_piece = Piece{},
        .captured_square = Square{},
        .castling_rights = castling_rights_,
        .en_passant_square = en_passant_square_,
        .halfmove_clock = halfmove_clock_,
        .fullmove_number = fullmove_number_,
        .zobrist_key = zobrist_key_,
    };

    side_to_move_ = opposite(side_to_move_);
    en_passant_square_ = Square{};
    zobrist_key_ =
        state.zobrist_key ^ en_passant_zobrist(state.en_passant_square) ^ side_to_move_zobrist();
    return true;
}

void Position::unmake_null_move(const StateInfo& state) noexcept {
    side_to_move_ = opposite(side_to_move_);
    castling_rights_ = state.castling_rights;
    en_passant_square_ = state.en_passant_square;
    halfmove_clock_ = state.halfmove_clock;
    fullmove_number_ = state.fullmove_number;
    zobrist_key_ = state.zobrist_key;
}

void Position::clear() noexcept {
    piece_bitboards_.fill(empty_bitboard);
    occupancies_.fill(empty_bitboard);
    occupancy_ = empty_bitboard;
    mailbox_.fill(Piece{});
    side_to_move_ = Color::White;
    castling_rights_ = CastlingRights::none();
    en_passant_square_ = Square{};
    halfmove_clock_ = 0;
    fullmove_number_ = 1;
    zobrist_key_ = 0;
}

void Position::put_piece(Piece piece, Square square_value) noexcept {
    if (!piece.is_valid() || !square_value.is_valid()) {
        return;
    }

    const auto square_index = static_cast<std::size_t>(square_value.index());
    const auto piece_index = static_cast<std::size_t>(piece.index());
    const auto color_index = static_cast<std::size_t>(piece.color());
    const auto bitboard = square_bb(square_value);
    mailbox_[square_index] = piece;
    piece_bitboards_[piece_index] |= bitboard;
    occupancies_[color_index] |= bitboard;
    occupancy_ |= bitboard;
}

void Position::remove_piece(Piece piece, Square square_value) noexcept {
    if (!piece.is_valid() || !square_value.is_valid()) {
        return;
    }

    const auto square_index = static_cast<std::size_t>(square_value.index());
    const auto piece_index = static_cast<std::size_t>(piece.index());
    const auto color_index = static_cast<std::size_t>(piece.color());
    const auto bitboard = square_bb(square_value);
    mailbox_[square_index] = Piece{};
    piece_bitboards_[piece_index] &= ~bitboard;
    occupancies_[color_index] &= ~bitboard;
    occupancy_ &= ~bitboard;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void Position::move_piece(Piece piece, Square from_square, Square to_square) noexcept {
    remove_piece(piece, from_square);
    put_piece(piece, to_square);
}

void Position::refresh_derived_state() noexcept {
    occupancies_.fill(empty_bitboard);
    occupancy_ = empty_bitboard;

    for (int piece_index = 0; piece_index < Piece::count; ++piece_index) {
        const auto bitboard = piece_bitboards_[static_cast<std::size_t>(piece_index)];
        const auto color_index = static_cast<std::size_t>(piece_index / Piece::type_count);
        occupancies_[color_index] |= bitboard;
        occupancy_ |= bitboard;
    }

    zobrist_key_ = recompute_zobrist_key();
}

} // namespace eng

// NOLINTEND(readability-identifier-length)
