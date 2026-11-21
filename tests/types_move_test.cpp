#include <eng/move.hpp>

#include <doctest/doctest.h>

#include <cstdint>
#include <type_traits>

namespace {

constexpr int file_a = 0;
constexpr int file_d = 3;
constexpr int file_e = 4;
constexpr int file_g = 6;
constexpr int file_h = 7;
constexpr int rank_1 = 0;
constexpr int rank_2 = 1;
constexpr int rank_4 = 3;
constexpr int rank_5 = 4;
constexpr int rank_8 = 7;

[[nodiscard]] constexpr eng::Square square(int file_index, int rank_index) {
    return eng::Square::from_file_rank(eng::File::from_index(file_index),
                                       eng::Rank::from_index(rank_index));
}

} // namespace

TEST_CASE("color and piece domain helpers are strongly typed") {
    static_assert(std::is_trivially_copyable_v<eng::Piece>);

    CHECK(eng::opposite(eng::Color::White) == eng::Color::Black);
    CHECK(eng::opposite(eng::Color::Black) == eng::Color::White);
    CHECK(eng::opposite(eng::Color::None) == eng::Color::None);

    constexpr auto white_queen = eng::Piece::make(eng::Color::White, eng::PieceType::Queen);
    constexpr auto black_knight = eng::Piece::make(eng::Color::Black, eng::PieceType::Knight);
    constexpr auto invalid_piece = eng::Piece::make(eng::Color::None, eng::PieceType::Pawn);

    static_assert(white_queen.is_valid());
    static_assert(white_queen.index() == static_cast<int>(eng::PieceType::Queen));
    static_assert(black_knight.is_valid());
    static_assert(black_knight.index() ==
                  (eng::Piece::type_count + static_cast<int>(eng::PieceType::Knight)));

    CHECK(white_queen.color() == eng::Color::White);
    CHECK(white_queen.type() == eng::PieceType::Queen);
    CHECK(black_knight.color() == eng::Color::Black);
    CHECK(black_knight.type() == eng::PieceType::Knight);
    CHECK(invalid_piece.is_none());
    CHECK(invalid_piece.color() == eng::Color::None);
    CHECK(invalid_piece.type() == eng::PieceType::None);
}

TEST_CASE("squares preserve a1 based indexing and bitboard masks") {
    static_assert(std::is_trivially_copyable_v<eng::Square>);

    constexpr auto a1_square = square(file_a, rank_1);
    constexpr auto e2_square = square(file_e, rank_2);
    constexpr auto h8_square = square(file_h, rank_8);
    constexpr auto invalid_square = eng::Square::from_index(eng::Square::count);
    constexpr auto e2_index = (rank_2 * eng::File::count) + file_e;
    constexpr auto h8_index = (rank_8 * eng::File::count) + file_h;

    static_assert(a1_square.is_valid());
    static_assert(a1_square.index() == file_a);
    static_assert(e2_square.index() == e2_index);
    static_assert(h8_square.index() == h8_index);
    static_assert(!invalid_square.is_valid());

    CHECK(e2_square.file() == eng::File::from_index(file_e));
    CHECK(e2_square.rank() == eng::Rank::from_index(rank_2));
    CHECK(eng::square_bb(a1_square) == eng::Bitboard{1});
    CHECK(eng::square_bb(e2_square) == (eng::Bitboard{1} << e2_index));
    CHECK(eng::square_bb(h8_square) == (eng::Bitboard{1} << h8_index));
    CHECK(eng::square_bb(invalid_square) == eng::Bitboard{0});
}

TEST_CASE("move encoding stores from to kind and promotion in 16 bits") {
    static_assert(sizeof(eng::Move) == sizeof(std::uint16_t));
    static_assert(std::is_trivially_copyable_v<eng::Move>);

    constexpr auto e2_square = square(file_e, rank_2);
    constexpr auto e4_square = square(file_e, rank_4);
    constexpr auto d5_square = square(file_d, rank_5);
    constexpr auto e8_square = square(file_e, rank_8);
    constexpr auto g1_square = square(file_g, rank_1);

    constexpr auto quiet_move = eng::Move::normal(e2_square, e4_square);
    constexpr auto capture_like_move = eng::Move::normal(e4_square, d5_square);
    constexpr auto promotion_move =
        eng::Move::promotion(e2_square, e8_square, eng::PieceType::Queen);
    constexpr auto en_passant_move = eng::Move::en_passant(e4_square, d5_square);
    constexpr auto castling_move = eng::Move::castling(e2_square, g1_square);

    constexpr auto expected_quiet_raw =
        static_cast<eng::Move::Raw>(e2_square.index() | (e4_square.index() << eng::Move::to_shift));
    constexpr auto expected_promotion_raw = static_cast<eng::Move::Raw>(
        e2_square.index() | (e8_square.index() << eng::Move::to_shift) |
        (static_cast<int>(eng::MoveKind::Promotion) << eng::Move::kind_shift) |
        (eng::Move::promotion_mask << eng::Move::promotion_shift));
    constexpr auto expected_en_passant_raw = static_cast<eng::Move::Raw>(
        e4_square.index() | (d5_square.index() << eng::Move::to_shift) |
        (static_cast<int>(eng::MoveKind::EnPassant) << eng::Move::kind_shift));
    constexpr auto expected_castling_raw = static_cast<eng::Move::Raw>(
        e2_square.index() | (g1_square.index() << eng::Move::to_shift) |
        (static_cast<int>(eng::MoveKind::Castling) << eng::Move::kind_shift));

    static_assert(quiet_move.raw() == expected_quiet_raw);
    static_assert(promotion_move.raw() == expected_promotion_raw);
    static_assert(en_passant_move.raw() == expected_en_passant_raw);
    static_assert(castling_move.raw() == expected_castling_raw);

    CHECK(quiet_move.from() == e2_square);
    CHECK(quiet_move.to() == e4_square);
    CHECK(quiet_move.kind() == eng::MoveKind::Normal);
    CHECK(!quiet_move.is_promotion());

    CHECK(capture_like_move.kind() == eng::MoveKind::Normal);
    CHECK(capture_like_move.from() == e4_square);
    CHECK(capture_like_move.to() == d5_square);

    CHECK(promotion_move.kind() == eng::MoveKind::Promotion);
    CHECK(promotion_move.is_promotion());
    CHECK(promotion_move.promotion_piece() == eng::PieceType::Queen);

    CHECK(en_passant_move.is_en_passant());
    CHECK(en_passant_move.promotion_piece() == eng::PieceType::None);

    CHECK(castling_move.is_castling());
    CHECK(castling_move.promotion_piece() == eng::PieceType::None);
}

TEST_CASE("move constructors reject invalid endpoint and promotion inputs") {
    constexpr auto e2_square = square(file_e, rank_2);
    constexpr auto e4_square = square(file_e, rank_4);

    CHECK(eng::Move::normal(e2_square, e2_square).is_none());
    CHECK(eng::Move::normal(eng::Square{}, e4_square).is_none());
    CHECK(eng::Move::promotion(e2_square, e4_square, eng::PieceType::Pawn).is_none());
    CHECK(eng::Move::promotion(e2_square, e4_square, eng::PieceType::King).is_none());
    CHECK(eng::Move::promotion(e2_square, e4_square, eng::PieceType::None).is_none());
}

TEST_CASE("none and null moves are distinct sentinels") {
    constexpr auto no_move = eng::Move::none();
    constexpr auto null_move = eng::Move::null();

    CHECK(no_move.raw() == eng::Move::none_value);
    CHECK(no_move.is_none());
    CHECK(!no_move.from().is_valid());
    CHECK(!no_move.to().is_valid());

    CHECK(null_move.raw() == eng::Move::null_value);
    CHECK(null_move.is_null());
    CHECK(!null_move.is_none());
    CHECK(null_move.from() == null_move.to());

    CHECK(eng::Move::from_raw(null_move.raw()).is_null());
}
