#include <eng/movegen.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <string_view>

// Square names such as e1 and h8 are clearer than expanded identifiers in chess tests.
// NOLINTBEGIN(readability-identifier-length)

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
constexpr int rank_3 = 2;
constexpr int rank_4 = 3;
constexpr int rank_5 = 4;
constexpr int rank_6 = 5;
constexpr int rank_7 = 6;
constexpr int rank_8 = 7;

[[nodiscard]] constexpr eng::Square square(int file_index, int rank_index) {
    return eng::Square::from_file_rank(eng::File::from_index(file_index),
                                       eng::Rank::from_index(rank_index));
}

[[nodiscard]] eng::Position parse(std::string_view fen) {
    const auto position = eng::Position::from_fen(fen);
    REQUIRE(position.has_value());
    return *position;
}

[[nodiscard]] bool contains_move(const eng::MoveList& moves, eng::Move expected) {
    return std::ranges::find(moves, expected) != moves.end();
}

[[nodiscard]] bool contains_normal(const eng::MoveList& moves, eng::Square from, eng::Square to) {
    return contains_move(moves, eng::Move::normal(from, to));
}

[[nodiscard]] bool contains_promotion(const eng::MoveList& moves, eng::Square from, eng::Square to,
                                      eng::PieceType promotion_piece) {
    return contains_move(moves, eng::Move::promotion(from, to, promotion_piece));
}

[[nodiscard]] int count_from(const eng::MoveList& moves, eng::Square from) {
    return static_cast<int>(
        std::ranges::count_if(moves, [from](eng::Move move) { return move.from() == from; }));
}

void check_all_promotions(const eng::MoveList& moves, eng::Square from, eng::Square to) {
    CHECK(contains_promotion(moves, from, to, eng::PieceType::Knight));
    CHECK(contains_promotion(moves, from, to, eng::PieceType::Bishop));
    CHECK(contains_promotion(moves, from, to, eng::PieceType::Rook));
    CHECK(contains_promotion(moves, from, to, eng::PieceType::Queen));
}

} // namespace

TEST_CASE("start position generates context-aware pseudo-legal candidates") {
    const auto position = eng::Position::startpos();

    const auto all_moves = eng::generate_moves(position);
    const auto quiets = eng::generate_quiets(position);
    const auto captures = eng::generate_captures(position);

    CHECK(!eng::side_to_move_in_check(position));
    CHECK(all_moves.size() == 20U);
    CHECK(quiets.size() == 20U);
    CHECK(captures.empty());

    CHECK(contains_normal(all_moves, square(file_e, rank_2), square(file_e, rank_4)));
    CHECK(contains_normal(all_moves, square(file_g, rank_1), square(file_f, rank_3)));
    CHECK(!contains_move(quiets,
                         eng::Move::castling(square(file_e, rank_1), square(file_g, rank_1))));
    CHECK(!contains_move(quiets,
                         eng::Move::castling(square(file_e, rank_1), square(file_c, rank_1))));
}

TEST_CASE("pawns generate pushes captures en-passant and all promotion pieces") {
    const auto position = parse("4k2r/6P1/8/3pP3/8/8/3P4/4K3 w - d6 0 1");

    const auto quiets = eng::generate_quiets(position);
    const auto captures = eng::generate_captures(position);

    CHECK(contains_normal(quiets, square(file_d, rank_2), square(file_d, rank_3)));
    CHECK(contains_normal(quiets, square(file_d, rank_2), square(file_d, rank_4)));
    CHECK(contains_normal(quiets, square(file_e, rank_5), square(file_e, rank_6)));
    check_all_promotions(quiets, square(file_g, rank_7), square(file_g, rank_8));

    CHECK(contains_move(captures,
                        eng::Move::en_passant(square(file_e, rank_5), square(file_d, rank_6))));
    check_all_promotions(captures, square(file_g, rank_7), square(file_h, rank_8));
}

TEST_CASE("piece move generation respects own blockers and enemy capture targets") {
    const auto position = parse("4k3/8/3p4/8/1p1Q1P2/8/3P4/4K3 w - - 0 1");
    const auto all_moves = eng::generate_moves(position, eng::MoveGenStage::All);

    CHECK(contains_normal(all_moves, square(file_d, rank_4), square(file_c, rank_4)));
    CHECK(contains_normal(all_moves, square(file_d, rank_4), square(file_b, rank_4)));
    CHECK(!contains_normal(all_moves, square(file_d, rank_4), square(file_a, rank_4)));
    CHECK(!contains_normal(all_moves, square(file_d, rank_4), square(file_f, rank_4)));

    CHECK(contains_normal(all_moves, square(file_d, rank_4), square(file_d, rank_5)));
    CHECK(contains_normal(all_moves, square(file_d, rank_4), square(file_d, rank_6)));
    CHECK(!contains_normal(all_moves, square(file_d, rank_4), square(file_d, rank_7)));
}

TEST_CASE("castling is generated as a quiet candidate when orthodox path is empty") {
    const auto open_castling = parse("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    const auto blocked_castling = eng::Position::startpos();

    const auto open_quiets = eng::generate_quiets(open_castling);
    const auto blocked_quiets = eng::generate_quiets(blocked_castling);

    CHECK(contains_move(open_quiets,
                        eng::Move::castling(square(file_e, rank_1), square(file_g, rank_1))));
    CHECK(contains_move(open_quiets,
                        eng::Move::castling(square(file_e, rank_1), square(file_c, rank_1))));
    CHECK(!contains_move(blocked_quiets,
                         eng::Move::castling(square(file_e, rank_1), square(file_g, rank_1))));
    CHECK(!contains_move(blocked_quiets,
                         eng::Move::castling(square(file_e, rank_1), square(file_c, rank_1))));
}

TEST_CASE("attack detection reports pawn knight slider and king attacks") {
    const auto pawn_check = parse("4k3/8/8/3p4/4K3/8/8/8 w - - 0 1");
    CHECK(eng::in_check(pawn_check, eng::Color::White));
    CHECK(eng::side_to_move_in_check(pawn_check));
    CHECK(eng::is_square_attacked(pawn_check, square(file_e, rank_4), eng::Color::Black));

    const auto knight_check = parse("4k3/8/8/8/4K3/8/5n2/8 w - - 0 1");
    CHECK(eng::in_check(knight_check, eng::Color::White));
    CHECK(eng::attackers_to(knight_check, square(file_e, rank_4), eng::Color::Black) ==
          eng::square_bb(square(file_f, rank_2)));

    const auto double_check = parse("k3r3/8/8/8/7b/8/3N4/4K3 w - - 0 1");
    const auto attackers =
        eng::attackers_to(double_check, square(file_e, rank_1), eng::Color::Black);
    CHECK(eng::popcount(attackers) == 2);
    CHECK(eng::has_square(attackers, square(file_e, rank_8)));
    CHECK(eng::has_square(attackers, square(file_h, rank_4)));
}

TEST_CASE("single-check evasions include king moves blocks and captures only") {
    const auto line_check = parse("k3r3/8/8/8/8/8/8/2B1K3 w - - 0 1");
    const auto line_evasions = eng::generate_evasions(line_check);

    CHECK(eng::side_to_move_in_check(line_check));
    CHECK(contains_normal(line_evasions, square(file_c, rank_1), square(file_e, rank_3)));
    CHECK(!contains_normal(line_evasions, square(file_c, rank_1), square(file_a, rank_3)));
    CHECK(!contains_normal(line_evasions, square(file_e, rank_1), square(file_e, rank_2)));

    const auto knight_check = parse("4k3/8/8/8/8/5n2/7N/4K3 w - - 0 1");
    const auto knight_evasions = eng::generate_moves(knight_check);

    CHECK(eng::side_to_move_in_check(knight_check));
    CHECK(contains_normal(knight_evasions, square(file_h, rank_2), square(file_f, rank_3)));
}

TEST_CASE("single-check evasions include pawn double-push interpositions") {
    const auto white_to_block = parse("4k3/8/8/8/r6K/8/6P1/8 w - - 0 1");
    const auto white_evasions = eng::generate_evasions(white_to_block);
    const auto white_contextual_moves = eng::generate_moves(white_to_block);

    CHECK(eng::side_to_move_in_check(white_to_block));
    CHECK(contains_normal(white_evasions, square(file_g, rank_2), square(file_g, rank_4)));
    CHECK(contains_normal(white_contextual_moves, square(file_g, rank_2), square(file_g, rank_4)));
    CHECK(!contains_normal(white_evasions, square(file_g, rank_2), square(file_g, rank_3)));
    CHECK(!contains_normal(white_contextual_moves, square(file_g, rank_2), square(file_g, rank_3)));

    const auto black_to_block = parse("8/6p1/8/R6k/8/8/8/4K3 b - - 0 1");
    const auto black_evasions = eng::generate_evasions(black_to_block);
    const auto black_contextual_moves = eng::generate_moves(black_to_block);

    CHECK(eng::side_to_move_in_check(black_to_block));
    CHECK(contains_normal(black_evasions, square(file_g, rank_7), square(file_g, rank_5)));
    CHECK(contains_normal(black_contextual_moves, square(file_g, rank_7), square(file_g, rank_5)));
    CHECK(!contains_normal(black_evasions, square(file_g, rank_7), square(file_g, rank_6)));
    CHECK(!contains_normal(black_contextual_moves, square(file_g, rank_7), square(file_g, rank_6)));
}

TEST_CASE("double-check evasions generate only safe king moves") {
    const auto position = parse("k3r3/8/8/8/7b/8/3N4/4K3 w - - 0 1");
    const auto evasions = eng::generate_evasions(position);

    CHECK(eng::side_to_move_in_check(position));
    CHECK(evasions.size() == 2U);
    CHECK(contains_normal(evasions, square(file_e, rank_1), square(file_d, rank_1)));
    CHECK(contains_normal(evasions, square(file_e, rank_1), square(file_f, rank_1)));
    CHECK(count_from(evasions, square(file_d, rank_2)) == 0);
}

// NOLINTEND(readability-identifier-length)
