#include <eng/see.hpp>

#include <doctest/doctest.h>

#include <string_view>

// Square names such as e1 and h8 are clearer than expanded identifiers in chess tests.
// NOLINTBEGIN(readability-identifier-length)

namespace {

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

} // namespace

TEST_CASE("see scores clearly winning and losing captures") {
    const auto winning = parse("4k3/8/8/8/4q3/8/4R3/4K3 w - - 0 1");
    const auto rook_takes_queen = eng::Move::normal(square(file_e, rank_2), square(file_e, rank_4));

    CHECK(eng::static_exchange_score(winning, rook_takes_queen).value() == 900);
    CHECK(eng::static_exchange_ge(winning, rook_takes_queen));

    const auto losing = parse("4k3/8/8/3p4/4p3/8/4Q3/4K3 w - - 0 1");
    const auto queen_takes_pawn = eng::Move::normal(square(file_e, rank_2), square(file_e, rank_4));

    CHECK(eng::static_exchange_score(losing, queen_takes_pawn).value() == -800);
    CHECK(!eng::static_exchange_ge(losing, queen_takes_pawn));
}

TEST_CASE("see honors nonzero thresholds") {
    const auto position = parse("4k3/8/8/3p4/4p3/8/4Q3/4K3 w - - 0 1");
    const auto queen_takes_pawn = eng::Move::normal(square(file_e, rank_2), square(file_e, rank_4));

    CHECK(eng::static_exchange_score(position, queen_takes_pawn).value() == -800);
    CHECK(eng::static_exchange_ge(position, queen_takes_pawn, eng::Score::centipawns(-900)));
    CHECK(!eng::static_exchange_ge(position, queen_takes_pawn, eng::Score::centipawns(-799)));
}

TEST_CASE("see forces the requested capture instead of substituting the least valuable attacker") {
    const auto position = parse("4r1k1/8/8/8/4p3/3P4/4Q3/6K1 w - - 0 1");
    const auto queen_takes_pawn = eng::Move::normal(square(file_e, rank_2), square(file_e, rank_4));

    CHECK(eng::static_exchange_score(position, queen_takes_pawn).value() == -300);
    CHECK(!eng::static_exchange_ge(position, queen_takes_pawn));
}

TEST_CASE("see handles neutral king recaptures") {
    const auto position = parse("8/8/8/4k3/4r3/8/4R3/4K3 w - - 0 1");
    const auto rook_trade = eng::Move::normal(square(file_e, rank_2), square(file_e, rank_4));

    CHECK(eng::static_exchange_score(position, rook_trade).value() == 0);
    CHECK(eng::static_exchange_ge(position, rook_trade));
}

TEST_CASE("see rejects king recaptures onto protected targets") {
    const auto position = parse("8/8/8/4k3/3Rr3/4K3/8/8 w - - 0 1");
    const auto rook_takes_rook = eng::Move::normal(square(file_d, rank_4), square(file_e, rank_4));

    CHECK(eng::static_exchange_score(position, rook_takes_rook).value() == 500);
    CHECK(eng::static_exchange_ge(position, rook_takes_rook, eng::Score::centipawns(500)));
}

TEST_CASE("see recalculates x-ray attackers after captures") {
    const auto position = parse("k3r3/8/8/8/4q3/8/4R3/4K3 w - - 0 1");
    const auto rook_takes_queen = eng::Move::normal(square(file_e, rank_2), square(file_e, rank_4));

    CHECK(eng::static_exchange_score(position, rook_takes_queen).value() == 400);
    CHECK(eng::static_exchange_ge(position, rook_takes_queen));
}

TEST_CASE("see includes promotion gain on promotion captures") {
    const auto position = parse("4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1");
    const auto promote_with_capture =
        eng::Move::promotion(square(file_g, rank_7), square(file_h, rank_8), eng::PieceType::Queen);

    CHECK(eng::static_exchange_score(position, promote_with_capture).value() == 1300);
    CHECK(eng::static_exchange_ge(position, promote_with_capture, eng::Score::centipawns(1300)));
    CHECK(!eng::static_exchange_ge(position, promote_with_capture, eng::Score::centipawns(1301)));
}

TEST_CASE("see handles en passant captures") {
    const auto position = parse("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    const auto en_passant = eng::Move::en_passant(square(file_e, rank_5), square(file_d, rank_6));

    CHECK(eng::static_exchange_score(position, en_passant).value() == 100);
    CHECK(eng::static_exchange_ge(position, en_passant));
}

TEST_CASE("see rejects malformed en passant captures") {
    const auto position = parse("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1");
    const auto en_passant = eng::Move::en_passant(square(file_e, rank_5), square(file_d, rank_6));

    CHECK(eng::static_exchange_score(position, en_passant).value() == 0);
    CHECK(eng::static_exchange_ge(position, en_passant));
}

TEST_CASE("see excludes pinned defenders from exchange sequences") {
    const auto position = parse("4k3/4b3/3p4/8/8/8/8/3QR1K1 w - - 0 1");
    const auto queen_takes_pawn = eng::Move::normal(square(file_d, rank_1), square(file_d, rank_6));

    CHECK(eng::static_exchange_score(position, queen_takes_pawn).value() == 100);
    CHECK(eng::static_exchange_ge(position, queen_takes_pawn));
}

TEST_CASE("see returns zero for non captures and castling") {
    const auto position = eng::Position::startpos();

    CHECK(eng::static_exchange_score(
              position, eng::Move::normal(square(file_g, rank_1), square(file_f, rank_3)))
              .value() == 0);
    CHECK(eng::static_exchange_score(
              parse("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
              eng::Move::castling(square(file_e, rank_1), square(file_g, rank_1)))
              .value() == 0);
}

// NOLINTEND(readability-identifier-length)
