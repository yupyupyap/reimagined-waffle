#include <eng/movegen.hpp>
#include <eng/position.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Square names such as e1 and h8 are clearer than expanded identifiers in chess tests.
// NOLINTBEGIN(readability-identifier-length)

namespace {

constexpr int file_a = 0;
constexpr int file_d = 3;
constexpr int file_e = 4;
constexpr int file_f = 5;
constexpr int file_g = 6;
constexpr int rank_1 = 0;
constexpr int rank_2 = 1;
constexpr int rank_3 = 2;
constexpr int rank_4 = 3;
constexpr int rank_5 = 4;
constexpr int rank_6 = 5;
constexpr int rank_7 = 6;
constexpr int rank_8 = 7;
constexpr int random_walk_plies = 32;
constexpr std::uint64_t random_walk_seed = 0x4C43455F5441534BULL;
constexpr std::uint64_t lcg_multiplier = 6364136223846793005ULL;
constexpr std::uint64_t lcg_increment = 1442695040888963407ULL;

[[nodiscard]] constexpr eng::Square square(int file_index, int rank_index) {
    return eng::Square::from_file_rank(eng::File::from_index(file_index),
                                       eng::Rank::from_index(rank_index));
}

[[nodiscard]] eng::Position parse(std::string_view fen) {
    const auto position = eng::Position::from_fen(fen);
    REQUIRE(position.has_value());
    return *position;
}

void check_restored(const eng::Position& position, std::string_view expected_fen,
                    eng::ZobristKey expected_key) {
    CHECK(position.to_fen() == expected_fen);
    CHECK(position.zobrist_key() == expected_key);
    CHECK(position.zobrist_key() == position.recompute_zobrist_key());
    CHECK(position.has_consistent_occupancies());
}

[[nodiscard]] bool contains_move(const eng::MoveList& moves, eng::Move expected) {
    return std::ranges::find(moves, expected) != moves.end();
}

[[nodiscard]] std::size_t next_index(std::uint64_t& seed, std::size_t size) {
    seed = (seed * lcg_multiplier) + lcg_increment;
    return static_cast<std::size_t>(seed % size);
}

} // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("make and unmake restore quiet opening moves exactly") {
    auto position = eng::Position::startpos();
    const auto original_fen = position.to_fen();
    const auto original_key = position.zobrist_key();
    const std::array moves{
        eng::Move::normal(square(file_e, rank_2), square(file_e, rank_4)),
        eng::Move::normal(square(file_e, rank_7), square(file_e, rank_5)),
        eng::Move::normal(square(file_g, rank_1), square(file_f, rank_3)),
    };
    std::vector<eng::StateInfo> states(moves.size());

    for (std::size_t index = 0; index < moves.size(); ++index) {
        REQUIRE(position.make_move(moves[index], states[index]));
        CHECK(position.zobrist_key() == position.recompute_zobrist_key());
        CHECK(position.has_consistent_occupancies());
    }

    for (std::size_t remaining = moves.size(); remaining > 0U; --remaining) {
        const auto index = remaining - 1U;
        position.unmake_move(moves[index], states[index]);
        CHECK(position.zobrist_key() == position.recompute_zobrist_key());
        CHECK(position.has_consistent_occupancies());
    }

    check_restored(position, original_fen, original_key);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("deterministic legal random walk unmake restores original position") {
    auto position = eng::Position::startpos();
    const auto original_fen = position.to_fen();
    const auto original_key = position.zobrist_key();
    std::vector<eng::Move> moves;
    std::vector<eng::StateInfo> states;
    std::uint64_t seed = random_walk_seed;

    for (int ply = 0; ply < random_walk_plies; ++ply) {
        const auto legal_moves = eng::generate_legal_moves(position);
        REQUIRE(!legal_moves.empty());
        const auto move = legal_moves[next_index(seed, legal_moves.size())];
        eng::StateInfo state;
        REQUIRE(position.make_move(move, state));
        moves.push_back(move);
        states.push_back(state);
        CHECK(position.zobrist_key() == position.recompute_zobrist_key());
        CHECK(position.has_consistent_occupancies());
    }

    for (std::size_t remaining = moves.size(); remaining > 0U; --remaining) {
        const auto index = remaining - 1U;
        position.unmake_move(moves[index], states[index]);
        CHECK(position.zobrist_key() == position.recompute_zobrist_key());
        CHECK(position.has_consistent_occupancies());
    }

    check_restored(position, original_fen, original_key);
}

TEST_CASE("make and unmake restore captures promotions en-passant and castling") {
    struct Case {
        std::string_view fen;
        eng::Move move;
    };

    const std::array cases{
        Case{.fen = "4k3/8/8/3p4/4P3/8/8/4K3 w - - 7 12",
             .move = eng::Move::normal(square(file_e, rank_4), square(file_d, rank_5))},
        Case{.fen = "4k3/6P1/8/8/8/8/8/4K3 w - - 4 9",
             .move = eng::Move::promotion(square(file_g, rank_7), square(file_g, rank_8),
                                          eng::PieceType::Queen)},
        Case{.fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 3",
             .move = eng::Move::en_passant(square(file_e, rank_5), square(file_d, rank_6))},
        Case{.fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 10 20",
             .move = eng::Move::castling(square(file_e, rank_1), square(file_g, rank_1))},
    };

    for (const auto& test_case : cases) {
        auto position = parse(test_case.fen);
        const auto original_key = position.zobrist_key();
        eng::StateInfo state;

        REQUIRE(position.make_move(test_case.move, state));
        CHECK(position.zobrist_key() == position.recompute_zobrist_key());
        CHECK(position.has_consistent_occupancies());

        position.unmake_move(test_case.move, state);
        check_restored(position, test_case.fen, original_key);
    }
}

TEST_CASE("make_move updates counters castling rights en-passant and board state") {
    auto double_push = eng::Position::startpos();
    eng::StateInfo double_push_state;
    REQUIRE(double_push.make_move(eng::Move::normal(square(file_e, rank_2), square(file_e, rank_4)),
                                  double_push_state));
    CHECK(double_push.en_passant_square() == square(file_e, rank_3));
    CHECK(double_push.halfmove_clock() == 0);
    CHECK(double_push.fullmove_number() == 1);

    auto capture_rook = parse("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 3 9");
    eng::StateInfo capture_state;
    REQUIRE(capture_rook.make_move(
        eng::Move::normal(square(file_a, rank_1), square(file_a, rank_8)), capture_state));
    CHECK(!capture_rook.castling_rights().white_queen_side());
    CHECK(capture_rook.castling_rights().white_king_side());
    CHECK(capture_rook.castling_rights().black_king_side());
    CHECK(!capture_rook.castling_rights().black_queen_side());
    CHECK(capture_rook.halfmove_clock() == 0);
}

TEST_CASE("make and unmake null move restore reversible state exactly") {
    auto position = parse("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 3");
    const auto original_fen = position.to_fen();
    const auto original_key = position.zobrist_key();
    eng::StateInfo state;

    REQUIRE(position.make_null_move(state));
    CHECK(position.to_fen() == "4k3/8/8/3pP3/8/8/8/4K3 b - - 0 3");
    CHECK(position.zobrist_key() == position.recompute_zobrist_key());
    CHECK(position.has_consistent_occupancies());

    position.unmake_null_move(state);
    check_restored(position, original_fen, original_key);
}

TEST_CASE("make_move rejects moves that leave the own king in check") {
    struct IllegalCase {
        std::string_view fen;
        eng::Move move;
    };

    const std::array cases{
        IllegalCase{.fen = "4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1",
                    .move = eng::Move::normal(square(file_e, rank_2), square(file_d, rank_2))},
        IllegalCase{.fen = "4k3/8/8/8/8/8/3r4/4K3 w - - 0 1",
                    .move = eng::Move::normal(square(file_e, rank_1), square(file_d, rank_1))},
        IllegalCase{.fen = "4k3/8/8/r2pP2K/8/8/8/8 w - d6 0 1",
                    .move = eng::Move::en_passant(square(file_e, rank_5), square(file_d, rank_6))},
        IllegalCase{.fen = "r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1",
                    .move = eng::Move::castling(square(file_e, rank_1), square(file_g, rank_1))},
    };

    for (const auto& test_case : cases) {
        auto position = parse(test_case.fen);
        const auto original_key = position.zobrist_key();
        eng::StateInfo state;

        CHECK(!position.make_move(test_case.move, state));
        check_restored(position, test_case.fen, original_key);
    }
}

TEST_CASE("legal move filtering removes pseudo-legal pinned and unsafe castling moves") {
    const auto pinned = parse("4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1");
    const auto pinned_pseudo = eng::generate_moves(pinned);
    const auto pinned_legal = eng::generate_legal_moves(pinned);
    const auto pinned_move = eng::Move::normal(square(file_e, rank_2), square(file_d, rank_2));

    CHECK(contains_move(pinned_pseudo, pinned_move));
    CHECK(!contains_move(pinned_legal, pinned_move));
    CHECK(pinned.to_fen() == "4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1");

    const auto unsafe_castling = parse("r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1");
    const auto castle = eng::Move::castling(square(file_e, rank_1), square(file_g, rank_1));
    CHECK(contains_move(eng::generate_quiets(unsafe_castling), castle));
    CHECK(!contains_move(eng::generate_legal_moves(unsafe_castling), castle));
}

TEST_CASE("legal move helpers detect checkmate and stalemate without mutating the position") {
    auto checkmate = parse("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    const auto checkmate_fen = checkmate.to_fen();
    const auto checkmate_key = checkmate.zobrist_key();
    CHECK(eng::is_checkmate(checkmate));
    CHECK(!eng::has_legal_moves(checkmate));
    check_restored(checkmate, checkmate_fen, checkmate_key);

    auto stalemate = parse("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    const auto stalemate_fen = stalemate.to_fen();
    const auto stalemate_key = stalemate.zobrist_key();
    CHECK(eng::is_stalemate(stalemate));
    CHECK(!eng::has_legal_moves(stalemate));
    check_restored(stalemate, stalemate_fen, stalemate_key);
}

// NOLINTEND(readability-identifier-length)
