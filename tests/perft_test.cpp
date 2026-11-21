#include <eng/movegen.hpp>
#include <eng/perft.hpp>
#include <eng/position.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

namespace {

constexpr int max_fast_gate_depth = 4;
constexpr std::size_t reference_depth_count = 7;
using OptionalPerftNodeCount = std::optional<eng::PerftNodeCount>;

struct PerftReference {
    std::string_view name;
    std::string_view fen;
    std::array<OptionalPerftNodeCount, reference_depth_count> nodes;
};

constexpr std::array cpw_references{
    PerftReference{
        .name = "Initial position",
        .fen = eng::Position::startpos_fen,
        .nodes = {1, 20, 400, 8902, 197281, 4865609, 119060324},
    },
    PerftReference{
        .name = "Position 2 Kiwipete",
        .fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w "
               "KQkq - 0 1",
        .nodes = {1, 48, 2039, 97862, 4085603, 193690690, 8031647685ULL},
    },
    PerftReference{
        .name = "Position 3",
        .fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        .nodes = {1, 14, 191, 2812, 43238, 674624, 11030083},
    },
    PerftReference{
        .name = "Position 4",
        .fen = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq "
               "- 0 1",
        .nodes = {1, 6, 264, 9467, 422333, 15833292, 706045033},
    },
    PerftReference{
        .name = "Position 5",
        .fen = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        .nodes = {1, 44, 1486, 62379, 2103487, 89941194, std::nullopt},
    },
    PerftReference{
        .name = "Position 6",
        .fen = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/"
               "R4RK1 w - - 0 10",
        .nodes = {1, 46, 2079, 89890, 3894594, 164075551, 6923051137ULL},
    },
};

[[nodiscard]] eng::Position parse(std::string_view fen) {
    const auto position = eng::Position::from_fen(fen);
    REQUIRE(position.has_value());
    return *position;
}

[[nodiscard]] bool contains_divide_entry(const std::vector<eng::PerftDivideEntry>& entries,
                                         eng::Move move, eng::PerftNodeCount nodes) {
    return std::ranges::find(entries, eng::PerftDivideEntry{.move = move, .nodes = nodes}) !=
           entries.end();
}

} // namespace

TEST_CASE("perft returns one node at depth zero without mutating position") {
    auto position = eng::Position::startpos();
    const auto original_fen = position.to_fen();
    const auto original_key = position.zobrist_key();

    CHECK(eng::perft(position, 0) == 1);
    CHECK(position.to_fen() == original_fen);
    CHECK(position.zobrist_key() == original_key);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("perft matches CPW standard positions through fast gate depth") {
    for (const auto& reference : cpw_references) {
        CAPTURE(reference.name);
        for (int depth = 0; depth <= max_fast_gate_depth; ++depth) {
            auto position = parse(reference.fen);
            CAPTURE(depth);
            const auto expected = reference.nodes[static_cast<std::size_t>(depth)];
            REQUIRE(expected.has_value());
            CHECK(eng::perft(position, depth) == expected.value_or(0));
            CHECK(position.to_fen() == reference.fen);
            CHECK(position.zobrist_key() == position.recompute_zobrist_key());
            CHECK(position.has_consistent_occupancies());
        }
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("divide entries sum to perft and expose root move counts") {
    constexpr int divide_depth = 2;
    auto position = eng::Position::startpos();
    const auto entries = eng::divide(position, divide_depth);
    eng::PerftNodeCount total = 0;
    for (const auto& entry : entries) {
        total += entry.nodes;
    }

    CHECK(entries.size() == 20U);
    CHECK(total == eng::perft(position, divide_depth));
    CHECK(total == 400);
    CHECK(contains_divide_entry(
        entries, eng::Move::normal(eng::Square::from_index(8), eng::Square::from_index(16)), 20));
    CHECK(contains_divide_entry(
        entries, eng::Move::normal(eng::Square::from_index(12), eng::Square::from_index(28)), 20));
    CHECK(contains_divide_entry(
        entries, eng::Move::normal(eng::Square::from_index(1), eng::Square::from_index(18)), 20));
    CHECK(position.to_fen() == eng::Position::startpos_fen);
    CHECK(position.zobrist_key() == position.recompute_zobrist_key());
}
