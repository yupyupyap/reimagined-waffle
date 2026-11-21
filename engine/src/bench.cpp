#include <eng/bench.hpp>

#include <eng/bitboard.hpp>
#include <eng/position.hpp>
#include <eng/search.hpp>
#include <eng/tt.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <ostream>
#include <string_view>

namespace eng {
namespace {

constexpr int success = 0;
constexpr int bench_error = 3;
constexpr std::uint64_t milliseconds_per_second = 1000;

struct BenchPosition {
    std::string_view fen;
};

constexpr std::array bench_positions{
    BenchPosition{
        .fen = Position::startpos_fen,
    },
    BenchPosition{
        .fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w "
               "KQkq - 0 1",
    },
    BenchPosition{
        .fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    },
    BenchPosition{
        .fen = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq "
               "- 0 1",
    },
    BenchPosition{
        .fen = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    },
    BenchPosition{
        .fen = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/"
               "R4RK1 w - - 0 10",
    },
};

[[nodiscard]] int clamp_bench_depth(int depth) noexcept {
    if (depth < 1) {
        return 1;
    }
    if (depth >= max_search_ply) {
        return max_search_ply - 1;
    }
    return depth;
}

[[nodiscard]] std::uint64_t nodes_per_second(std::uint64_t nodes,
                                             std::int64_t elapsed_ms) noexcept {
    if (elapsed_ms <= 0) {
        return nodes;
    }
    return (nodes * milliseconds_per_second) / static_cast<std::uint64_t>(elapsed_ms);
}

BenchResult run_bench_impl(int depth, std::ostream* detail_output) {
    init_bitboards();

    const auto search_depth = clamp_bench_depth(depth);
    BenchResult result{
        .positions = static_cast<int>(bench_positions.size()),
        .depth = search_depth,
        .nodes = 0,
        .elapsed_ms = 0,
        .nps = 0,
    };

    const auto started_at = std::chrono::steady_clock::now();
    int position_index = 0;
    TranspositionTable transposition_table;
    const auto tt_resized = transposition_table.resize_mb(16);
    (void)tt_resized;
    for (const auto& bench_position : bench_positions) {
        ++position_index;
        const auto parsed = Position::from_fen(bench_position.fen);
        if (!parsed.has_value()) {
            result.positions = -1;
            return result;
        }

        transposition_table.new_search();
        SearchLimits limits{.depth = Depth::plies(search_depth)};
        limits.transposition_table = &transposition_table;
        const auto search_result = search(*parsed, limits);
        result.nodes += search_result.nodes;
        if (detail_output != nullptr) {
            *detail_output << "position " << position_index << '/' << result.positions << " nodes "
                           << search_result.nodes << '\n';
        }
    }
    result.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - started_at)
                            .count();
    result.nps = nodes_per_second(result.nodes, result.elapsed_ms);
    return result;
}

} // namespace

BenchResult run_bench(int depth) { return run_bench_impl(depth, nullptr); }

void write_bench(std::ostream& output, const BenchResult& result) {
    output << "LocalChessEngine bench\n";
    output << "positions: " << result.positions << '\n';
    output << "depth: " << result.depth << '\n';
    output << "time ms: " << result.elapsed_ms << '\n';
    output << "nodes: " << result.nodes << '\n';
    output << "nps: " << result.nps << '\n';
    output << result.nodes << " nodes " << result.nps << " nps\n";
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int run_bench_cli(std::ostream& output, std::ostream& error) {
    output << "LocalChessEngine bench\n";
    const auto result = run_bench_impl(bench_default_depth, &output);
    if (result.positions < 0) {
        error << "bench failed: invalid built-in FEN\n";
        return bench_error;
    }

    output << "positions: " << result.positions << '\n';
    output << "depth: " << result.depth << '\n';
    output << "time ms: " << result.elapsed_ms << '\n';
    output << "nodes: " << result.nodes << '\n';
    output << "nps: " << result.nps << '\n';
    output << result.nodes << " nodes " << result.nps << " nps\n";
    return success;
}

} // namespace eng
