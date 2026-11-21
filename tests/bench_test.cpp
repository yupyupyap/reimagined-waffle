#include <eng/bench.hpp>

#include <doctest/doctest.h>

#include <sstream>
#include <string>

TEST_CASE("bench returns deterministic node count for fixed depth") {
    const auto first = eng::run_bench(1);
    const auto second = eng::run_bench(1);

    CHECK(first.positions == eng::bench_position_count);
    CHECK(first.depth == 1);
    CHECK(first.nodes > 0);
    CHECK(first.nodes == second.nodes);
    CHECK(second.positions == eng::bench_position_count);
    CHECK(second.depth == 1);
}

TEST_CASE("bench default keeps pruning toggles disabled") {
    const auto result = eng::run_bench();

    CHECK(result.positions == eng::bench_position_count);
    CHECK(result.depth == eng::bench_default_depth);
    CHECK(result.nodes == 18417);
}

TEST_CASE("bench output ends with OpenBench parseable nodes and nps line") {
    const auto result = eng::run_bench(1);

    std::ostringstream output;
    eng::write_bench(output, result);

    const auto text = output.str();
    const auto expected_final_line =
        std::to_string(result.nodes) + " nodes " + std::to_string(result.nps) + " nps\n";
    CHECK(text.ends_with(expected_final_line));
    CHECK(text.contains("positions: 6\n"));
    CHECK(text.contains("depth: 1\n"));
    CHECK(text.contains("time ms: "));
}
