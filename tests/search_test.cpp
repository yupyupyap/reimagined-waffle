#include <eng/movepick.hpp>
#include <eng/search.hpp>
#include <eng/tt.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

// Square names such as e1 and h8 are clearer than expanded identifiers in chess tests.
// NOLINTBEGIN(readability-identifier-length)

namespace {

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
constexpr int rank_6 = 5;
constexpr int rank_7 = 6;
constexpr int rank_8 = 7;
constexpr int limit_test_depth = 6;
constexpr int aspiration_test_depth = 5;
constexpr int pruning_test_depth = 5;
constexpr int iir_test_depth = 5;
constexpr int singular_test_depth = 8;
constexpr int singular_tt_score_cp = 50;
constexpr int multi_cut_tt_score_cp = 1000;
constexpr int future_deadline_seconds = 30;
constexpr std::uint64_t lazy_smp_node_limit = 512;

[[nodiscard]] constexpr eng::Square square(int file_index, int rank_index) {
    return eng::Square::from_file_rank(eng::File::from_index(file_index),
                                       eng::Rank::from_index(rank_index));
}

[[nodiscard]] eng::Position parse(std::string_view fen) {
    const auto position = eng::Position::from_fen(fen);
    REQUIRE(position.has_value());
    return *position;
}

[[nodiscard]] eng::SearchLimits depth(int plies) {
    return eng::SearchLimits{.depth = eng::Depth::plies(plies)};
}

struct IterationLog {
    std::vector<eng::SearchResult> results;
};

void record_iteration(const eng::SearchResult& result, void* userdata) {
    static_cast<IterationLog*>(userdata)->results.push_back(result);
}

[[nodiscard]] bool pv_is_legal(eng::Position position, const eng::SearchResult& result) {
    for (std::size_t index = 0; index < result.principal_variation_size; ++index) {
        eng::StateInfo state;
        if (!position.make_move(result.principal_variation[index], state)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool move_is_one_of(eng::Move move, std::span<const eng::Move> moves) noexcept {
    return std::ranges::any_of(moves, [move](eng::Move candidate) { return move == candidate; });
}

} // namespace

TEST_CASE("search finds a mate in one") {
    const auto position = parse("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2");

    const auto result = eng::search(position, depth(1));

    CHECK(result.best_move == eng::Move::normal(square(file_d, rank_8), square(file_h, rank_4)));
    CHECK(result.score.value() > 29000);
    CHECK(result.nodes > 0);
    REQUIRE(result.principal_variation_size >= 1U);
    CHECK(result.principal_variation[0] == result.best_move);
}

TEST_CASE("search prefers an obvious queen capture at depth one") {
    const auto position = parse("4k3/8/8/8/4q3/8/4R3/4K3 w - - 0 1");

    const auto result = eng::search(position, depth(1));

    CHECK(result.best_move == eng::Move::normal(square(file_e, rank_2), square(file_e, rank_4)));
    CHECK(result.score.value() > 400);
}

TEST_CASE("root move restrictions limit searched candidates") {
    const auto only_move = eng::Move::normal(square(file_g, rank_1), square(file_f, rank_3));
    const std::array root_moves{only_move};

    const auto result =
        eng::search(eng::Position::startpos(), depth(1),
                    std::span<const eng::Move>{root_moves.data(), root_moves.size()});

    CHECK(result.best_move == only_move);
    REQUIRE(result.principal_variation_size >= 1U);
    CHECK(result.principal_variation[0] == only_move);
}

TEST_CASE("search returns no move for checkmate and stalemate") {
    const auto checkmate = parse("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    const auto mate_result = eng::search(checkmate, depth(1));
    CHECK(mate_result.best_move == eng::Move::none());
    CHECK(mate_result.score.value() < -29000);

    const auto stalemate = parse("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    const auto stalemate_result = eng::search(stalemate, depth(1));
    CHECK(stalemate_result.best_move == eng::Move::none());
    CHECK(stalemate_result.score.value() == 0);
}

TEST_CASE("search does not mutate the input position") {
    const auto position = eng::Position::startpos();
    const auto original_fen = position.to_fen();
    const auto original_key = position.zobrist_key();

    const auto result = eng::search(position, depth(2));

    CHECK(result.best_move.is_valid());
    CHECK(position.to_fen() == original_fen);
    CHECK(position.zobrist_key() == original_key);
    CHECK(position.zobrist_key() == position.recompute_zobrist_key());
    CHECK(position.has_consistent_occupancies());
}

TEST_CASE("transposition table reduces repeated fixed-depth search work") {
    const auto position = parse("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w "
                                "KQkq - 0 1");
    const auto original_fen = position.to_fen();
    const auto original_key = position.zobrist_key();

    eng::TranspositionTable table;
    REQUIRE(table.resize_mb(1));

    auto limits = depth(3);
    limits.transposition_table = &table;
    table.new_search();
    const auto first = eng::search(position, limits);

    table.new_search();
    const auto second = eng::search(position, limits);

    CHECK(first.best_move.is_valid());
    CHECK(second.best_move.is_valid());
    CHECK(second.best_move == first.best_move);
    CHECK(second.nodes < first.nodes);
    REQUIRE(second.principal_variation_size >= 1U);
    CHECK(second.principal_variation[0] == second.best_move);
    CHECK(position.to_fen() == original_fen);
    CHECK(position.zobrist_key() == original_key);
}

TEST_CASE("qsearch does not expand losing captures after stand pat") {
    const auto position = parse("k7/4q3/8/4P3/3P4/8/8/6K1 w - - 0 1");
    const std::array root_moves{
        eng::Move::normal(square(file_g, rank_1), square(file_h, rank_1)),
    };

    const auto result = eng::search(
        position, depth(1), std::span<const eng::Move>{root_moves.data(), root_moves.size()});

    CHECK(result.best_move == root_moves.front());
    CHECK(result.nodes == 1);
}

TEST_CASE("search respects a root node limit and keeps a legal move") {
    auto limits = depth(limit_test_depth);
    limits.nodes = 1;
    limits.poll_interval = 1;
    limits.iterative = true;

    const auto result = eng::search(eng::Position::startpos(), limits);

    CHECK(result.stopped);
    CHECK(result.stop_reason == eng::SearchStopReason::NodeLimit);
    CHECK(result.nodes == 1);
    CHECK(result.best_move.is_valid());
    REQUIRE(result.principal_variation_size >= 1U);
    CHECK(result.principal_variation[0] == result.best_move);
}

TEST_CASE("search responds to external stop and hard deadline limits") {
    std::atomic_bool stop_requested{true};
    auto stopped_limits = depth(limit_test_depth);
    stopped_limits.stop_requested = &stop_requested;
    stopped_limits.poll_interval = 1;
    stopped_limits.iterative = true;

    const auto stopped_result = eng::search(eng::Position::startpos(), stopped_limits);

    CHECK(stopped_result.stopped);
    CHECK(stopped_result.stop_reason == eng::SearchStopReason::ExternalStop);
    CHECK(stopped_result.best_move.is_valid());

    auto timed_limits = depth(limit_test_depth);
    timed_limits.hard_deadline = std::chrono::steady_clock::now() - std::chrono::milliseconds{1};
    timed_limits.has_hard_deadline = true;
    timed_limits.poll_interval = 1;
    timed_limits.iterative = true;

    const auto timed_result = eng::search(eng::Position::startpos(), timed_limits);

    CHECK(timed_result.stopped);
    CHECK(timed_result.stop_reason == eng::SearchStopReason::HardDeadline);
    CHECK(timed_result.best_move.is_valid());
}

TEST_CASE("lazy smp returns a legal coordinator pv with shared transposition table") {
    eng::TranspositionTable table;
    REQUIRE(table.resize_mb(1));
    table.new_search();

    auto limits = depth(3);
    limits.iterative = true;
    limits.threads = 4;
    limits.transposition_table = &table;

    const auto position = eng::Position::startpos();
    const auto result = eng::search(position, limits);

    CHECK_FALSE(result.stopped);
    CHECK(result.depth == eng::Depth::plies(3));
    CHECK(result.best_move.is_valid());
    CHECK(result.nodes > 0);
    CHECK(result.hashfull_permill >= 0);
    CHECK(result.hashfull_permill <= 1000);
    CHECK(pv_is_legal(position, result));
}

TEST_CASE("lazy smp coordinates node and external stop limits") {
    auto node_limits = depth(limit_test_depth);
    node_limits.iterative = true;
    node_limits.threads = 4;
    node_limits.nodes = lazy_smp_node_limit;
    node_limits.poll_interval = 1;

    const auto node_result = eng::search(eng::Position::startpos(), node_limits);

    CHECK(node_result.stopped);
    CHECK(node_result.stop_reason == eng::SearchStopReason::NodeLimit);
    CHECK(node_result.nodes >= node_limits.nodes);
    CHECK(node_result.best_move.is_valid());

    std::atomic_bool stop_requested{true};
    auto stopped_limits = depth(limit_test_depth);
    stopped_limits.iterative = true;
    stopped_limits.threads = 4;
    stopped_limits.stop_requested = &stop_requested;
    stopped_limits.poll_interval = 1;

    const auto stopped_result = eng::search(eng::Position::startpos(), stopped_limits);

    CHECK(stopped_result.stopped);
    CHECK(stopped_result.stop_reason == eng::SearchStopReason::ExternalStop);
    CHECK(stopped_result.best_move.is_valid());
}

TEST_CASE("iterative search stops on soft deadline after stable best move") {
    const auto position = parse("4k3/8/8/8/4q3/8/4R3/4K3 w - - 0 1");
    const std::array root_moves{
        eng::Move::normal(square(file_e, rank_2), square(file_e, rank_4)),
        eng::Move::normal(square(file_e, rank_1), square(file_d, rank_1)),
    };
    auto limits = depth(limit_test_depth);
    limits.soft_deadline = std::chrono::steady_clock::now() - std::chrono::milliseconds{1};
    limits.hard_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds{future_deadline_seconds};
    limits.has_soft_deadline = true;
    limits.has_hard_deadline = true;
    limits.poll_interval = 1;
    limits.iterative = true;

    const auto result = eng::search(
        position, limits, std::span<const eng::Move>{root_moves.data(), root_moves.size()});

    CHECK(result.stopped);
    CHECK(result.stop_reason == eng::SearchStopReason::SoftDeadline);
    CHECK(result.best_move == root_moves.front());
    CHECK(result.best_move_stability >= 2U);
}

// doctest assertion macros intentionally inflate this test's measured complexity.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("iterative search callback reports each completed depth with legal pv") {
    const auto position = eng::Position::startpos();
    IterationLog log;
    auto limits = depth(3);
    limits.iterative = true;
    limits.iteration_callback = record_iteration;
    limits.iteration_callback_userdata = &log;

    const auto result = eng::search(position, limits);

    CHECK_FALSE(result.stopped);
    CHECK(result.depth == eng::Depth::plies(3));
    REQUIRE(log.results.size() == 3U);
    for (std::size_t index = 0; index < log.results.size(); ++index) {
        CHECK(log.results[index].depth == eng::Depth::plies(static_cast<int>(index + 1U)));
        REQUIRE(log.results[index].principal_variation_size >= 1U);
        CHECK(log.results[index].principal_variation[0] == log.results[index].best_move);
        CHECK(pv_is_legal(position, log.results[index]));
        CHECK(log.results[index].bound == eng::SearchScoreBound::Exact);
        CHECK(log.results[index].nps > 0U);
        CHECK(log.results[index].hashfull_permill >= 0);
        CHECK(log.results[index].hashfull_permill <= 1000);
    }
    CHECK(pv_is_legal(position, result));
    CHECK(result.elapsed_ms >= 0);
    CHECK(result.bound == eng::SearchScoreBound::Exact);
    CHECK(result.nps > 0U);
    CHECK(result.hashfull_permill >= 0);
    CHECK(result.hashfull_permill <= 1000);
}

// doctest assertion macros intentionally inflate this test's measured complexity.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("iterative search keeps root move restrictions across completed depths") {
    const std::array root_moves{
        eng::Move::normal(square(file_g, rank_1), square(file_f, rank_3)),
        eng::Move::normal(square(file_b, rank_1), square(file_c, rank_3)),
    };
    IterationLog log;
    auto limits = depth(3);
    limits.iterative = true;
    limits.iteration_callback = record_iteration;
    limits.iteration_callback_userdata = &log;

    const auto result =
        eng::search(eng::Position::startpos(), limits,
                    std::span<const eng::Move>{root_moves.data(), root_moves.size()});

    CHECK_FALSE(result.stopped);
    REQUIRE(log.results.size() == 3U);
    for (const auto& iteration : log.results) {
        REQUIRE(iteration.principal_variation_size >= 1U);
        CHECK(move_is_one_of(iteration.principal_variation[0],
                             std::span<const eng::Move>{root_moves.data(), root_moves.size()}));
    }
    CHECK(move_is_one_of(result.best_move,
                         std::span<const eng::Move>{root_moves.data(), root_moves.size()}));
}

TEST_CASE("aspiration windows record fail high or low research") {
    auto limits = depth(aspiration_test_depth);
    limits.iterative = true;

    const auto result = eng::search(eng::Position::startpos(), limits);

    CHECK_FALSE(result.stopped);
    CHECK(result.depth == eng::Depth::plies(5));
    CHECK(result.aspiration_fail_lows + result.aspiration_fail_highs > 0U);
    CHECK(result.bound == eng::SearchScoreBound::Exact);
    CHECK(pv_is_legal(eng::Position::startpos(), result));
}

TEST_CASE("iterative search reports stability for a single legal root candidate") {
    const auto only_move = eng::Move::normal(square(file_g, rank_1), square(file_f, rank_3));
    const std::array root_moves{only_move};
    auto limits = depth(limit_test_depth);
    limits.iterative = true;

    const auto result =
        eng::search(eng::Position::startpos(), limits,
                    std::span<const eng::Move>{root_moves.data(), root_moves.size()});

    CHECK_FALSE(result.stopped);
    CHECK(result.stop_reason == eng::SearchStopReason::None);
    CHECK(result.best_move == only_move);
    CHECK(result.depth == eng::Depth::plies(1));
    CHECK(result.best_move_stability == 1U);
}

TEST_CASE("search updates injected move ordering state on quiet cutoffs") {
    auto ordering = std::make_unique<eng::MoveOrderingState>();
    auto limits = depth(4);
    limits.move_ordering_state = ordering.get();

    const auto position = eng::Position::startpos();
    const auto result = eng::search(position, limits);

    CHECK_FALSE(result.stopped);
    CHECK(result.best_move.is_valid());
    CHECK(pv_is_legal(position, result));
    CHECK(ordering->has_any_killer());
    CHECK(ordering->has_any_history());
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("individual pruning toggles collect stats and keep a legal pv") {
    struct PruningCase {
        eng::SearchPruningOptions options;
        std::uint64_t eng::SearchPruningStats::* counter;
    };

    const std::array cases{
        PruningCase{.options = eng::SearchPruningOptions{.null_move_pruning = true},
                    .counter = &eng::SearchPruningStats::null_move_cutoffs},
        PruningCase{.options = eng::SearchPruningOptions{.late_move_reductions = true},
                    .counter = &eng::SearchPruningStats::late_move_reductions},
        PruningCase{.options = eng::SearchPruningOptions{.reverse_futility_pruning = true},
                    .counter = &eng::SearchPruningStats::reverse_futility_prunes},
        PruningCase{.options = eng::SearchPruningOptions{.late_move_pruning = true},
                    .counter = &eng::SearchPruningStats::late_move_prunes},
        PruningCase{.options = eng::SearchPruningOptions{.futility_pruning = true},
                    .counter = &eng::SearchPruningStats::futility_prunes},
        PruningCase{.options = eng::SearchPruningOptions{.see_pruning = true},
                    .counter = &eng::SearchPruningStats::see_prunes},
    };

    const auto position = eng::Position::startpos();
    for (const auto& test_case : cases) {
        eng::SearchPruningStats stats;
        auto limits = depth(pruning_test_depth);
        limits.pruning = test_case.options;
        limits.pruning_stats = &stats;

        const auto result = eng::search(position, limits);

        CHECK_FALSE(result.stopped);
        CHECK(result.best_move.is_valid());
        CHECK(pv_is_legal(position, result));
        CHECK((result.pruning_stats.*test_case.counter) > 0U);
        CHECK((stats.*test_case.counter) == (result.pruning_stats.*test_case.counter));
    }
}

TEST_CASE("advanced search extensions stay inactive by default") {
    eng::SearchExtensionStats stats;
    auto limits = depth(4);
    limits.extension_stats = &stats;

    const auto position = eng::Position::startpos();
    const auto result = eng::search(position, limits);

    CHECK_FALSE(result.stopped);
    CHECK(result.best_move.is_valid());
    CHECK(pv_is_legal(position, result));
    CHECK(result.extension_stats == eng::SearchExtensionStats{});
    CHECK(stats == result.extension_stats);
}

TEST_CASE("internal iterative reductions collect stats without a tt move") {
    eng::SearchExtensionStats stats;
    auto limits = depth(iir_test_depth);
    limits.extensions.internal_iterative_reductions = true;
    limits.extension_stats = &stats;

    const auto position = eng::Position::startpos();
    const auto result = eng::search(position, limits);

    CHECK_FALSE(result.stopped);
    CHECK(result.best_move.is_valid());
    CHECK(pv_is_legal(position, result));
    CHECK(result.extension_stats.internal_iterative_reductions > 0U);
    CHECK(stats.internal_iterative_reductions ==
          result.extension_stats.internal_iterative_reductions);
}

TEST_CASE("check extensions collect stats on a checking root move") {
    eng::SearchExtensionStats stats;
    auto limits = depth(1);
    limits.extensions.check_extensions = true;
    limits.extension_stats = &stats;

    const auto position = parse("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1");
    const auto checking_move = eng::Move::normal(square(file_e, rank_2), square(file_e, rank_7));
    const std::array root_moves{checking_move};

    const auto result = eng::search(
        position, limits, std::span<const eng::Move>{root_moves.data(), root_moves.size()});

    CHECK_FALSE(result.stopped);
    CHECK(result.best_move == checking_move);
    CHECK(pv_is_legal(position, result));
    CHECK(result.extension_stats.check_extensions > 0U);
    CHECK(stats.check_extensions == result.extension_stats.check_extensions);
}

TEST_CASE("singular search uses a deep enough tt move without taking a tt cutoff") {
    eng::TranspositionTable table;
    REQUIRE(table.resize_mb(1));
    table.new_search();

    const auto position = parse("4k3/8/8/8/8/8/8/4K2R w - - 0 1");
    const auto root_move = eng::Move::normal(square(file_h, rank_1), square(file_h, rank_2));
    auto child = position;
    eng::StateInfo state;
    REQUIRE(child.make_move(root_move, state));

    const auto tt_move = eng::Move::normal(square(file_e, rank_8), square(file_d, rank_8));
    table.save(child.zobrist_key(), tt_move, eng::Score::centipawns(singular_tt_score_cp),
               eng::Score::centipawns(0), eng::Depth::plies(4), eng::TranspositionBound::Lower,
               eng::Ply::from_root(1), false);

    eng::SearchExtensionStats stats;
    auto limits = depth(singular_test_depth);
    limits.transposition_table = &table;
    limits.extensions.singular_extensions = true;
    limits.extension_stats = &stats;
    const std::array root_moves{root_move};

    const auto result = eng::search(
        position, limits, std::span<const eng::Move>{root_moves.data(), root_moves.size()});

    CHECK_FALSE(result.stopped);
    CHECK(result.best_move == root_move);
    CHECK(pv_is_legal(position, result));
    CHECK(result.extension_stats.singular_searches > 0U);
    CHECK(stats.singular_searches == result.extension_stats.singular_searches);
}

TEST_CASE("multi-cut requires singular verification fail-high") {
    eng::TranspositionTable table;
    REQUIRE(table.resize_mb(1));
    table.new_search();

    const auto position = eng::Position::startpos();
    const auto first_root_move = eng::Move::normal(square(file_b, rank_1), square(file_c, rank_3));
    const auto second_root_move = eng::Move::normal(square(file_g, rank_1), square(file_f, rank_3));

    auto child = position;
    eng::StateInfo state;
    REQUIRE(child.make_move(second_root_move, state));

    const auto tt_move = eng::Move::normal(square(file_b, rank_8), square(file_c, rank_6));
    table.save(child.zobrist_key(), tt_move, eng::Score::centipawns(multi_cut_tt_score_cp),
               eng::Score::centipawns(0), eng::Depth::plies(4), eng::TranspositionBound::Lower,
               eng::Ply::from_root(1), false);

    eng::SearchExtensionStats stats;
    auto limits = depth(singular_test_depth);
    limits.transposition_table = &table;
    limits.extensions.multi_cut = true;
    limits.extension_stats = &stats;
    const std::array root_moves{first_root_move, second_root_move};

    const auto result = eng::search(
        position, limits, std::span<const eng::Move>{root_moves.data(), root_moves.size()});

    CHECK_FALSE(result.stopped);
    CHECK(result.best_move.is_valid());
    CHECK(pv_is_legal(position, result));
    CHECK(result.extension_stats.singular_searches > 0U);
    CHECK(result.extension_stats.multi_cut_cutoffs == 0U);
    CHECK(stats == result.extension_stats);
}

TEST_CASE("combined pruning preserves mate and terminal positions") {
    const eng::SearchPruningOptions all_pruning{.null_move_pruning = true,
                                                .late_move_reductions = true,
                                                .reverse_futility_pruning = true,
                                                .late_move_pruning = true,
                                                .futility_pruning = true,
                                                .see_pruning = true};
    auto limits = depth(pruning_test_depth);
    limits.pruning = all_pruning;

    const auto mate_in_one = parse("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2");
    const auto mate_result = eng::search(mate_in_one, limits);
    CHECK(mate_result.best_move ==
          eng::Move::normal(square(file_d, rank_8), square(file_h, rank_4)));
    CHECK(mate_result.score.value() > 29000);
    CHECK(pv_is_legal(mate_in_one, mate_result));

    const auto checkmate = parse("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    const auto checkmate_result = eng::search(checkmate, limits);
    CHECK(checkmate_result.best_move == eng::Move::none());
    CHECK(checkmate_result.score.value() < -29000);

    const auto stalemate = parse("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    const auto stalemate_result = eng::search(stalemate, limits);
    CHECK(stalemate_result.best_move == eng::Move::none());
    CHECK(stalemate_result.score.value() == 0);
}

TEST_CASE("combined advanced extensions preserve terminal positions and stop limits") {
    const eng::SearchExtensionOptions all_extensions{.singular_extensions = true,
                                                     .multi_cut = true,
                                                     .internal_iterative_reductions = true,
                                                     .check_extensions = true};
    auto limits = depth(pruning_test_depth);
    limits.extensions = all_extensions;

    const auto mate_in_one = parse("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2");
    const auto mate_result = eng::search(mate_in_one, limits);
    CHECK(mate_result.best_move ==
          eng::Move::normal(square(file_d, rank_8), square(file_h, rank_4)));
    CHECK(mate_result.score.value() > 29000);
    CHECK(pv_is_legal(mate_in_one, mate_result));

    const auto checkmate = parse("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    const auto checkmate_result = eng::search(checkmate, limits);
    CHECK(checkmate_result.best_move == eng::Move::none());
    CHECK(checkmate_result.score.value() < -29000);

    const auto stalemate = parse("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    const auto stalemate_result = eng::search(stalemate, limits);
    CHECK(stalemate_result.best_move == eng::Move::none());
    CHECK(stalemate_result.score.value() == 0);

    auto node_limits = depth(limit_test_depth);
    node_limits.extensions = all_extensions;
    node_limits.iterative = true;
    node_limits.nodes = 1;
    node_limits.poll_interval = 1;

    const auto node_result = eng::search(eng::Position::startpos(), node_limits);
    CHECK(node_result.stopped);
    CHECK(node_result.stop_reason == eng::SearchStopReason::NodeLimit);
    CHECK(node_result.best_move.is_valid());
}

// NOLINTEND(readability-identifier-length)
