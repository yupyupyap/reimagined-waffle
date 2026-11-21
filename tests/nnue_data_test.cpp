#include <eng/nnue_data.hpp>

#include <eng/uci.hpp>

#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

constexpr int file_e = 4;
constexpr int file_g = 6;
constexpr int rank_1 = 0;
constexpr int rank_2 = 1;
constexpr int rank_3 = 2;
constexpr int rank_4 = 3;
constexpr int first_trainable_ply = 8;
constexpr int out_of_range_eval = 3000;
constexpr int sample_eval_cp = 12;
constexpr std::uint16_t sample_win_permille = 250;
constexpr std::uint16_t sample_draw_permille = 500;
constexpr std::uint64_t sample_nodes = 20;

[[nodiscard]] constexpr eng::Square square(int file_index, int rank_index) {
    return eng::Square::from_file_rank(eng::File::from_index(file_index),
                                       eng::Rank::from_index(rank_index));
}

[[nodiscard]] eng::Position parse(std::string_view fen) {
    const auto position = eng::Position::from_fen(fen);
    REQUIRE(position.has_value());
    return *position;
}

[[nodiscard]] std::filesystem::path temp_path(std::string_view suffix) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("localchessengine-nnue-data-" + std::to_string(now) + std::string{suffix});
}

[[nodiscard]] eng::NnueTrainingRecord sample_record() {
    const auto position = eng::Position::startpos();
    return eng::NnueTrainingRecord{
        .fen = position.to_fen(),
        .best_move = eng::Move::normal(square(file_g, rank_1), square(file_g, rank_3)),
        .eval_cp = sample_eval_cp,
        .wdl = eng::NnueWdlResult{.white_win_permille = sample_win_permille,
                                  .draw_permille = sample_draw_permille,
                                  .black_win_permille = sample_win_permille},
        .ply = first_trainable_ply,
        .depth = 1,
        .nodes = sample_nodes,
        .zobrist_key = position.zobrist_key(),
        .game_index = 1,
    };
}

} // namespace

TEST_CASE("NNUE dataset binary round trip preserves records and checksum") {
    const auto path = temp_path(".binpack");
    const std::vector records{sample_record()};

    const auto written = eng::write_nnue_dataset(path, records);
    REQUIRE(written.has_value());

    const auto read = eng::read_nnue_dataset(path);
    REQUIRE(read.has_value());
    REQUIRE(read->records.size() == records.size());
    CHECK(read->records.front() == records.front());
    CHECK(read->checksum == eng::nnue_dataset_checksum(records));
    CHECK(eng::move_to_uci(read->records.front().best_move) == "g1g3");

    std::filesystem::remove(path);
}

TEST_CASE("NNUE dataset reader rejects invalid magic and truncation") {
    const auto valid_path = temp_path("-valid.binpack");
    const auto bad_magic_path = temp_path("-bad-magic.binpack");
    const auto truncated_path = temp_path("-truncated.binpack");
    const std::vector records{sample_record()};
    REQUIRE(eng::write_nnue_dataset(valid_path, records).has_value());

    {
        std::ifstream input(valid_path, std::ios::binary);
        std::ofstream output(bad_magic_path, std::ios::binary);
        output << input.rdbuf();
    }
    {
        std::fstream file(bad_magic_path, std::ios::in | std::ios::out | std::ios::binary);
        file.put('X');
    }
    {
        std::ofstream output(truncated_path, std::ios::binary);
        output.put('L');
        output.put('C');
        output.put('E');
    }

    const auto bad_magic = eng::read_nnue_dataset(bad_magic_path);
    REQUIRE(!bad_magic.has_value());
    CHECK(bad_magic.error().code == eng::NnueDataErrorCode::InvalidMagic);

    const auto truncated = eng::read_nnue_dataset(truncated_path);
    REQUIRE(!truncated.has_value());
    CHECK(truncated.error().code == eng::NnueDataErrorCode::Truncated);

    std::filesystem::remove(valid_path);
    std::filesystem::remove(bad_magic_path);
    std::filesystem::remove(truncated_path);
}

TEST_CASE("NNUE data filters return deterministic first reasons") {
    const eng::NnueDataGenerationConfig config;
    const auto start = eng::Position::startpos();
    const auto quiet = eng::Move::normal(square(file_g, rank_1), square(file_g, rank_3));
    CHECK(eng::nnue_data_filter_reason(start, quiet, 0, 0, config) ==
          eng::NnueDataFilterReason::OpeningPly);

    const auto check = parse("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1");
    CHECK(eng::nnue_data_filter_reason(check, quiet, 0, first_trainable_ply, config) ==
          eng::NnueDataFilterReason::InCheck);

    const auto capture = parse("4k3/8/8/8/4q3/8/4R3/4K3 w - - 0 1");
    const auto rook_takes_queen = eng::Move::normal(square(file_e, rank_2), square(file_e, rank_4));
    CHECK(eng::nnue_data_filter_reason(capture, rook_takes_queen, 0, first_trainable_ply, config) ==
          eng::NnueDataFilterReason::BestMoveCapture);

    CHECK(eng::nnue_data_filter_reason(start, quiet, out_of_range_eval, first_trainable_ply,
                                       config) == eng::NnueDataFilterReason::EvalOutOfRange);
    CHECK(eng::nnue_data_filter_reason(start, eng::Move::none(), 0, first_trainable_ply, config) ==
          eng::NnueDataFilterReason::TerminalOrNoBestMove);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("NNUE data generation creates a small filtered smoke dataset") {
    const eng::NnueDataGenerationConfig config{
        .positions = 3,
        .games = 2,
        .max_plies = 24,
        .depth = eng::Depth::plies(1),
        .seed = 27027,
    };

    const auto result = eng::generate_nnue_dataset(config);

    CHECK(result.records.size() == config.positions);
    CHECK(result.stats.accepted >= config.positions);
    CHECK(result.stats.accepted + result.stats.filtered() == result.stats.positions_considered);
    CHECK(result.stats.filtered_opening_ply >= 8U);
    CHECK(result.stats.positions_considered >= result.records.size());
    CHECK(result.checksum == eng::nnue_dataset_checksum(result.records));
    for (const auto& record : result.records) {
        CHECK(record.ply >= first_trainable_ply);
        CHECK(record.depth == 1);
        CHECK(record.best_move.is_valid());
        CHECK(std::abs(record.eval_cp) <= config.max_abs_eval_cp);
        CHECK(record.wdl.white_win_permille + record.wdl.draw_permille +
                  record.wdl.black_win_permille ==
              eng::nnue_wdl_permille_total);
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("NNUE data generation supports mixed playout and per-game sampling") {
    const eng::NnueDataGenerationConfig config{
        .positions = 4,
        .games = 16,
        .max_plies = 48,
        .depth = eng::Depth::plies(1),
        .seed = 34034,
        .playout_policy = eng::NnueDataPlayoutPolicy::Mixed,
        .random_playout_permille = 500,
        .top_n_playout_permille = 500,
        .top_n_playout = 4,
        .max_records_per_game = 1,
        .accepted_sample_stride = 2,
    };

    const auto result = eng::generate_nnue_dataset(config);

    CHECK(result.records.size() == config.positions);
    CHECK(eng::nnue_data_playout_policy_name(config.playout_policy) == "mixed");
    std::vector<std::uint32_t> records_per_game(config.games);
    for (const auto& record : result.records) {
        if (record.game_index < records_per_game.size()) {
            ++records_per_game[static_cast<std::size_t>(record.game_index)];
        }
    }
    for (const auto& record : result.records) {
        CHECK(records_per_game[static_cast<std::size_t>(record.game_index)] <= 1);
        CHECK(((record.ply - first_trainable_ply) % config.accepted_sample_stride) == 0);
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("NNUE FEN relabeling filters invalid and duplicate positions") {
    const std::vector<std::string> fens{
        "not a fen",
        std::string{eng::Position::startpos_fen},
        std::string{eng::Position::startpos_fen},
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    };
    eng::NnueFenRelabelConfig config;
    config.train_positions = 2;
    config.holdout_positions = 0;
    config.match_openings = 0;
    config.depth = eng::Depth::plies(1);
    config.holdout_permille = 0;
    config.match_permille = 0;
    config.min_pieces = 2;

    const auto result = eng::relabel_nnue_fens(fens, config);

    CHECK(result.train_records.size() == 2U);
    CHECK(result.holdout_records.empty());
    CHECK(result.match_opening_records.empty());
    CHECK(result.stats.filtered_invalid_fen == 1U);
    CHECK(result.stats.filtered_duplicate_fen == 1U);
    CHECK(result.train_checksum == eng::nnue_dataset_checksum(result.train_records));
    std::unordered_set<std::string> unique_fens;
    for (const auto& record : result.train_records) {
        CHECK(unique_fens.insert(record.fen).second);
        CHECK(record.depth == 1);
        CHECK(record.best_move.is_valid());
        CHECK(std::abs(record.eval_cp) <= config.max_abs_eval_cp);
        CHECK(record.wdl.white_win_permille + record.wdl.draw_permille +
                  record.wdl.black_win_permille ==
              eng::nnue_wdl_permille_total);
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("NNUE FEN relabeling can produce match-opening split only") {
    const std::vector<std::string> fens{
        std::string{eng::Position::startpos_fen},
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    };
    eng::NnueFenRelabelConfig config;
    config.train_positions = 0;
    config.holdout_positions = 0;
    config.match_openings = 1;
    config.depth = eng::Depth::plies(1);
    config.holdout_permille = 0;
    config.match_permille = eng::nnue_wdl_permille_total;
    config.min_pieces = 2;

    const auto result = eng::relabel_nnue_fens(fens, config);

    CHECK(result.train_records.empty());
    CHECK(result.holdout_records.empty());
    REQUIRE(result.match_opening_records.size() == 1U);
    CHECK(result.stats.accepted_match_openings == 1U);
    CHECK(result.match_openings_checksum ==
          eng::nnue_dataset_checksum(result.match_opening_records));
}
