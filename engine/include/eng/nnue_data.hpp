#ifndef ENG_NNUE_DATA_HPP
#define ENG_NNUE_DATA_HPP

#include "eng/move.hpp"
#include "eng/position.hpp"
#include "eng/search.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace eng {

inline constexpr std::string_view nnue_dataset_format = "localchessengine.nnue.dataset.v1";
inline constexpr std::string_view nnue_dataset_manifest_event = "trainer.dataset.manifest";
inline constexpr std::uint16_t nnue_wdl_permille_total = 1000;
inline constexpr std::uint32_t nnue_data_default_positions = 16;
inline constexpr std::uint32_t nnue_data_default_games = 4;
inline constexpr std::uint32_t nnue_data_default_max_plies = 48;
inline constexpr std::uint64_t nnue_data_default_seed = 27027;
inline constexpr int nnue_data_default_min_ply = 8;
inline constexpr int nnue_data_default_max_abs_eval_cp = 2500;
inline constexpr int nnue_data_default_adjudication_draw_cp = 10;
inline constexpr std::uint16_t nnue_data_default_random_playout_permille = 150;
inline constexpr std::uint16_t nnue_data_default_top_n_playout_permille = 350;
inline constexpr std::uint32_t nnue_data_default_top_n_playout = 4;
inline constexpr std::uint32_t nnue_data_unlimited_records_per_game = 0;
inline constexpr std::uint32_t nnue_data_default_accepted_sample_stride = 1;
inline constexpr std::uint32_t nnue_relabel_default_train_positions = 1000;
inline constexpr std::uint32_t nnue_relabel_default_holdout_positions = 200;
inline constexpr std::uint32_t nnue_relabel_default_match_openings = 64;
inline constexpr std::uint32_t nnue_relabel_default_hash_mb = 16;
inline constexpr std::uint16_t nnue_relabel_default_holdout_permille = 150;
inline constexpr std::uint16_t nnue_relabel_default_match_permille = 100;
inline constexpr std::uint32_t nnue_relabel_default_min_pieces = 6;
inline constexpr std::uint32_t nnue_relabel_default_max_pieces = 32;
inline constexpr int nnue_relabel_default_max_stability_delta_cp = 150;

struct NnueWdlResult {
    std::uint16_t white_win_permille{0};
    std::uint16_t draw_permille{nnue_wdl_permille_total};
    std::uint16_t black_win_permille{0};

    friend bool operator==(const NnueWdlResult& lhs, const NnueWdlResult& rhs) = default;
};

struct NnueTrainingRecord {
    std::string fen;
    Move best_move{Move::none()};
    int eval_cp{0};
    NnueWdlResult wdl;
    std::uint16_t ply{0};
    std::uint16_t depth{0};
    std::uint64_t nodes{0};
    ZobristKey zobrist_key{0};
    std::uint32_t game_index{0};

    friend bool operator==(const NnueTrainingRecord& lhs, const NnueTrainingRecord& rhs) = default;
};

struct NnueDataset {
    std::vector<NnueTrainingRecord> records;
    std::uint64_t checksum{0};
};

enum class NnueDataFilterReason : std::uint8_t {
    Accepted = 0,
    OpeningPly,
    InCheck,
    BestMoveCapture,
    EvalOutOfRange,
    TerminalOrNoBestMove
};

[[nodiscard]] std::string_view nnue_data_filter_reason_name(NnueDataFilterReason reason) noexcept;

enum class NnueDataPlayoutPolicy : std::uint8_t {
    Best = 0,
    Random,
    Mixed,
};

[[nodiscard]] std::string_view nnue_data_playout_policy_name(NnueDataPlayoutPolicy policy) noexcept;

struct NnueDatasetStats {
    std::uint64_t games_started{0};
    std::uint64_t games_completed{0};
    std::uint64_t plies_played{0};
    std::uint64_t positions_considered{0};
    std::uint64_t accepted{0};
    std::uint64_t filtered_opening_ply{0};
    std::uint64_t filtered_in_check{0};
    std::uint64_t filtered_best_move_capture{0};
    std::uint64_t filtered_eval_out_of_range{0};
    std::uint64_t filtered_terminal_or_no_bestmove{0};
    std::uint64_t sampled_out_by_stride{0};
    std::uint64_t sampled_out_by_game_limit{0};

    [[nodiscard]] std::uint64_t filtered() const noexcept;
};

struct NnueDataGenerationConfig {
    std::uint32_t positions{nnue_data_default_positions};
    std::uint32_t games{nnue_data_default_games};
    std::uint32_t max_plies{nnue_data_default_max_plies};
    Depth depth{Depth::plies(1)};
    std::uint64_t seed{nnue_data_default_seed};
    int min_ply{nnue_data_default_min_ply};
    int max_abs_eval_cp{nnue_data_default_max_abs_eval_cp};
    int adjudication_draw_cp{nnue_data_default_adjudication_draw_cp};
    NnueDataPlayoutPolicy playout_policy{NnueDataPlayoutPolicy::Best};
    std::uint16_t random_playout_permille{nnue_data_default_random_playout_permille};
    std::uint16_t top_n_playout_permille{nnue_data_default_top_n_playout_permille};
    std::uint32_t top_n_playout{nnue_data_default_top_n_playout};
    std::uint32_t max_records_per_game{nnue_data_unlimited_records_per_game};
    std::uint32_t accepted_sample_stride{nnue_data_default_accepted_sample_stride};
};

struct NnueDataGenerationResult {
    std::vector<NnueTrainingRecord> records;
    NnueDatasetStats stats;
    std::uint64_t checksum{0};
};

enum class NnueFenRelabelFilterReason : std::uint8_t {
    Accepted = 0,
    InvalidFen,
    DuplicateFen,
    SplitFull,
    Material,
    InCheck,
    BestMoveCapture,
    EvalOutOfRange,
    TerminalOrNoBestMove,
    UnstableEval
};

[[nodiscard]] std::string_view
nnue_fen_relabel_filter_reason_name(NnueFenRelabelFilterReason reason) noexcept;

struct NnueFenRelabelStats {
    std::uint64_t source_fens_seen{0};
    std::uint64_t accepted_train{0};
    std::uint64_t accepted_holdout{0};
    std::uint64_t accepted_match_openings{0};
    std::uint64_t filtered_invalid_fen{0};
    std::uint64_t filtered_duplicate_fen{0};
    std::uint64_t filtered_split_full{0};
    std::uint64_t filtered_material{0};
    std::uint64_t filtered_in_check{0};
    std::uint64_t filtered_best_move_capture{0};
    std::uint64_t filtered_eval_out_of_range{0};
    std::uint64_t filtered_terminal_or_no_bestmove{0};
    std::uint64_t filtered_unstable_eval{0};

    [[nodiscard]] std::uint64_t accepted() const noexcept;
    [[nodiscard]] std::uint64_t filtered() const noexcept;
};

struct NnueFenRelabelConfig {
    std::uint32_t train_positions{nnue_relabel_default_train_positions};
    std::uint32_t holdout_positions{nnue_relabel_default_holdout_positions};
    std::uint32_t match_openings{nnue_relabel_default_match_openings};
    Depth depth{Depth::plies(3)};
    std::uint64_t nodes{0};
    std::uint32_t hash_mb{nnue_relabel_default_hash_mb};
    std::uint16_t holdout_permille{nnue_relabel_default_holdout_permille};
    std::uint16_t match_permille{nnue_relabel_default_match_permille};
    int max_abs_eval_cp{nnue_data_default_max_abs_eval_cp};
    std::uint32_t min_pieces{nnue_relabel_default_min_pieces};
    std::uint32_t max_pieces{nnue_relabel_default_max_pieces};
    std::uint32_t stability_depth_delta{0};
    int max_stability_delta_cp{nnue_relabel_default_max_stability_delta_cp};
    bool skip_in_check{true};
    bool skip_best_move_capture{true};
};

struct NnueFenRelabelResult {
    std::vector<NnueTrainingRecord> train_records;
    std::vector<NnueTrainingRecord> holdout_records;
    std::vector<NnueTrainingRecord> match_opening_records;
    NnueFenRelabelStats stats;
    std::uint64_t train_checksum{0};
    std::uint64_t holdout_checksum{0};
    std::uint64_t match_openings_checksum{0};
};

enum class NnueDataErrorCode : std::uint8_t {
    Io = 0,
    InvalidMagic,
    UnsupportedVersion,
    Truncated,
    InvalidRecord,
    ChecksumMismatch
};

struct NnueDataError {
    NnueDataErrorCode code{NnueDataErrorCode::Io};
    std::string message;

    friend bool operator==(const NnueDataError& lhs, const NnueDataError& rhs) = default;
};

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] NnueDataFilterReason
nnue_data_filter_reason(const Position& position, Move best_move, int white_relative_eval_cp,
                        int ply, const NnueDataGenerationConfig& config) noexcept;

[[nodiscard]] std::uint64_t
nnue_dataset_checksum(std::span<const NnueTrainingRecord> records) noexcept;
[[nodiscard]] NnueDataGenerationResult
generate_nnue_dataset(const NnueDataGenerationConfig& config);
[[nodiscard]] NnueFenRelabelResult relabel_nnue_fens(std::span<const std::string> fens,
                                                     const NnueFenRelabelConfig& config);

[[nodiscard]] std::expected<void, NnueDataError>
write_nnue_dataset(const std::filesystem::path& path, std::span<const NnueTrainingRecord> records);
[[nodiscard]] std::expected<NnueDataset, NnueDataError>
read_nnue_dataset(const std::filesystem::path& path);

} // namespace eng

#endif // ENG_NNUE_DATA_HPP
