#include <eng/nnue_data.hpp>

#include <eng/eval.hpp>
#include <eng/movegen.hpp>
#include <eng/movepick.hpp>
#include <eng/tt.hpp>
#include <eng/uci.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

namespace eng {
namespace {

constexpr std::array<char, 16> dataset_magic{'L', 'C', 'E', 'N', 'N', 'U', 'E', 'D',
                                             'A', 'T', 'A', 'S', 'E', 'T', '1', '\0'};
constexpr std::uint32_t dataset_version = 1;
constexpr std::uint64_t checksum_offset_basis = 14695981039346656037ULL;
constexpr std::uint64_t checksum_prime = 1099511628211ULL;
constexpr int bits_per_byte = 8;
constexpr int checksum_u16_bits = 16;
constexpr int checksum_u32_bits = 32;
constexpr int checksum_u64_bits = 64;
constexpr unsigned byte_mask = 0xFFU;
constexpr int max_dataset_depth = max_search_ply - 1;
constexpr std::uint64_t rng_increment = 0x9e3779b97f4a7c15ULL;
constexpr std::uint64_t rng_mix_1 = 0xbf58476d1ce4e5b9ULL;
constexpr std::uint64_t rng_mix_2 = 0x94d049bb133111ebULL;
constexpr unsigned rng_shift_1 = 30U;
constexpr unsigned rng_shift_2 = 27U;
constexpr unsigned rng_shift_3 = 31U;
constexpr int relabel_wdl_scale_cp = 400;
constexpr int chess_pawns_per_side = File::count;

void mix_checksum_byte(std::uint64_t& hash, std::uint8_t value) noexcept {
    hash ^= value;
    hash *= checksum_prime;
}

void mix_checksum_u16(std::uint64_t& hash, std::uint16_t value) noexcept {
    for (int shift = 0; shift < checksum_u16_bits; shift += bits_per_byte) {
        mix_checksum_byte(hash, static_cast<std::uint8_t>(value >> shift));
    }
}

void mix_checksum_u32(std::uint64_t& hash, std::uint32_t value) noexcept {
    for (int shift = 0; shift < checksum_u32_bits; shift += bits_per_byte) {
        mix_checksum_byte(hash, static_cast<std::uint8_t>(value >> shift));
    }
}

void mix_checksum_i32(std::uint64_t& hash, int value) noexcept {
    mix_checksum_u32(hash, static_cast<std::uint32_t>(value));
}

void mix_checksum_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
    for (int shift = 0; shift < checksum_u64_bits; shift += bits_per_byte) {
        mix_checksum_byte(hash, static_cast<std::uint8_t>(value >> shift));
    }
}

void mix_checksum_string(std::uint64_t& hash, std::string_view value) noexcept {
    mix_checksum_u64(hash, value.size());
    for (const auto symbol : value) {
        mix_checksum_byte(hash, static_cast<std::uint8_t>(symbol));
    }
}

[[nodiscard]] NnueDataError make_error(NnueDataErrorCode code, std::string message) {
    return NnueDataError{.code = code, .message = std::move(message)};
}

void write_u16(std::ostream& output, std::uint16_t value) {
    for (int shift = 0; shift < checksum_u16_bits; shift += bits_per_byte) {
        output.put(static_cast<char>((value >> shift) & byte_mask));
    }
}

void write_u32(std::ostream& output, std::uint32_t value) {
    for (int shift = 0; shift < checksum_u32_bits; shift += bits_per_byte) {
        output.put(static_cast<char>((value >> shift) & byte_mask));
    }
}

void write_i32(std::ostream& output, int value) {
    write_u32(output, static_cast<std::uint32_t>(value));
}

void write_u64(std::ostream& output, std::uint64_t value) {
    for (int shift = 0; shift < checksum_u64_bits; shift += bits_per_byte) {
        output.put(static_cast<char>((value >> shift) & byte_mask));
    }
}

[[nodiscard]] std::expected<std::uint8_t, NnueDataError> read_u8(std::istream& input) {
    char value = '\0';
    if (!input.get(value)) {
        return std::unexpected(make_error(NnueDataErrorCode::Truncated, "unexpected end of file"));
    }
    return static_cast<std::uint8_t>(value);
}

[[nodiscard]] std::expected<std::uint16_t, NnueDataError> read_u16(std::istream& input) {
    std::uint16_t value = 0;
    for (int shift = 0; shift < checksum_u16_bits; shift += bits_per_byte) {
        const auto byte = read_u8(input);
        if (!byte.has_value()) {
            return std::unexpected(byte.error());
        }
        value = static_cast<std::uint16_t>(value | static_cast<std::uint16_t>(*byte << shift));
    }
    return value;
}

[[nodiscard]] std::expected<std::uint32_t, NnueDataError> read_u32(std::istream& input) {
    std::uint32_t value = 0;
    for (int shift = 0; shift < checksum_u32_bits; shift += bits_per_byte) {
        const auto byte = read_u8(input);
        if (!byte.has_value()) {
            return std::unexpected(byte.error());
        }
        value |= static_cast<std::uint32_t>(*byte) << shift;
    }
    return value;
}

[[nodiscard]] std::expected<int, NnueDataError> read_i32(std::istream& input) {
    const auto value = read_u32(input);
    if (!value.has_value()) {
        return std::unexpected(value.error());
    }
    return static_cast<int>(*value);
}

[[nodiscard]] std::expected<std::uint64_t, NnueDataError> read_u64(std::istream& input) {
    std::uint64_t value = 0;
    for (int shift = 0; shift < checksum_u64_bits; shift += bits_per_byte) {
        const auto byte = read_u8(input);
        if (!byte.has_value()) {
            return std::unexpected(byte.error());
        }
        value |= static_cast<std::uint64_t>(*byte) << shift;
    }
    return value;
}

[[nodiscard]] bool ensure_parent_directory(const std::filesystem::path& path) {
    const auto parent = path.parent_path();
    if (parent.empty()) {
        return true;
    }

    std::error_code error;
    std::filesystem::create_directories(parent, error);
    return !error;
}

[[nodiscard]] bool is_capture_move(const Position& position, Move move) noexcept {
    return move.is_valid() && (move.is_en_passant() || position.piece_on(move.to()).is_valid());
}

void increment_filter_stat(NnueDatasetStats& stats, NnueDataFilterReason reason) noexcept {
    switch (reason) {
    case NnueDataFilterReason::Accepted:
        ++stats.accepted;
        break;
    case NnueDataFilterReason::OpeningPly:
        ++stats.filtered_opening_ply;
        break;
    case NnueDataFilterReason::InCheck:
        ++stats.filtered_in_check;
        break;
    case NnueDataFilterReason::BestMoveCapture:
        ++stats.filtered_best_move_capture;
        break;
    case NnueDataFilterReason::EvalOutOfRange:
        ++stats.filtered_eval_out_of_range;
        break;
    case NnueDataFilterReason::TerminalOrNoBestMove:
        ++stats.filtered_terminal_or_no_bestmove;
        break;
    }
}

void increment_relabel_filter_stat(NnueFenRelabelStats& stats,
                                   NnueFenRelabelFilterReason reason) noexcept {
    switch (reason) {
    case NnueFenRelabelFilterReason::Accepted:
        break;
    case NnueFenRelabelFilterReason::InvalidFen:
        ++stats.filtered_invalid_fen;
        break;
    case NnueFenRelabelFilterReason::DuplicateFen:
        ++stats.filtered_duplicate_fen;
        break;
    case NnueFenRelabelFilterReason::SplitFull:
        ++stats.filtered_split_full;
        break;
    case NnueFenRelabelFilterReason::Material:
        ++stats.filtered_material;
        break;
    case NnueFenRelabelFilterReason::InCheck:
        ++stats.filtered_in_check;
        break;
    case NnueFenRelabelFilterReason::BestMoveCapture:
        ++stats.filtered_best_move_capture;
        break;
    case NnueFenRelabelFilterReason::EvalOutOfRange:
        ++stats.filtered_eval_out_of_range;
        break;
    case NnueFenRelabelFilterReason::TerminalOrNoBestMove:
        ++stats.filtered_terminal_or_no_bestmove;
        break;
    case NnueFenRelabelFilterReason::UnstableEval:
        ++stats.filtered_unstable_eval;
        break;
    }
}

[[nodiscard]] std::uint64_t splitmix64_next(std::uint64_t& state) noexcept {
    state += rng_increment;
    auto value = state;
    value = (value ^ (value >> rng_shift_1)) * rng_mix_1;
    value = (value ^ (value >> rng_shift_2)) * rng_mix_2;
    return value ^ (value >> rng_shift_3);
}

[[nodiscard]] int clamped_depth(Depth depth) noexcept {
    return std::clamp(depth.value(), 1, max_dataset_depth);
}

[[nodiscard]] int white_relative_score(const Position& position, Score score) noexcept {
    return position.side_to_move() == Color::Black ? -score.value() : score.value();
}

[[nodiscard]] int white_relative_hce(const Position& position) noexcept {
    return white_relative_score(position, evaluate(position));
}

[[nodiscard]] NnueWdlResult logistic_wdl_from_eval_cp(int white_relative_eval_cp) noexcept {
    const auto expectation = 1.0 / (1.0 + std::exp(-static_cast<double>(white_relative_eval_cp) /
                                                   static_cast<double>(relabel_wdl_scale_cp)));
    auto white = static_cast<int>(std::lround(expectation * nnue_wdl_permille_total));
    white = std::clamp(white, 0, static_cast<int>(nnue_wdl_permille_total));
    return NnueWdlResult{
        .white_win_permille = static_cast<std::uint16_t>(white),
        .draw_permille = 0,
        .black_win_permille =
            static_cast<std::uint16_t>(static_cast<int>(nnue_wdl_permille_total) - white),
    };
}

[[nodiscard]] bool relabel_material_supported(const Position& position,
                                              const NnueFenRelabelConfig& config) noexcept {
    const auto total_pieces = static_cast<std::uint32_t>(popcount(position.occupancy()));
    if (total_pieces < config.min_pieces || total_pieces > config.max_pieces) {
        return false;
    }
    if (popcount(position.pieces(Color::White, PieceType::Pawn)) > chess_pawns_per_side ||
        popcount(position.pieces(Color::Black, PieceType::Pawn)) > chess_pawns_per_side) {
        return false;
    }
    return true;
}

enum class RelabelSplit : std::uint8_t { Train = 0, Holdout, MatchOpening };

struct RelabelLabelResult {
    SearchResult label;
    int eval_cp{0};
    NnueFenRelabelFilterReason filter_reason{NnueFenRelabelFilterReason::Accepted};
};

[[nodiscard]] RelabelSplit split_for_relabel_fen(const Position& position,
                                                 const NnueFenRelabelConfig& config) noexcept {
    const auto bucket = position.zobrist_key() % nnue_wdl_permille_total;
    if (bucket < config.match_permille) {
        return RelabelSplit::MatchOpening;
    }
    if (bucket < static_cast<std::uint64_t>(config.match_permille) + config.holdout_permille) {
        return RelabelSplit::Holdout;
    }
    return RelabelSplit::Train;
}

[[nodiscard]] bool relabel_split_full(const NnueFenRelabelResult& result, RelabelSplit split,
                                      const NnueFenRelabelConfig& config) noexcept {
    switch (split) {
    case RelabelSplit::Train:
        return result.train_records.size() >= config.train_positions;
    case RelabelSplit::Holdout:
        return result.holdout_records.size() >= config.holdout_positions;
    case RelabelSplit::MatchOpening:
        return result.match_opening_records.size() >= config.match_openings;
    }
    return true;
}

[[nodiscard]] bool relabel_targets_reached(const NnueFenRelabelResult& result,
                                           const NnueFenRelabelConfig& config) noexcept {
    return result.train_records.size() >= config.train_positions &&
           result.holdout_records.size() >= config.holdout_positions &&
           result.match_opening_records.size() >= config.match_openings;
}

[[nodiscard]] SearchResult run_relabel_search(const Position& position, Depth depth,
                                              std::uint64_t nodes, TranspositionTable& table) {
    table.new_search();
    SearchLimits limits{.depth = Depth::plies(std::clamp(depth.value(), 1, max_dataset_depth))};
    limits.nodes = nodes;
    limits.transposition_table = &table;
    limits.iterative = limits.depth.value() > 1;
    limits.threads = 1;
    return search(position, limits);
}

[[nodiscard]] NnueDataFilterReason
relabel_record_filter_reason(const Position& position, Move best_move, int white_relative_eval_cp,
                             const NnueFenRelabelConfig& relabel_config) noexcept {
    NnueDataGenerationConfig filter_config;
    filter_config.min_ply = 0;
    filter_config.max_abs_eval_cp = relabel_config.max_abs_eval_cp;
    return nnue_data_filter_reason(position, best_move, white_relative_eval_cp, 0, filter_config);
}

[[nodiscard]] NnueFenRelabelFilterReason
to_relabel_filter_reason(NnueDataFilterReason reason) noexcept {
    switch (reason) {
    case NnueDataFilterReason::Accepted:
    case NnueDataFilterReason::OpeningPly:
        return NnueFenRelabelFilterReason::Accepted;
    case NnueDataFilterReason::InCheck:
        return NnueFenRelabelFilterReason::InCheck;
    case NnueDataFilterReason::BestMoveCapture:
        return NnueFenRelabelFilterReason::BestMoveCapture;
    case NnueDataFilterReason::EvalOutOfRange:
        return NnueFenRelabelFilterReason::EvalOutOfRange;
    case NnueDataFilterReason::TerminalOrNoBestMove:
        return NnueFenRelabelFilterReason::TerminalOrNoBestMove;
    }
    return NnueFenRelabelFilterReason::TerminalOrNoBestMove;
}

[[nodiscard]] NnueFenRelabelFilterReason
accepted_relabel_label_filter(const Position& position, Move best_move, int white_relative_eval_cp,
                              const NnueFenRelabelConfig& config) noexcept {
    const auto reason = to_relabel_filter_reason(
        relabel_record_filter_reason(position, best_move, white_relative_eval_cp, config));
    if (reason == NnueFenRelabelFilterReason::BestMoveCapture && !config.skip_best_move_capture) {
        return NnueFenRelabelFilterReason::Accepted;
    }
    return reason;
}

[[nodiscard]] NnueFenRelabelFilterReason
relabel_position_filter_reason(const Position& position,
                               const NnueFenRelabelConfig& config) noexcept {
    if (!relabel_material_supported(position, config)) {
        return NnueFenRelabelFilterReason::Material;
    }
    if (config.skip_in_check && side_to_move_in_check(position)) {
        return NnueFenRelabelFilterReason::InCheck;
    }
    if (!has_legal_moves(position)) {
        return NnueFenRelabelFilterReason::TerminalOrNoBestMove;
    }
    return NnueFenRelabelFilterReason::Accepted;
}

[[nodiscard]] RelabelLabelResult make_relabel_label(const Position& position,
                                                    const NnueFenRelabelConfig& config,
                                                    TranspositionTable& table) {
    RelabelLabelResult result;
    result.label = run_relabel_search(position, config.depth, config.nodes, table);
    result.eval_cp = white_relative_score(position, result.label.score);
    result.filter_reason =
        accepted_relabel_label_filter(position, result.label.best_move, result.eval_cp, config);
    if (result.filter_reason != NnueFenRelabelFilterReason::Accepted) {
        return result;
    }

    if (config.stability_depth_delta != 0U) {
        const auto stable_depth =
            Depth::plies(config.depth.value() + static_cast<int>(config.stability_depth_delta));
        const auto stable_label = run_relabel_search(position, stable_depth, config.nodes, table);
        const auto stable_eval_cp = white_relative_score(position, stable_label.score);
        if (std::abs(stable_eval_cp - result.eval_cp) > config.max_stability_delta_cp) {
            result.filter_reason = NnueFenRelabelFilterReason::UnstableEval;
            return result;
        }
        result.label = stable_label;
        result.eval_cp = stable_eval_cp;
        result.filter_reason =
            accepted_relabel_label_filter(position, result.label.best_move, result.eval_cp, config);
        if (result.filter_reason != NnueFenRelabelFilterReason::Accepted) {
            return result;
        }
    }

    if (!result.label.best_move.is_valid() || result.label.best_move.is_null()) {
        result.filter_reason = NnueFenRelabelFilterReason::TerminalOrNoBestMove;
        return result;
    }
    if (std::abs(result.eval_cp) > config.max_abs_eval_cp) {
        result.filter_reason = NnueFenRelabelFilterReason::EvalOutOfRange;
    }
    return result;
}

void push_relabel_record(NnueFenRelabelResult& result, RelabelSplit split,
                         NnueTrainingRecord record) {
    switch (split) {
    case RelabelSplit::Train:
        record.game_index = static_cast<std::uint32_t>(result.train_records.size());
        result.train_records.push_back(std::move(record));
        ++result.stats.accepted_train;
        break;
    case RelabelSplit::Holdout:
        record.game_index = static_cast<std::uint32_t>(result.holdout_records.size());
        result.holdout_records.push_back(std::move(record));
        ++result.stats.accepted_holdout;
        break;
    case RelabelSplit::MatchOpening:
        record.game_index = static_cast<std::uint32_t>(result.match_opening_records.size());
        result.match_opening_records.push_back(std::move(record));
        ++result.stats.accepted_match_openings;
        break;
    }
}

[[nodiscard]] NnueWdlResult decisive_wdl(Color winner) noexcept {
    if (winner == Color::White) {
        return NnueWdlResult{.white_win_permille = nnue_wdl_permille_total,
                             .draw_permille = 0,
                             .black_win_permille = 0};
    }
    if (winner == Color::Black) {
        return NnueWdlResult{.white_win_permille = 0,
                             .draw_permille = 0,
                             .black_win_permille = nnue_wdl_permille_total};
    }
    return {};
}

[[nodiscard]] NnueWdlResult terminal_wdl(const Position& position) noexcept {
    if (is_checkmate(position)) {
        return decisive_wdl(opposite(position.side_to_move()));
    }
    return {};
}

[[nodiscard]] NnueWdlResult adjudicated_wdl(const Position& position,
                                            int adjudication_draw_cp) noexcept {
    const auto score = white_relative_hce(position);
    if (std::abs(score) <= std::max(0, adjudication_draw_cp)) {
        return {};
    }
    return decisive_wdl(score > 0 ? Color::White : Color::Black);
}

[[nodiscard]] Move choose_playout_move(const MoveList& legal_moves, Move best_move, int ply,
                                       const Position& position,
                                       const NnueDataGenerationConfig& config,
                                       std::uint64_t& rng_state) noexcept {
    if (legal_moves.empty()) {
        return Move::none();
    }

    const auto random_legal = [&]() noexcept {
        const auto index =
            static_cast<std::size_t>(splitmix64_next(rng_state) % legal_moves.size());
        return legal_moves[index];
    };
    const auto ordered_top_n = [&]() noexcept {
        std::array<Move, max_pseudo_legal_moves> candidates{};
        std::size_t count = 0;
        MovePicker picker(position, legal_moves,
                          MovePickerContext{.tt_move = best_move, .ply = Ply::from_root(ply)});
        const auto limit = std::max<std::uint32_t>(config.top_n_playout, 1);
        for (auto move = picker.next();
             move.is_valid() && count < candidates.size() && count < limit; move = picker.next()) {
            candidates[count] = move;
            ++count;
        }
        if (count == 0) {
            return random_legal();
        }
        const auto index = static_cast<std::size_t>(splitmix64_next(rng_state) % count);
        return candidates[index];
    };

    if (config.playout_policy == NnueDataPlayoutPolicy::Random) {
        return random_legal();
    }
    if (config.playout_policy == NnueDataPlayoutPolicy::Mixed) {
        const auto roll =
            static_cast<std::uint16_t>(splitmix64_next(rng_state) % nnue_wdl_permille_total);
        if (roll < config.random_playout_permille) {
            return random_legal();
        }
        if (roll < config.random_playout_permille + config.top_n_playout_permille) {
            return ordered_top_n();
        }
        if (ply >= config.min_ply && best_move.is_valid()) {
            return best_move;
        }
        return ordered_top_n();
    }

    if (ply >= config.min_ply && best_move.is_valid()) {
        return best_move;
    }

    return random_legal();
}

[[nodiscard]] SearchResult run_label_search(const Position& position,
                                            const NnueDataGenerationConfig& config,
                                            TranspositionTable& table) {
    table.new_search();
    SearchLimits limits{.depth = Depth::plies(clamped_depth(config.depth))};
    limits.transposition_table = &table;
    limits.iterative = limits.depth.value() > 1;
    limits.threads = 1;
    return search(position, limits);
}

[[nodiscard]] std::expected<NnueTrainingRecord, NnueDataError> read_record(std::istream& input) {
    const auto fen_size = read_u16(input);
    if (!fen_size.has_value()) {
        return std::unexpected(fen_size.error());
    }

    std::string fen(*fen_size, '\0');
    if (!fen.empty() && !input.read(fen.data(), static_cast<std::streamsize>(fen.size()))) {
        return std::unexpected(
            make_error(NnueDataErrorCode::Truncated, "truncated FEN payload in dataset record"));
    }

    const auto best_move = read_u16(input);
    const auto eval_cp = read_i32(input);
    const auto white_win = read_u16(input);
    const auto draw = read_u16(input);
    const auto black_win = read_u16(input);
    const auto ply = read_u16(input);
    const auto depth = read_u16(input);
    const auto nodes = read_u64(input);
    const auto zobrist_key = read_u64(input);
    const auto game_index = read_u32(input);
    for (const auto* parsed : {&best_move, &white_win, &draw, &black_win, &ply, &depth}) {
        if (!parsed->has_value()) {
            return std::unexpected(parsed->error());
        }
    }
    if (!eval_cp.has_value()) {
        return std::unexpected(eval_cp.error());
    }
    if (!nodes.has_value()) {
        return std::unexpected(nodes.error());
    }
    if (!zobrist_key.has_value()) {
        return std::unexpected(zobrist_key.error());
    }
    if (!game_index.has_value()) {
        return std::unexpected(game_index.error());
    }

    const auto parsed_position = Position::from_fen(fen);
    if (!parsed_position.has_value()) {
        return std::unexpected(
            make_error(NnueDataErrorCode::InvalidRecord, "dataset record contains invalid FEN"));
    }

    const auto wdl_sum =
        static_cast<int>(*white_win) + static_cast<int>(*draw) + static_cast<int>(*black_win);
    if (std::cmp_not_equal(wdl_sum, nnue_wdl_permille_total)) {
        return std::unexpected(make_error(NnueDataErrorCode::InvalidRecord,
                                          "dataset record WDL permille does not sum to 1000"));
    }

    const auto move = Move::from_raw(*best_move);
    if (!move.is_valid() || move.is_null()) {
        return std::unexpected(
            make_error(NnueDataErrorCode::InvalidRecord, "dataset record has invalid best move"));
    }

    return NnueTrainingRecord{
        .fen = std::move(fen),
        .best_move = move,
        .eval_cp = *eval_cp,
        .wdl = NnueWdlResult{.white_win_permille = *white_win,
                             .draw_permille = *draw,
                             .black_win_permille = *black_win},
        .ply = *ply,
        .depth = *depth,
        .nodes = *nodes,
        .zobrist_key = *zobrist_key,
        .game_index = *game_index,
    };
}

void write_record(std::ostream& output, const NnueTrainingRecord& record) {
    write_u16(output, static_cast<std::uint16_t>(record.fen.size()));
    output.write(record.fen.data(), static_cast<std::streamsize>(record.fen.size()));
    write_u16(output, record.best_move.raw());
    write_i32(output, record.eval_cp);
    write_u16(output, record.wdl.white_win_permille);
    write_u16(output, record.wdl.draw_permille);
    write_u16(output, record.wdl.black_win_permille);
    write_u16(output, record.ply);
    write_u16(output, record.depth);
    write_u64(output, record.nodes);
    write_u64(output, record.zobrist_key);
    write_u32(output, record.game_index);
}

} // namespace

std::string_view nnue_data_filter_reason_name(NnueDataFilterReason reason) noexcept {
    switch (reason) {
    case NnueDataFilterReason::Accepted:
        return "accepted";
    case NnueDataFilterReason::OpeningPly:
        return "opening_ply";
    case NnueDataFilterReason::InCheck:
        return "in_check";
    case NnueDataFilterReason::BestMoveCapture:
        return "best_move_capture";
    case NnueDataFilterReason::EvalOutOfRange:
        return "eval_out_of_range";
    case NnueDataFilterReason::TerminalOrNoBestMove:
        return "terminal_or_no_bestmove";
    }
    return "unknown";
}

std::uint64_t NnueDatasetStats::filtered() const noexcept {
    return filtered_opening_ply + filtered_in_check + filtered_best_move_capture +
           filtered_eval_out_of_range + filtered_terminal_or_no_bestmove;
}

std::string_view nnue_data_playout_policy_name(NnueDataPlayoutPolicy policy) noexcept {
    switch (policy) {
    case NnueDataPlayoutPolicy::Best:
        return "best";
    case NnueDataPlayoutPolicy::Random:
        return "random";
    case NnueDataPlayoutPolicy::Mixed:
        return "mixed";
    }
    return "unknown";
}

std::string_view nnue_fen_relabel_filter_reason_name(NnueFenRelabelFilterReason reason) noexcept {
    switch (reason) {
    case NnueFenRelabelFilterReason::Accepted:
        return "accepted";
    case NnueFenRelabelFilterReason::InvalidFen:
        return "invalid_fen";
    case NnueFenRelabelFilterReason::DuplicateFen:
        return "duplicate_fen";
    case NnueFenRelabelFilterReason::SplitFull:
        return "split_full";
    case NnueFenRelabelFilterReason::Material:
        return "material";
    case NnueFenRelabelFilterReason::InCheck:
        return "in_check";
    case NnueFenRelabelFilterReason::BestMoveCapture:
        return "best_move_capture";
    case NnueFenRelabelFilterReason::EvalOutOfRange:
        return "eval_out_of_range";
    case NnueFenRelabelFilterReason::TerminalOrNoBestMove:
        return "terminal_or_no_bestmove";
    case NnueFenRelabelFilterReason::UnstableEval:
        return "unstable_eval";
    }
    return "unknown";
}

std::uint64_t NnueFenRelabelStats::accepted() const noexcept {
    return accepted_train + accepted_holdout + accepted_match_openings;
}

std::uint64_t NnueFenRelabelStats::filtered() const noexcept {
    return filtered_invalid_fen + filtered_duplicate_fen + filtered_split_full + filtered_material +
           filtered_in_check + filtered_best_move_capture + filtered_eval_out_of_range +
           filtered_terminal_or_no_bestmove + filtered_unstable_eval;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
NnueDataFilterReason nnue_data_filter_reason(const Position& position, Move best_move,
                                             int white_relative_eval_cp, int ply,
                                             const NnueDataGenerationConfig& config) noexcept {
    if (!best_move.is_valid() || is_checkmate(position) || is_stalemate(position)) {
        return NnueDataFilterReason::TerminalOrNoBestMove;
    }
    if (ply < config.min_ply) {
        return NnueDataFilterReason::OpeningPly;
    }
    if (side_to_move_in_check(position)) {
        return NnueDataFilterReason::InCheck;
    }
    if (is_capture_move(position, best_move)) {
        return NnueDataFilterReason::BestMoveCapture;
    }
    if (std::abs(white_relative_eval_cp) > config.max_abs_eval_cp) {
        return NnueDataFilterReason::EvalOutOfRange;
    }
    return NnueDataFilterReason::Accepted;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

void maybe_store_training_candidate(NnueDataGenerationResult& result,
                                    std::vector<NnueTrainingRecord>& candidates,
                                    const NnueDataGenerationConfig& config,
                                    const Position& position, const SearchResult& label,
                                    int eval_cp, std::uint32_t ply, std::uint32_t game) {
    increment_filter_stat(result.stats, NnueDataFilterReason::Accepted);
    const auto stride = std::max<std::uint32_t>(config.accepted_sample_stride, 1);
    const auto accepted_offset =
        static_cast<std::uint32_t>(std::max(0, static_cast<int>(ply) - config.min_ply));
    if ((accepted_offset % stride) != 0U) {
        ++result.stats.sampled_out_by_stride;
        return;
    }
    if (config.max_records_per_game != 0U && candidates.size() >= config.max_records_per_game) {
        ++result.stats.sampled_out_by_game_limit;
        return;
    }

    candidates.push_back(NnueTrainingRecord{
        .fen = position.to_fen(),
        .best_move = label.best_move,
        .eval_cp = eval_cp,
        .wdl = {},
        .ply = static_cast<std::uint16_t>(ply),
        .depth = static_cast<std::uint16_t>(clamped_depth(config.depth)),
        .nodes = label.nodes,
        .zobrist_key = position.zobrist_key(),
        .game_index = game,
    });
}

std::uint64_t nnue_dataset_checksum(std::span<const NnueTrainingRecord> records) noexcept {
    auto hash = checksum_offset_basis;
    for (const auto& record : records) {
        mix_checksum_string(hash, record.fen);
        mix_checksum_u16(hash, record.best_move.raw());
        mix_checksum_i32(hash, record.eval_cp);
        mix_checksum_u16(hash, record.wdl.white_win_permille);
        mix_checksum_u16(hash, record.wdl.draw_permille);
        mix_checksum_u16(hash, record.wdl.black_win_permille);
        mix_checksum_u16(hash, record.ply);
        mix_checksum_u16(hash, record.depth);
        mix_checksum_u64(hash, record.nodes);
        mix_checksum_u64(hash, record.zobrist_key);
        mix_checksum_u32(hash, record.game_index);
    }
    return hash;
}

NnueDataGenerationResult generate_nnue_dataset(const NnueDataGenerationConfig& config) {
    init_bitboards();

    NnueDataGenerationResult result;
    result.records.reserve(config.positions);

    TranspositionTable table;
    const auto resized = table.resize_mb(16);
    (void)resized;

    auto rng_state = config.seed;
    for (std::uint32_t game = 0; game < config.games && result.records.size() < config.positions;
         ++game) {
        ++result.stats.games_started;
        auto position = Position::startpos();
        std::vector<NnueTrainingRecord> candidates;
        candidates.reserve(config.max_plies);
        NnueWdlResult game_wdl;
        bool terminal_result = false;

        for (std::uint32_t ply = 0; ply < config.max_plies; ++ply) {
            const auto legal_moves = generate_legal_moves(position);
            ++result.stats.positions_considered;
            if (legal_moves.empty()) {
                increment_filter_stat(result.stats, NnueDataFilterReason::TerminalOrNoBestMove);
                game_wdl = terminal_wdl(position);
                terminal_result = true;
                break;
            }

            const auto label = run_label_search(position, config, table);
            const auto eval_cp = white_relative_score(position, label.score);
            const auto reason = nnue_data_filter_reason(position, label.best_move, eval_cp,
                                                        static_cast<int>(ply), config);
            if (reason == NnueDataFilterReason::Accepted) {
                maybe_store_training_candidate(result, candidates, config, position, label, eval_cp,
                                               ply, game);
            } else {
                increment_filter_stat(result.stats, reason);
            }

            const auto move = choose_playout_move(
                legal_moves, label.best_move, static_cast<int>(ply), position, config, rng_state);
            StateInfo state;
            if (!position.make_move(move, state)) {
                break;
            }
            ++result.stats.plies_played;
        }

        if (!terminal_result) {
            game_wdl = adjudicated_wdl(position, config.adjudication_draw_cp);
        }
        ++result.stats.games_completed;

        for (auto& candidate : candidates) {
            if (result.records.size() >= config.positions) {
                break;
            }
            candidate.wdl = game_wdl;
            result.records.push_back(std::move(candidate));
        }
    }

    result.checksum = nnue_dataset_checksum(result.records);
    return result;
}

NnueFenRelabelResult relabel_nnue_fens(std::span<const std::string> fens,
                                       const NnueFenRelabelConfig& config) {
    init_bitboards();

    NnueFenRelabelResult result;
    result.train_records.reserve(config.train_positions);
    result.holdout_records.reserve(config.holdout_positions);
    result.match_opening_records.reserve(config.match_openings);

    TranspositionTable table;
    const auto resized = table.resize_mb(std::max<std::uint32_t>(config.hash_mb, 1));
    (void)resized;

    std::unordered_set<std::string> seen_fens;
    seen_fens.reserve(fens.size());

    for (const auto& raw_fen : fens) {
        if (relabel_targets_reached(result, config)) {
            break;
        }
        ++result.stats.source_fens_seen;

        auto parsed = Position::from_fen(raw_fen);
        if (!parsed.has_value()) {
            increment_relabel_filter_stat(result.stats, NnueFenRelabelFilterReason::InvalidFen);
            continue;
        }
        auto position = *parsed;
        const auto fen = position.to_fen();
        if (!seen_fens.insert(fen).second) {
            increment_relabel_filter_stat(result.stats, NnueFenRelabelFilterReason::DuplicateFen);
            continue;
        }
        const auto split = split_for_relabel_fen(position, config);
        if (relabel_split_full(result, split, config)) {
            increment_relabel_filter_stat(result.stats, NnueFenRelabelFilterReason::SplitFull);
            continue;
        }
        const auto position_filter = relabel_position_filter_reason(position, config);
        if (position_filter != NnueFenRelabelFilterReason::Accepted) {
            increment_relabel_filter_stat(result.stats, position_filter);
            continue;
        }

        const auto label = make_relabel_label(position, config, table);
        if (label.filter_reason != NnueFenRelabelFilterReason::Accepted) {
            increment_relabel_filter_stat(result.stats, label.filter_reason);
            continue;
        }

        push_relabel_record(
            result, split,
            NnueTrainingRecord{
                .fen = fen,
                .best_move = label.label.best_move,
                .eval_cp = label.eval_cp,
                .wdl = logistic_wdl_from_eval_cp(label.eval_cp),
                .ply = 0,
                .depth = static_cast<std::uint16_t>(std::clamp(
                    config.depth.value() + static_cast<int>(config.stability_depth_delta), 1,
                    max_dataset_depth)),
                .nodes = label.label.nodes,
                .zobrist_key = position.zobrist_key(),
                .game_index = 0,
            });
    }

    result.train_checksum = nnue_dataset_checksum(result.train_records);
    result.holdout_checksum = nnue_dataset_checksum(result.holdout_records);
    result.match_openings_checksum = nnue_dataset_checksum(result.match_opening_records);
    return result;
}

std::expected<void, NnueDataError> write_nnue_dataset(const std::filesystem::path& path,
                                                      std::span<const NnueTrainingRecord> records) {
    if (!ensure_parent_directory(path)) {
        return std::unexpected(
            make_error(NnueDataErrorCode::Io, "could not create dataset parent directory"));
    }
    if (records.size() > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) {
        return std::unexpected(
            make_error(NnueDataErrorCode::InvalidRecord, "dataset contains too many records"));
    }
    for (const auto& record : records) {
        if (record.fen.size() > std::numeric_limits<std::uint16_t>::max()) {
            return std::unexpected(
                make_error(NnueDataErrorCode::InvalidRecord, "record FEN is too long"));
        }
        if (!record.best_move.is_valid() || record.best_move.is_null()) {
            return std::unexpected(
                make_error(NnueDataErrorCode::InvalidRecord, "record best move is invalid"));
        }
    }

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return std::unexpected(make_error(NnueDataErrorCode::Io,
                                          "could not open dataset for writing: " + path.string()));
    }

    output.write(dataset_magic.data(), static_cast<std::streamsize>(dataset_magic.size()));
    write_u32(output, dataset_version);
    write_u64(output, records.size());
    write_u64(output, nnue_dataset_checksum(records));
    for (const auto& record : records) {
        write_record(output, record);
    }

    if (!output) {
        return std::unexpected(make_error(NnueDataErrorCode::Io,
                                          "could not write complete dataset: " + path.string()));
    }
    return {};
}

std::expected<NnueDataset, NnueDataError> read_nnue_dataset(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return std::unexpected(make_error(NnueDataErrorCode::Io,
                                          "could not open dataset for reading: " + path.string()));
    }

    std::array<char, dataset_magic.size()> magic{};
    if (!input.read(magic.data(), static_cast<std::streamsize>(magic.size()))) {
        return std::unexpected(
            make_error(NnueDataErrorCode::Truncated, "dataset is truncated before magic header"));
    }
    if (magic != dataset_magic) {
        return std::unexpected(
            make_error(NnueDataErrorCode::InvalidMagic, "dataset magic does not match"));
    }

    const auto version = read_u32(input);
    if (!version.has_value()) {
        return std::unexpected(version.error());
    }
    if (*version != dataset_version) {
        return std::unexpected(
            make_error(NnueDataErrorCode::UnsupportedVersion, "unsupported dataset version"));
    }

    const auto record_count = read_u64(input);
    const auto expected_checksum = read_u64(input);
    if (!record_count.has_value()) {
        return std::unexpected(record_count.error());
    }
    if (!expected_checksum.has_value()) {
        return std::unexpected(expected_checksum.error());
    }
    if (*record_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return std::unexpected(
            make_error(NnueDataErrorCode::InvalidRecord, "dataset record count is too large"));
    }

    NnueDataset dataset;
    dataset.records.reserve(static_cast<std::size_t>(*record_count));
    for (std::uint64_t index = 0; index < *record_count; ++index) {
        auto record = read_record(input);
        if (!record.has_value()) {
            return std::unexpected(record.error());
        }
        dataset.records.push_back(std::move(*record));
    }

    char trailing = '\0';
    if (input.get(trailing)) {
        return std::unexpected(
            make_error(NnueDataErrorCode::InvalidRecord, "dataset contains trailing bytes"));
    }
    dataset.checksum = nnue_dataset_checksum(dataset.records);
    if (dataset.checksum != *expected_checksum) {
        return std::unexpected(
            make_error(NnueDataErrorCode::ChecksumMismatch, "dataset checksum mismatch"));
    }
    return dataset;
}

} // namespace eng
