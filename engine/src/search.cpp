#include <eng/search.hpp>

#include <eng/bitboard.hpp>
#include <eng/eval.hpp>
#include <eng/movegen.hpp>
#include <eng/movepick.hpp>
#include <eng/nnue.hpp>
#include <eng/see.hpp>
#include <eng/tt.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <pthread/qos.h>
#endif

namespace eng {
namespace {

constexpr int search_infinity = 32000;
constexpr int mate_score = tt_mate_score;
constexpr int max_depth = max_search_ply - 1;
constexpr int aspiration_start_depth = 5;
constexpr int aspiration_initial_delta = 12;
constexpr int aspiration_growth_numerator = 7;
constexpr int aspiration_growth_denominator = 5;
constexpr int aspiration_max_researches = 8;
constexpr int null_move_min_depth = 3;
constexpr int null_move_base_reduction = 4;
constexpr int null_move_depth_divisor = 4;
constexpr int null_move_eval_margin_divisor = 200;
constexpr int null_move_eval_margin_cap = 3;
constexpr int reverse_futility_max_depth = 8;
constexpr int reverse_futility_margin = 80;
constexpr int late_move_pruning_max_depth = 8;
constexpr int late_move_pruning_base = 3;
constexpr int late_move_pruning_not_improving_divisor = 2;
constexpr int futility_max_depth = 8;
constexpr int futility_base_margin = 100;
constexpr int futility_depth_margin = 100;
constexpr int see_capture_margin = 50;
constexpr int see_quiet_margin = 75;
constexpr int history_lmr_scale = 8192;
constexpr int singular_extension_min_depth = 7;
constexpr int singular_extension_tt_depth_margin = 3;
constexpr int singular_extension_beta_margin = 2;
constexpr int internal_iterative_reduction_min_depth = 4;
constexpr std::size_t shared_search_state_alignment = 128;
constexpr std::size_t search_context_alignment = 64;
constexpr std::uint64_t milliseconds_per_second = 1000;

struct PrincipalVariation {
    std::array<Move, max_search_ply> moves{};
    std::size_t size{0};
};

struct alignas(shared_search_state_alignment) SharedSearchState {
    std::atomic<std::uint64_t> nodes{0};
    std::atomic_bool stopped{false};
    std::atomic<std::uint8_t> stop_reason{static_cast<std::uint8_t>(SearchStopReason::None)};
};

struct alignas(search_context_alignment) SearchContext {
    SearchLimits limits;
    std::chrono::steady_clock::time_point started_at;
    SharedSearchState* shared{nullptr};
    std::uint64_t nodes{0};
    std::uint64_t next_poll_node{1};
    std::array<int, max_search_ply> static_evaluations{};
    std::array<NnueAccumulator, max_search_ply> nnue_accumulators{};
    std::array<bool, max_search_ply> nnue_accumulator_valid{};
    SearchPruningStats pruning_stats;
    SearchExtensionStats extension_stats;
    int root_depth{default_search_depth};
    int seldepth{0};
    bool stopped{false};
    bool owns_soft_deadline{true};
    SearchStopReason stop_reason{SearchStopReason::None};
};

struct TTProbe {
    Move move{Move::none()};
    Score score;
    Score static_eval;
    Depth depth;
    TranspositionBound bound{TranspositionBound::None};
    std::optional<int> cutoff_score;
    bool has_entry{false};
};

struct ChildSearch {
    int score{-search_infinity};
    PrincipalVariation line;
};

struct AspirationStats {
    std::uint32_t fail_lows{0};
    std::uint32_t fail_highs{0};
};

[[nodiscard]] constexpr int clamp_depth(int depth) noexcept {
    if (depth < 1) {
        return 1;
    }
    return std::min(depth, max_depth);
}

void set_line(PrincipalVariation& target, Move move, const PrincipalVariation& child) noexcept {
    target.moves = {};
    target.moves[0] = move;
    target.size = 1;

    const auto copy_count = std::min(child.size, target.moves.size() - 1U);
    for (std::size_t index = 0; index < copy_count; ++index) {
        target.moves[index + 1U] = child.moves[index];
    }
    target.size += copy_count;
}

void update_seldepth(SearchContext& context, int ply) noexcept {
    context.seldepth = std::max(context.seldepth, ply);
}

[[nodiscard]] std::uint64_t normalized_poll_interval(const SearchLimits& limits) noexcept {
    return std::max<std::uint64_t>(limits.poll_interval, 1);
}

[[nodiscard]] int normalized_thread_count(const SearchLimits& limits) noexcept {
    return std::max(limits.threads, default_search_threads);
}

[[nodiscard]] SearchContext make_search_context(const SearchLimits& limits,
                                                SharedSearchState* shared = nullptr,
                                                bool owns_soft_deadline = true) noexcept {
    SearchContext context;
    context.limits = limits;
    context.started_at = std::chrono::steady_clock::now();
    context.shared = shared;
    context.next_poll_node = normalized_poll_interval(limits);
    context.owns_soft_deadline = owns_soft_deadline;
    return context;
}

[[nodiscard]] std::int64_t elapsed_ms(const SearchContext& context) noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                 context.started_at)
        .count();
}

[[nodiscard]] constexpr bool is_mate_score(int score) noexcept {
    return score > tt_mate_threshold || score < -tt_mate_threshold;
}

[[nodiscard]] bool is_capture_or_promotion(const Position& position, Move move) noexcept {
    return move.is_valid() &&
           (move.is_en_passant() || move.is_promotion() || position.piece_on(move.to()).is_valid());
}

[[nodiscard]] bool has_non_pawn_material(const Position& position, Color color) noexcept {
    return (position.pieces(color, PieceType::Knight) | position.pieces(color, PieceType::Bishop) |
            position.pieces(color, PieceType::Rook) | position.pieces(color, PieceType::Queen)) !=
           empty_bitboard;
}

[[nodiscard]] bool has_legal_move_for_pruning(const Position& position) {
    return has_legal_moves(position);
}

void remember_static_evaluation(SearchContext& context, int ply, int static_eval) noexcept {
    if (ply >= 0 && std::cmp_less(ply, context.static_evaluations.size())) {
        context.static_evaluations[static_cast<std::size_t>(ply)] = static_eval;
    }
}

[[nodiscard]] bool is_improving(const SearchContext& context, int ply, int static_eval) noexcept {
    if (ply < 2 || std::cmp_greater_equal(ply, context.static_evaluations.size())) {
        return false;
    }
    return static_eval > context.static_evaluations[static_cast<std::size_t>(ply - 2)];
}

[[nodiscard]] int quiet_history_score(const SearchContext& context, const Position& position,
                                      Move move, MoveContinuationKey previous) noexcept {
    if (context.limits.move_ordering_state == nullptr || !is_quiet_ordering_move(position, move)) {
        return 0;
    }
    return context.limits.move_ordering_state->quiet_score(position.piece_on(move.from()),
                                                           move.to(), previous);
}

[[nodiscard]] bool nnue_enabled(const SearchContext& context) noexcept {
    return context.limits.nnue_network != nullptr && context.limits.nnue_network->is_loaded();
}

[[nodiscard]] bool ply_in_accumulator_range(int ply) noexcept {
    return ply >= 0 && std::cmp_less(ply, max_search_ply);
}

void prepare_nnue_accumulators(SearchContext& context, const Position& position) {
    if (!nnue_enabled(context)) {
        context.nnue_accumulator_valid = {};
        return;
    }

    const auto hidden_size = context.limits.nnue_network->hidden_size();
    for (auto& accumulator : context.nnue_accumulators) {
        if (accumulator.hidden_size() != hidden_size) {
            accumulator.reset(hidden_size);
        }
    }
    context.nnue_accumulator_valid = {};
    context.nnue_accumulators[0] = refresh_nnue_accumulator(position, *context.limits.nnue_network,
                                                            context.limits.nnue_backend);
    context.nnue_accumulator_valid[0] = true;
}

void push_nnue_accumulator_after_make(SearchContext& context, Move move, const StateInfo& state,
                                      int parent_ply) {
    if (!nnue_enabled(context)) {
        return;
    }
    const auto child_ply = parent_ply + 1;
    if (!ply_in_accumulator_range(parent_ply) || !ply_in_accumulator_range(child_ply)) {
        return;
    }

    const auto parent_index = static_cast<std::size_t>(parent_ply);
    const auto child_index = static_cast<std::size_t>(child_ply);
    context.nnue_accumulator_valid[child_index] = false;
    if (!context.nnue_accumulator_valid[parent_index]) {
        return;
    }

    auto& child = context.nnue_accumulators[child_index];
    if (!child.copy_from(context.nnue_accumulators[parent_index])) {
        return;
    }
    if (!update_nnue_accumulator_after_make(child, *context.limits.nnue_network, move, state,
                                            context.limits.nnue_backend)) {
        return;
    }
    context.nnue_accumulator_valid[child_index] = true;
}

void push_nnue_accumulator_after_null_move(SearchContext& context, int parent_ply) {
    if (!nnue_enabled(context)) {
        return;
    }
    const auto child_ply = parent_ply + 1;
    if (!ply_in_accumulator_range(parent_ply) || !ply_in_accumulator_range(child_ply) ||
        !context.nnue_accumulator_valid[static_cast<std::size_t>(parent_ply)]) {
        return;
    }

    const auto parent_index = static_cast<std::size_t>(parent_ply);
    const auto child_index = static_cast<std::size_t>(child_ply);
    context.nnue_accumulator_valid[child_index] = false;
    if (context.nnue_accumulators[child_index].copy_from(context.nnue_accumulators[parent_index])) {
        context.nnue_accumulator_valid[child_index] = true;
    }
}

[[nodiscard]] int evaluate_position(const SearchContext& context, const Position& position,
                                    int ply) noexcept {
    if (context.limits.nnue_network != nullptr && context.limits.nnue_network->is_loaded()) {
        if (ply_in_accumulator_range(ply) &&
            context.nnue_accumulator_valid[static_cast<std::size_t>(ply)]) {
            return evaluate_nnue(position, *context.limits.nnue_network,
                                 context.nnue_accumulators[static_cast<std::size_t>(ply)],
                                 context.limits.nnue_backend)
                .value();
        }
        return evaluate_nnue(position, *context.limits.nnue_network, context.limits.nnue_backend)
            .value();
    }
    return evaluate(position).value();
}

[[nodiscard]] const std::array<std::array<int, max_search_ply>, max_search_ply>&
lmr_reduction_table() {
    static const auto table = [] {
        std::array<std::array<int, max_search_ply>, max_search_ply> result{};
        for (int depth = 1; depth < max_search_ply; ++depth) {
            for (int move_number = 1; move_number < max_search_ply; ++move_number) {
                const auto reduction = 0.77 + (std::log(static_cast<double>(depth)) *
                                               std::log(static_cast<double>(move_number)) / 2.36);
                result[static_cast<std::size_t>(depth)][static_cast<std::size_t>(move_number)] =
                    std::max(0, static_cast<int>(reduction));
            }
        }
        return result;
    }();
    return table;
}

[[nodiscard]] int late_move_reduction(int depth, int legal_moves_seen, bool pv_node, bool improving,
                                      int history_score) noexcept {
    const auto clamped_depth = std::clamp(depth, 1, max_search_ply - 1);
    const auto clamped_move_number = std::clamp(legal_moves_seen, 1, max_search_ply - 1);
    auto reduction = lmr_reduction_table()[static_cast<std::size_t>(clamped_depth)]
                                          [static_cast<std::size_t>(clamped_move_number)];
    if (pv_node) {
        --reduction;
    }
    if (!improving) {
        ++reduction;
    }
    reduction -= history_score / history_lmr_scale;
    return std::clamp(reduction, 0, std::max(0, depth - 1));
}

[[nodiscard]] int late_move_pruning_threshold(int depth, bool improving) noexcept {
    const auto denominator = improving ? 1 : late_move_pruning_not_improving_divisor;
    return late_move_pruning_base + ((depth * depth) / denominator);
}

[[nodiscard]] Move least_valuable_capture_to(const Position& position, Square target) noexcept {
    constexpr std::array<PieceType, Piece::type_count> attacker_order{
        PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
        PieceType::Rook, PieceType::Queen,  PieceType::King,
    };
    const auto attacker = position.side_to_move();
    for (const auto type : attacker_order) {
        auto candidates =
            attackers_to(position, target, attacker) & position.pieces(attacker, type);
        while (candidates != empty_bitboard) {
            const auto from = pop_lsb(candidates);
            const auto move = Move::normal(from, target);
            if (static_exchange_score(position, move).value() != 0) {
                return move;
            }
        }
    }
    return Move::none();
}

[[nodiscard]] bool quiet_static_exchange_ge(const Position& child_position, Square target,
                                            Score threshold) noexcept {
    const auto capture = least_valuable_capture_to(child_position, target);
    if (!capture.is_valid()) {
        return true;
    }
    const auto opponent_gain = static_exchange_score(child_position, capture).value();
    return -opponent_gain >= threshold.value();
}

[[nodiscard]] bool reverse_futility_prunes(const SearchContext& context, Position& position,
                                           int depth, int beta, int static_eval, bool pv_node,
                                           bool in_check_now) {
    if (!context.limits.pruning.reverse_futility_pruning || pv_node || in_check_now ||
        depth > reverse_futility_max_depth || is_mate_score(beta) ||
        static_eval - (reverse_futility_margin * depth) < beta) {
        return false;
    }
    return has_legal_move_for_pruning(position);
}

struct NullMovePruningInput {
    int depth{0};
    int beta{0};
    int ply{0};
    int static_eval{0};
    int extensions_used{0};
    bool pv_node{false};
    bool in_check_now{false};
    bool allow_null_move{false};
};

[[nodiscard]] bool null_move_prunes(SearchContext& context, Position& position,
                                    const NullMovePruningInput& input) noexcept;

[[nodiscard]] bool external_stop_requested(const SearchLimits& limits) noexcept {
    return limits.stop_requested != nullptr &&
           limits.stop_requested->load(std::memory_order_relaxed);
}

[[nodiscard]] SearchStopReason shared_stop_reason(const SharedSearchState& shared) noexcept {
    return static_cast<SearchStopReason>(shared.stop_reason.load(std::memory_order_relaxed));
}

void request_shared_stop(SharedSearchState& shared, SearchStopReason reason) noexcept {
    auto expected_reason = static_cast<std::uint8_t>(SearchStopReason::None);
    (void)shared.stop_reason.compare_exchange_strong(
        expected_reason, static_cast<std::uint8_t>(reason), std::memory_order_relaxed,
        std::memory_order_relaxed);
    shared.stopped.store(true, std::memory_order_release);
}

void sync_shared_stop(SearchContext& context) noexcept {
    if (context.shared == nullptr || !context.shared->stopped.load(std::memory_order_acquire)) {
        return;
    }

    context.stopped = true;
    context.stop_reason = shared_stop_reason(*context.shared);
}

void stop_search(SearchContext& context, SearchStopReason reason) noexcept {
    context.stopped = true;
    if (context.shared != nullptr) {
        request_shared_stop(*context.shared, reason);
        context.stop_reason = shared_stop_reason(*context.shared);
        return;
    }
    context.stop_reason = reason;
}

[[nodiscard]] std::uint64_t shared_or_local_nodes(const SearchContext& context) noexcept {
    if (context.shared != nullptr) {
        return context.shared->nodes.load(std::memory_order_relaxed);
    }
    return context.nodes;
}

void poll_search_limits(SearchContext& context, std::uint64_t searched_nodes) noexcept {
    sync_shared_stop(context);
    if (context.stopped) {
        return;
    }
    if (context.limits.nodes != 0 && searched_nodes >= context.limits.nodes) {
        stop_search(context, SearchStopReason::NodeLimit);
        return;
    }
    if (context.nodes < context.next_poll_node) {
        return;
    }

    context.next_poll_node += normalized_poll_interval(context.limits);
    if (external_stop_requested(context.limits)) {
        stop_search(context, SearchStopReason::ExternalStop);
        return;
    }
    if (context.limits.has_hard_deadline &&
        std::chrono::steady_clock::now() >= context.limits.hard_deadline) {
        stop_search(context, SearchStopReason::HardDeadline);
    }
}

[[nodiscard]] bool soft_deadline_reached(const SearchContext& context) noexcept {
    return context.owns_soft_deadline && context.limits.has_soft_deadline &&
           std::chrono::steady_clock::now() >= context.limits.soft_deadline;
}

[[nodiscard]] bool visit_node(SearchContext& context, int ply) noexcept {
    sync_shared_stop(context);
    if (context.stopped) {
        return false;
    }

    ++context.nodes;
    const auto searched_nodes =
        context.shared != nullptr
            ? context.shared->nodes.fetch_add(1, std::memory_order_relaxed) + 1U
            : context.nodes;
    update_seldepth(context, ply);
    poll_search_limits(context, searched_nodes);
    return !context.stopped;
}

[[nodiscard]] int terminal_score(const Position& position, int ply) noexcept {
    return side_to_move_in_check(position) ? (-mate_score + ply) : 0;
}

[[nodiscard]] TTProbe probe_transposition_table(const SearchContext& context, ZobristKey key,
                                                Depth depth, int alpha, int beta,
                                                Ply ply) noexcept {
    if (context.limits.transposition_table == nullptr) {
        return {};
    }

    const auto entry = context.limits.transposition_table->probe(key, ply);
    if (!entry.has_value()) {
        return {};
    }

    TTProbe result{.move = entry->move,
                   .score = entry->score,
                   .static_eval = entry->static_eval,
                   .depth = entry->depth,
                   .bound = entry->bound,
                   .has_entry = true};
    if (entry->depth.value() < depth.value()) {
        return result;
    }

    const auto tt_score = entry->score.value();
    if (entry->bound == TranspositionBound::Exact ||
        (entry->bound == TranspositionBound::Lower && tt_score >= beta) ||
        (entry->bound == TranspositionBound::Upper && tt_score <= alpha)) {
        result.cutoff_score = tt_score;
    }
    return result;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] TranspositionBound bound_for_score(int score, int alpha_original, int beta) noexcept {
    if (score <= alpha_original) {
        return TranspositionBound::Upper;
    }
    if (score >= beta) {
        return TranspositionBound::Lower;
    }
    return TranspositionBound::Exact;
}

[[nodiscard]] constexpr SearchScoreBound
search_bound_from_tt_bound(TranspositionBound bound) noexcept {
    switch (bound) {
    case TranspositionBound::Exact:
        return SearchScoreBound::Exact;
    case TranspositionBound::Lower:
        return SearchScoreBound::Lower;
    case TranspositionBound::Upper:
        return SearchScoreBound::Upper;
    case TranspositionBound::None:
        break;
    }
    return SearchScoreBound::None;
}

[[nodiscard]] int hashfull_permill(const SearchContext& context) noexcept {
    if (context.limits.transposition_table == nullptr) {
        return 0;
    }
    return context.limits.transposition_table->hashfull_permill();
}

void save_transposition(SearchContext& context, ZobristKey key, Move move, int score,
                        int static_eval, Depth depth, TranspositionBound bound, Ply ply,
                        bool is_pv) noexcept {
    if (context.limits.transposition_table == nullptr) {
        return;
    }

    context.limits.transposition_table->save(key, move, Score::centipawns(score),
                                             Score::centipawns(static_eval), depth, bound, ply,
                                             is_pv);
}

void finalize_result(SearchResult& result, const SearchContext& context) noexcept {
    result.nodes = shared_or_local_nodes(context);
    result.elapsed_ms = elapsed_ms(context);
    result.nps = result.elapsed_ms > 0 ? (result.nodes * milliseconds_per_second) /
                                             static_cast<std::uint64_t>(result.elapsed_ms)
                                       : result.nodes;
    result.hashfull_permill = hashfull_permill(context);
    result.seldepth = Ply::from_root(context.seldepth);
    result.pruning_stats = context.pruning_stats;
    result.extension_stats = context.extension_stats;
    if (context.limits.pruning_stats != nullptr) {
        *context.limits.pruning_stats = context.pruning_stats;
    }
    if (context.limits.extension_stats != nullptr) {
        *context.limits.extension_stats = context.extension_stats;
    }
    if (context.stopped) {
        result.stopped = true;
        result.stop_reason = context.stop_reason;
        result.bound = SearchScoreBound::None;
    }
}

void notify_iteration(const SearchContext& context, const SearchResult& result) noexcept {
    if (context.limits.iteration_callback != nullptr) {
        context.limits.iteration_callback(result, context.limits.iteration_callback_userdata);
    }
}

[[nodiscard]] MovePickerContext move_picker_context(const SearchContext& context, Move tt_move,
                                                    Ply ply, MoveContinuationKey previous,
                                                    bool score_captures_with_see = false) noexcept {
    return MovePickerContext{.tt_move = tt_move,
                             .ordering = context.limits.move_ordering_state,
                             .ply = ply,
                             .previous = previous,
                             .score_captures_with_see = score_captures_with_see};
}

void remember_quiet(std::array<Move, max_pseudo_legal_moves>& quiets, std::size_t& quiet_count,
                    Move move) noexcept {
    if (quiet_count >= quiets.size()) {
        return;
    }
    quiets[quiet_count] = move;
    ++quiet_count;
}

void update_move_ordering_on_cutoff(SearchContext& context, const Position& position, Move move,
                                    MoveContinuationKey previous, int depth, Ply ply,
                                    std::span<const Move> quiets_searched) noexcept {
    if (context.limits.move_ordering_state == nullptr || !is_quiet_ordering_move(position, move)) {
        return;
    }

    const auto bonus = move_ordering_bonus(Depth::plies(depth));
    auto& ordering = *context.limits.move_ordering_state;
    ordering.add_killer(ply, move);
    ordering.update_quiet_move(position, move, previous, bonus);

    for (const auto quiet : quiets_searched) {
        if (quiet != move) {
            ordering.update_quiet_move(position, quiet, previous, -bonus);
        }
    }
}

[[nodiscard]] int quiescence(Position& position, int alpha, int beta, int ply,
                             SearchContext& context, MoveContinuationKey previous) noexcept {
    if (!visit_node(context, ply)) {
        return evaluate_position(context, position, ply);
    }
    if (ply >= max_depth) {
        return evaluate_position(context, position, ply);
    }

    const auto alpha_original = alpha;
    const auto position_key = position.zobrist_key();
    const auto position_ply = Ply::from_root(ply);
    const auto tt_probe = probe_transposition_table(context, position_key, Depth::plies(0), alpha,
                                                    beta, position_ply);
    if (tt_probe.cutoff_score.has_value()) {
        return *tt_probe.cutoff_score;
    }

    const auto in_check_now = side_to_move_in_check(position);
    const auto static_eval = evaluate_position(context, position, ply);
    if (!in_check_now) {
        if (static_eval >= beta) {
            save_transposition(context, position_key, Move::none(), static_eval, static_eval,
                               Depth::plies(0), TranspositionBound::Lower, position_ply, false);
            return static_eval;
        }
        alpha = std::max(alpha, static_eval);
    }

    const auto moves = in_check_now ? generate_moves(position) : generate_captures(position);
    MovePicker picker(
        position, moves,
        move_picker_context(context, tt_probe.move, position_ply, previous, !in_check_now));

    bool found_legal_move = false;
    Move best_move = Move::none();
    for (auto move = picker.next(); move.is_valid() && !context.stopped; move = picker.next()) {
        if (!in_check_now && !static_exchange_ge(position, move)) {
            continue;
        }

        const auto child_previous = continuation_key_for_move(position, move);
        StateInfo state;
        if (!position.make_move(move, state)) {
            continue;
        }
        push_nnue_accumulator_after_make(context, move, state, ply);

        found_legal_move = true;
        const auto score = -quiescence(position, -beta, -alpha, ply + 1, context, child_previous);
        position.unmake_move(move, state);
        if (context.stopped) {
            break;
        }

        if (score >= beta) {
            save_transposition(context, position_key, move, score, static_eval, Depth::plies(0),
                               TranspositionBound::Lower, position_ply, false);
            return score;
        }
        if (score > alpha) {
            alpha = score;
            best_move = move;
        }
    }

    if (context.stopped) {
        return static_eval;
    }
    if (in_check_now && !found_legal_move) {
        const auto score = terminal_score(position, ply);
        save_transposition(context, position_key, Move::none(), score, static_eval, Depth::plies(0),
                           TranspositionBound::Exact, position_ply, true);
        return score;
    }
    save_transposition(context, position_key, best_move, alpha, static_eval, Depth::plies(0),
                       bound_for_score(alpha, alpha_original, beta), position_ply, false);
    return alpha;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] int alpha_beta(Position& position, int depth, int alpha, int beta, int ply,
                             SearchContext& context, MoveContinuationKey previous, bool pv_node,
                             bool allow_null_move, int extensions_used, Move excluded_move,
                             PrincipalVariation& principal_variation) noexcept;

[[nodiscard]] bool can_extend(const SearchContext& context, int extensions_used) noexcept {
    return extensions_used < std::max(1, context.root_depth);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] constexpr bool is_zero_window(int alpha, int beta) noexcept {
    return beta <= alpha + 1;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] bool internal_iterative_reduction_applies(const SearchContext& context,
                                                        const TTProbe& tt_probe, int depth,
                                                        int alpha, int beta, bool pv_node,
                                                        bool in_check_now,
                                                        Move excluded_move) noexcept {
    return context.limits.extensions.internal_iterative_reductions && !excluded_move.is_valid() &&
           !tt_probe.move.is_valid() && depth >= internal_iterative_reduction_min_depth &&
           !in_check_now && (pv_node || is_zero_window(alpha, beta));
}

[[nodiscard]] bool singular_tt_candidate(const TTProbe& tt_probe, int depth) noexcept {
    return depth >= singular_extension_min_depth && tt_probe.has_entry &&
           tt_probe.move.is_valid() &&
           tt_probe.depth.value() >= depth - singular_extension_tt_depth_margin &&
           tt_probe.bound != TranspositionBound::Upper && !is_mate_score(tt_probe.score.value());
}

struct SingularSearchResult {
    Move move{Move::none()};
    bool extend{false};
    bool multi_cut{false};
};

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[nodiscard]] SingularSearchResult
run_singular_search(Position& position, const TTProbe& tt_probe, int depth, int beta, int ply,
                    SearchContext& context, MoveContinuationKey previous, bool pv_node,
                    int extensions_used, Move excluded_move) noexcept {
    if (excluded_move.is_valid() || !singular_tt_candidate(tt_probe, depth)) {
        return {};
    }

    const auto extension_available = can_extend(context, extensions_used);
    if ((!context.limits.extensions.singular_extensions || !extension_available) &&
        !context.limits.extensions.multi_cut) {
        return {};
    }

    const auto singular_beta = tt_probe.score.value() - (singular_extension_beta_margin * depth);
    if (singular_beta <= -search_infinity || singular_beta >= search_infinity) {
        return {};
    }

    ++context.extension_stats.singular_searches;
    const auto singular_depth = std::max(0, (depth - 1) / 2);
    PrincipalVariation ignored_line;
    const auto score =
        alpha_beta(position, singular_depth, singular_beta - 1, singular_beta, ply, context,
                   previous, false, false, extensions_used, tt_probe.move, ignored_line);
    if (context.stopped) {
        return {};
    }

    SingularSearchResult result{.move = tt_probe.move};
    const auto singular_failed_low = score < singular_beta;
    const auto singular_failed_high = score >= singular_beta;
    if (singular_failed_low && context.limits.extensions.singular_extensions &&
        extension_available) {
        result.extend = true;
    } else if (singular_failed_high && context.limits.extensions.multi_cut && !pv_node &&
               singular_beta >= beta && beta < search_infinity && beta > -search_infinity &&
               !is_mate_score(beta)) {
        result.multi_cut = true;
    }
    return result;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

[[nodiscard]] bool null_move_prunes(SearchContext& context, Position& position,
                                    const NullMovePruningInput& input) noexcept {
    if (!context.limits.pruning.null_move_pruning || !input.allow_null_move || input.pv_node ||
        input.in_check_now || input.depth < null_move_min_depth || input.static_eval < input.beta ||
        is_mate_score(input.beta) || !has_non_pawn_material(position, position.side_to_move()) ||
        !has_legal_move_for_pruning(position)) {
        return false;
    }

    StateInfo state;
    if (!position.make_null_move(state)) {
        return false;
    }
    push_nnue_accumulator_after_null_move(context, input.ply);

    const auto eval_margin =
        std::max(0, (input.static_eval - input.beta) / null_move_eval_margin_divisor);
    const auto reduction = std::min(
        input.depth - 1, null_move_base_reduction + (input.depth / null_move_depth_divisor) +
                             std::min(null_move_eval_margin_cap, eval_margin));
    const auto null_depth = std::max(0, input.depth - reduction - 1);
    PrincipalVariation ignored_line;
    const auto score =
        -alpha_beta(position, null_depth, -input.beta, -input.beta + 1, input.ply + 1, context, {},
                    false, false, input.extensions_used, Move::none(), ignored_line);
    position.unmake_null_move(state);

    if (!context.stopped && score >= input.beta) {
        ++context.pruning_stats.null_move_cutoffs;
        return true;
    }
    return false;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[nodiscard]] ChildSearch search_child_pvs(Position& position, int child_depth, int alpha, int beta,
                                           Ply child_ply, int searched_legal_moves,
                                           SearchContext& context, MoveContinuationKey previous,
                                           bool child_pv, int reduction,
                                           int extensions_used) noexcept {
    ChildSearch result;
    if (searched_legal_moves == 0) {
        result.score =
            -alpha_beta(position, child_depth, -beta, -alpha, child_ply.value(), context, previous,
                        child_pv, true, extensions_used, Move::none(), result.line);
        return result;
    }

    const auto search_depth = std::max(0, child_depth - reduction);
    result.score =
        -alpha_beta(position, search_depth, -alpha - 1, -alpha, child_ply.value(), context,
                    previous, false, true, extensions_used, Move::none(), result.line);
    if (!context.stopped && reduction > 0 && result.score > alpha) {
        PrincipalVariation unreduced_line;
        result.score =
            -alpha_beta(position, child_depth, -alpha - 1, -alpha, child_ply.value(), context,
                        previous, false, true, extensions_used, Move::none(), unreduced_line);
        result.line = unreduced_line;
    }
    if (!context.stopped && result.score > alpha && result.score < beta) {
        PrincipalVariation full_window_line;
        result.score =
            -alpha_beta(position, child_depth, -beta, -alpha, child_ply.value(), context, previous,
                        child_pv, true, extensions_used, Move::none(), full_window_line);
        result.line = full_window_line;
    }
    return result;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

void remember_root_fallback(SearchResult& result, PrincipalVariation& best_line,
                            Move move) noexcept {
    if (result.best_move.is_valid()) {
        return;
    }

    result.best_move = move;
    best_line.moves = {};
    best_line.moves[0] = move;
    best_line.size = 1;
}

void update_root_best(SearchResult& result, PrincipalVariation& best_line, Move move,
                      const PrincipalVariation& child_line, int score, int& best_score) noexcept {
    if (result.best_move.is_valid() && score <= best_score) {
        return;
    }

    best_score = score;
    result.best_move = move;
    set_line(best_line, move, child_line);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters,readability-function-cognitive-complexity)
[[nodiscard]] int alpha_beta(Position& position, int depth, int alpha, int beta, int ply,
                             SearchContext& context, MoveContinuationKey previous, bool pv_node,
                             bool allow_null_move, int extensions_used, Move excluded_move,
                             PrincipalVariation& principal_variation) noexcept {
    principal_variation.size = 0;
    if (context.stopped) {
        return evaluate_position(context, position, ply);
    }
    if (depth <= 0 || ply >= max_depth) {
        return quiescence(position, alpha, beta, ply, context, previous);
    }

    if (!visit_node(context, ply)) {
        return evaluate_position(context, position, ply);
    }

    const auto alpha_original = alpha;
    const auto position_key = position.zobrist_key();
    const auto position_ply = Ply::from_root(ply);
    const auto can_use_transposition = !excluded_move.is_valid();
    const auto tt_probe =
        can_use_transposition
            ? probe_transposition_table(context, position_key, Depth::plies(depth), alpha, beta,
                                        position_ply)
            : TTProbe{};
    if (tt_probe.cutoff_score.has_value()) {
        return *tt_probe.cutoff_score;
    }

    const auto in_check_now = side_to_move_in_check(position);
    const auto static_eval = evaluate_position(context, position, ply);
    if (internal_iterative_reduction_applies(context, tt_probe, depth, alpha, beta, pv_node,
                                             in_check_now, excluded_move)) {
        --depth;
        ++context.extension_stats.internal_iterative_reductions;
    }
    const auto improving = is_improving(context, ply, static_eval);
    remember_static_evaluation(context, ply, static_eval);

    if (reverse_futility_prunes(context, position, depth, beta, static_eval, pv_node,
                                in_check_now)) {
        ++context.pruning_stats.reverse_futility_prunes;
        if (can_use_transposition) {
            save_transposition(context, position_key, Move::none(), static_eval, static_eval,
                               Depth::plies(depth), TranspositionBound::Lower, position_ply, false);
        }
        return static_eval;
    }

    if (null_move_prunes(context, position,
                         NullMovePruningInput{.depth = depth,
                                              .beta = beta,
                                              .ply = ply,
                                              .static_eval = static_eval,
                                              .extensions_used = extensions_used,
                                              .pv_node = pv_node,
                                              .in_check_now = in_check_now,
                                              .allow_null_move = allow_null_move})) {
        if (can_use_transposition) {
            save_transposition(context, position_key, Move::none(), beta, static_eval,
                               Depth::plies(depth), TranspositionBound::Lower, position_ply, false);
        }
        return beta;
    }

    const auto singular_search =
        run_singular_search(position, tt_probe, depth, beta, ply, context, previous, pv_node,
                            extensions_used, excluded_move);
    if (singular_search.multi_cut) {
        ++context.extension_stats.multi_cut_cutoffs;
        if (can_use_transposition) {
            save_transposition(context, position_key, tt_probe.move, beta, static_eval,
                               Depth::plies(depth), TranspositionBound::Lower, position_ply, false);
        }
        return beta;
    }

    const auto moves = generate_moves(position);
    MovePicker picker(position, moves,
                      move_picker_context(context, tt_probe.move, position_ply, previous));

    bool found_legal_move = false;
    int searched_legal_moves = 0;
    int legal_moves_seen = 0;
    int best_score = -search_infinity;
    PrincipalVariation best_line;
    std::array<Move, max_pseudo_legal_moves> quiets_searched{};
    std::size_t quiet_count = 0;

    for (auto move = picker.next(); move.is_valid() && !context.stopped; move = picker.next()) {
        if (move == excluded_move) {
            continue;
        }
        const auto quiet_move = is_quiet_ordering_move(position, move);
        const auto tactical_move = is_capture_or_promotion(position, move);
        const auto can_prune_move = found_legal_move && !pv_node && !in_check_now;
        const auto history_score =
            quiet_move ? quiet_history_score(context, position, move, previous) : 0;
        const auto capture_see_margin = Score::centipawns(-see_capture_margin * depth);
        const auto quiet_see_margin = Score::centipawns(-see_quiet_margin * depth * depth);
        const auto capture_loses_see =
            tactical_move && !static_exchange_ge(position, move, capture_see_margin);
        const auto child_previous = continuation_key_for_move(position, move);
        StateInfo state;
        if (!position.make_move(move, state)) {
            continue;
        }
        push_nnue_accumulator_after_make(context, move, state, ply);

        found_legal_move = true;
        ++legal_moves_seen;
        const auto gives_check = side_to_move_in_check(position);
        int extension = 0;
        if (singular_search.extend && move == singular_search.move &&
            can_extend(context, extensions_used)) {
            extension = 1;
            ++context.extension_stats.singular_extensions;
        } else if (context.limits.extensions.check_extensions && gives_check &&
                   can_extend(context, extensions_used)) {
            extension = 1;
            ++context.extension_stats.check_extensions;
        }
        if (can_prune_move && !gives_check) {
            if (context.limits.pruning.late_move_pruning && quiet_move &&
                depth <= late_move_pruning_max_depth &&
                legal_moves_seen > late_move_pruning_threshold(depth, improving)) {
                ++context.pruning_stats.late_move_prunes;
                position.unmake_move(move, state);
                continue;
            }
            if (context.limits.pruning.futility_pruning && quiet_move &&
                depth <= futility_max_depth &&
                static_eval + futility_base_margin + (futility_depth_margin * depth) <= alpha) {
                ++context.pruning_stats.futility_prunes;
                position.unmake_move(move, state);
                continue;
            }
            if (context.limits.pruning.see_pruning &&
                ((tactical_move && capture_loses_see) ||
                 (quiet_move &&
                  !quiet_static_exchange_ge(position, move.to(), quiet_see_margin)))) {
                ++context.pruning_stats.see_prunes;
                position.unmake_move(move, state);
                continue;
            }
        }

        int reduction = 0;
        if (context.limits.pruning.late_move_reductions && quiet_move && !gives_check &&
            !in_check_now && depth >= null_move_min_depth && searched_legal_moves > 0) {
            reduction =
                late_move_reduction(depth, legal_moves_seen, pv_node, improving, history_score);
            if (reduction > 0) {
                ++context.pruning_stats.late_move_reductions;
            }
        }

        if (quiet_move) {
            remember_quiet(quiets_searched, quiet_count, move);
        }
        const auto child = search_child_pvs(position, depth - 1 + extension, alpha, beta,
                                            Ply::from_root(ply + 1), searched_legal_moves, context,
                                            child_previous, pv_node && searched_legal_moves == 0,
                                            reduction, extensions_used + extension);
        ++searched_legal_moves;
        position.unmake_move(move, state);
        if (context.stopped) {
            break;
        }

        if (child.score > best_score) {
            best_score = child.score;
            set_line(best_line, move, child.line);
        }
        if (child.score > alpha) {
            alpha = child.score;
            principal_variation = best_line;
        }
        if (alpha >= beta) {
            update_move_ordering_on_cutoff(
                context, position, move, previous, depth, position_ply,
                std::span<const Move>{quiets_searched.data(), quiet_count});
            break;
        }
    }

    if (context.stopped) {
        return evaluate_position(context, position, ply);
    }
    if (!found_legal_move && excluded_move.is_valid()) {
        return alpha_original;
    }
    if (!found_legal_move && !context.stopped) {
        const auto score = terminal_score(position, ply);
        if (can_use_transposition) {
            save_transposition(context, position_key, Move::none(), score, static_eval,
                               Depth::plies(depth), TranspositionBound::Exact, position_ply, true);
        }
        return score;
    }
    if (!found_legal_move) {
        return evaluate_position(context, position, ply);
    }

    if (principal_variation.size == 0 && best_line.size != 0) {
        principal_variation = best_line;
    }
    const auto bound = bound_for_score(best_score, alpha_original, beta);
    const auto best_move = best_line.size != 0U ? best_line.moves[0] : Move::none();
    if (can_use_transposition) {
        save_transposition(context, position_key, best_move, best_score, static_eval,
                           Depth::plies(depth), bound, position_ply,
                           bound == TranspositionBound::Exact);
    }
    return best_score;
}

[[nodiscard]] MoveList legal_root_moves(const Position& position) {
    return generate_legal_moves(position);
}

[[nodiscard]] std::size_t count_legal_root_moves(const Position& position,
                                                 std::span<const Move> root_moves) {
    std::size_t count = 0;
    auto root = position;
    for (const auto move : root_moves) {
        StateInfo state;
        if (root.make_move(move, state)) {
            root.unmake_move(move, state);
            ++count;
        }
    }
    return count;
}

[[nodiscard]] int grow_aspiration_delta(int delta) noexcept {
    return std::max(delta + 1,
                    (delta * aspiration_growth_numerator) / aspiration_growth_denominator);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] SearchResult search_root_window(const Position& position,
                                              std::span<const Move> root_moves,
                                              SearchContext& context, int depth, int alpha,
                                              int beta) {
    SearchResult result;
    result.depth = Depth::plies(depth);
    context.root_depth = depth;
    prepare_nnue_accumulators(context, position);

    auto root = position;
    const auto alpha_original = alpha;
    const auto position_key = root.zobrist_key();
    const auto position_ply = Ply::from_root(0);
    const auto tt_probe = probe_transposition_table(context, position_key, Depth::plies(depth),
                                                    alpha, beta, position_ply);
    MovePicker picker(root, root_moves,
                      move_picker_context(context, tt_probe.move, position_ply, {}));

    bool found_legal_move = false;
    int searched_legal_moves = 0;
    int best_score = -search_infinity;
    PrincipalVariation best_line;
    std::array<Move, max_pseudo_legal_moves> quiets_searched{};
    std::size_t quiet_count = 0;

    for (auto move = picker.next(); move.is_valid() && !context.stopped; move = picker.next()) {
        const auto quiet_move = is_quiet_ordering_move(root, move);
        const auto child_previous = continuation_key_for_move(root, move);
        StateInfo state;
        if (!root.make_move(move, state)) {
            continue;
        }
        push_nnue_accumulator_after_make(context, move, state, 0);

        found_legal_move = true;
        const auto gives_check = side_to_move_in_check(root);
        int extension = 0;
        if (context.limits.extensions.check_extensions && gives_check && can_extend(context, 0)) {
            extension = 1;
            ++context.extension_stats.check_extensions;
        }
        if (quiet_move) {
            remember_quiet(quiets_searched, quiet_count, move);
        }
        remember_root_fallback(result, best_line, move);
        const auto child = search_child_pvs(
            root, depth - 1 + extension, alpha, beta, Ply::from_root(1), searched_legal_moves,
            context, child_previous, searched_legal_moves == 0, 0, extension);
        ++searched_legal_moves;
        root.unmake_move(move, state);
        if (context.stopped && result.best_move == move) {
            best_score = child.score;
            break;
        }
        if (context.stopped) {
            break;
        }

        update_root_best(result, best_line, move, child.line, child.score, best_score);
        alpha = std::max(alpha, child.score);
        if (alpha >= beta) {
            update_move_ordering_on_cutoff(
                context, root, move, {}, depth, position_ply,
                std::span<const Move>{quiets_searched.data(), quiet_count});
            break;
        }
    }

    if (found_legal_move) {
        result.score = Score::centipawns(best_score);
        result.principal_variation = best_line.moves;
        result.principal_variation_size = best_line.size;
        result.best_move_stability = 1;
        if (!context.stopped) {
            const auto bound = bound_for_score(best_score, alpha_original, beta);
            result.bound = search_bound_from_tt_bound(bound);
            const auto best_move = best_line.size != 0U ? best_line.moves[0] : Move::none();
            save_transposition(context, position_key, best_move, best_score,
                               evaluate_position(context, root, 0), Depth::plies(depth), bound,
                               position_ply, bound == TranspositionBound::Exact);
        }
    } else {
        result.score = Score::centipawns(terminal_score(root, 0));
        if (!context.stopped) {
            result.bound = SearchScoreBound::Exact;
        }
    }

    finalize_result(result, context);
    if (found_legal_move && result.principal_variation_size == 0U && best_line.size != 0U) {
        result.principal_variation = best_line.moves;
        result.principal_variation_size = best_line.size;
    }
    return result;
}

[[nodiscard]] SearchResult search_fixed_depth(const Position& position,
                                              std::span<const Move> root_moves,
                                              SearchContext& context, int depth) {
    return search_root_window(position, root_moves, context, depth, -search_infinity,
                              search_infinity);
}

[[nodiscard]] SearchResult search_depth_with_aspiration(const Position& position,
                                                        std::span<const Move> root_moves,
                                                        SearchContext& context, int depth,
                                                        std::optional<int> previous_score,
                                                        AspirationStats& stats) {
    auto alpha = -search_infinity;
    auto beta = search_infinity;
    auto delta = aspiration_initial_delta;
    bool use_aspiration = previous_score.has_value() && depth >= aspiration_start_depth &&
                          !is_mate_score(*previous_score);
    if (use_aspiration) {
        alpha = std::max(-search_infinity, *previous_score - delta);
        beta = std::min(search_infinity, *previous_score + delta);
    }

    int researches = 0;
    while (true) {
        auto result = search_root_window(position, root_moves, context, depth, alpha, beta);
        result.aspiration_fail_lows = stats.fail_lows;
        result.aspiration_fail_highs = stats.fail_highs;
        if (result.stopped || !use_aspiration || is_mate_score(result.score.value())) {
            return result;
        }

        if (result.score.value() <= alpha) {
            ++stats.fail_lows;
            result.aspiration_fail_lows = stats.fail_lows;
            ++researches;
            if (researches >= aspiration_max_researches) {
                alpha = -search_infinity;
                beta = search_infinity;
                use_aspiration = false;
            } else {
                delta = grow_aspiration_delta(delta);
                alpha = std::max(-search_infinity, result.score.value() - delta);
            }
            continue;
        }

        if (result.score.value() >= beta) {
            ++stats.fail_highs;
            result.aspiration_fail_highs = stats.fail_highs;
            ++researches;
            if (researches >= aspiration_max_researches) {
                alpha = -search_infinity;
                beta = search_infinity;
                use_aspiration = false;
            } else {
                delta = grow_aspiration_delta(delta);
                beta = std::min(search_infinity, result.score.value() + delta);
            }
            continue;
        }

        return result;
    }
}

[[nodiscard]] SearchResult search_iterative(const Position& position, const SearchLimits& limits,
                                            std::span<const Move> root_moves, int max_depth,
                                            SharedSearchState* shared = nullptr,
                                            bool owns_soft_deadline = true) {
    auto context = make_search_context(limits, shared, owns_soft_deadline);
    SearchResult best_result;
    Move previous_best = Move::none();
    std::optional<int> previous_score;
    AspirationStats aspiration_stats;
    const auto legal_root_count = count_legal_root_moves(position, root_moves);

    for (int depth = 1; depth <= max_depth && !context.stopped; ++depth) {
        auto iteration = search_depth_with_aspiration(position, root_moves, context, depth,
                                                      previous_score, aspiration_stats);
        iteration.best_move_changes = best_result.best_move_changes;

        if (!iteration.best_move.is_valid()) {
            best_result = iteration;
            if (!iteration.stopped) {
                notify_iteration(context, best_result);
            }
            break;
        }

        if (iteration.stopped) {
            if (!best_result.best_move.is_valid()) {
                best_result = iteration;
            }
            break;
        }

        if (!previous_best.is_valid() || previous_best != iteration.best_move) {
            if (previous_best.is_valid()) {
                ++iteration.best_move_changes;
            }
            previous_best = iteration.best_move;
            iteration.best_move_stability = 1;
        } else {
            iteration.best_move_stability = best_result.best_move_stability + 1U;
        }

        best_result = iteration;
        previous_score = iteration.score.value();
        notify_iteration(context, best_result);
        if (legal_root_count <= 1U) {
            break;
        }
        if (soft_deadline_reached(context) && best_result.best_move_stability >= 2U) {
            stop_search(context, SearchStopReason::SoftDeadline);
            finalize_result(best_result, context);
            break;
        }
    }

    finalize_result(best_result, context);
    return best_result;
}

void set_search_thread_qos() noexcept {
#ifdef __APPLE__
    (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

void update_result_elapsed_totals(SearchResult& result,
                                  std::chrono::steady_clock::time_point started_at,
                                  std::uint64_t nodes,
                                  const TranspositionTable* transposition_table) noexcept {
    result.nodes = nodes;
    result.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - started_at)
                            .count();
    result.nps = result.elapsed_ms > 0 ? (result.nodes * milliseconds_per_second) /
                                             static_cast<std::uint64_t>(result.elapsed_ms)
                                       : result.nodes;
    result.hashfull_permill =
        transposition_table != nullptr ? transposition_table->hashfull_permill() : 0;
}

[[nodiscard]] SearchResult search_single_thread(const Position& position, SearchLimits limits,
                                                std::span<const Move> root_moves,
                                                SharedSearchState* shared = nullptr,
                                                bool owns_soft_deadline = true) {
    std::unique_ptr<MoveOrderingState> local_move_ordering;
    if (limits.move_ordering_state == nullptr) {
        local_move_ordering = std::make_unique<MoveOrderingState>();
        limits.move_ordering_state = local_move_ordering.get();
    }

    limits.threads = default_search_threads;
    const auto depth = clamp_depth(limits.depth.value());
    if (limits.iterative) {
        return search_iterative(position, limits, root_moves, depth, shared, owns_soft_deadline);
    }

    auto context = make_search_context(limits, shared, owns_soft_deadline);
    return search_fixed_depth(position, root_moves, context, depth);
}

[[nodiscard]] SearchResult search_lazy_smp(const Position& position, SearchLimits limits,
                                           std::span<const Move> root_moves, int threads) {
    set_search_thread_qos();

    SharedSearchState shared;
    const auto started_at = std::chrono::steady_clock::now();
    std::vector<std::jthread> helpers;
    helpers.reserve(static_cast<std::size_t>(threads - 1));

    for (int thread_index = 1; thread_index < threads; ++thread_index) {
        helpers.emplace_back([position, root_moves, limits, &shared]() mutable {
            set_search_thread_qos();

            SearchLimits worker_limits = limits;
            worker_limits.iteration_callback = nullptr;
            worker_limits.iteration_callback_userdata = nullptr;
            worker_limits.move_ordering_state = nullptr;
            worker_limits.pruning_stats = nullptr;
            worker_limits.extension_stats = nullptr;
            worker_limits.threads = default_search_threads;
            (void)search_single_thread(position, worker_limits, root_moves, &shared, false);
        });
    }

    SearchResult result = search_single_thread(position, limits, root_moves, &shared, true);
    const auto shared_stopped_before_helper_shutdown =
        shared.stopped.load(std::memory_order_acquire);
    const auto shared_reason_before_helper_shutdown = shared_stop_reason(shared);
    if (!result.stopped && !shared_stopped_before_helper_shutdown) {
        request_shared_stop(shared, SearchStopReason::ExternalStop);
    }

    for (auto& helper : helpers) {
        if (helper.joinable()) {
            helper.join();
        }
    }

    if (!result.stopped && shared_stopped_before_helper_shutdown &&
        shared_reason_before_helper_shutdown != SearchStopReason::None) {
        result.stopped = true;
        result.stop_reason = shared_reason_before_helper_shutdown;
        result.bound = SearchScoreBound::None;
    }
    update_result_elapsed_totals(result, started_at, shared.nodes.load(std::memory_order_relaxed),
                                 limits.transposition_table);
    return result;
}

} // namespace

SearchResult search(const Position& position, SearchLimits limits) {
    const auto legal_moves = legal_root_moves(position);
    return search(position, limits, std::span<const Move>{legal_moves.begin(), legal_moves.size()});
}

SearchResult search(const Position& position, SearchLimits limits,
                    std::span<const Move> root_moves) {
    const auto threads = normalized_thread_count(limits);
    if (threads > default_search_threads) {
        return search_lazy_smp(position, limits, root_moves, threads);
    }

    return search_single_thread(position, limits, root_moves);
}

} // namespace eng
