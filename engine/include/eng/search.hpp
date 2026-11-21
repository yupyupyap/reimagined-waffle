#ifndef ENG_SEARCH_HPP
#define ENG_SEARCH_HPP

#include "eng/move.hpp"
#include "eng/nnue.hpp"
#include "eng/position.hpp"
#include "eng/types.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>

namespace eng {

class TranspositionTable;
class MoveOrderingState;
struct SearchResult;

using SearchIterationCallback = void (*)(const SearchResult& result, void* userdata);

inline constexpr int max_search_ply = 64;
inline constexpr int default_search_depth = 1;
inline constexpr int default_search_threads = 1;
inline constexpr std::uint64_t default_search_poll_interval = 2048;

enum class SearchStopReason : std::uint8_t {
    None = 0,
    NodeLimit,
    SoftDeadline,
    HardDeadline,
    ExternalStop
};

enum class SearchScoreBound : std::uint8_t { None = 0, Exact, Lower, Upper };

struct SearchPruningOptions {
    bool null_move_pruning{false};
    bool late_move_reductions{false};
    bool reverse_futility_pruning{false};
    bool late_move_pruning{false};
    bool futility_pruning{false};
    bool see_pruning{false};

    friend bool operator==(const SearchPruningOptions& lhs,
                           const SearchPruningOptions& rhs) = default;
};

struct SearchPruningStats {
    std::uint64_t null_move_cutoffs{0};
    std::uint64_t late_move_reductions{0};
    std::uint64_t reverse_futility_prunes{0};
    std::uint64_t late_move_prunes{0};
    std::uint64_t futility_prunes{0};
    std::uint64_t see_prunes{0};

    friend bool operator==(const SearchPruningStats& lhs, const SearchPruningStats& rhs) = default;
};

struct SearchExtensionOptions {
    bool singular_extensions{false};
    bool multi_cut{false};
    bool internal_iterative_reductions{false};
    bool check_extensions{false};

    friend bool operator==(const SearchExtensionOptions& lhs,
                           const SearchExtensionOptions& rhs) = default;
};

struct SearchExtensionStats {
    std::uint64_t singular_searches{0};
    std::uint64_t singular_extensions{0};
    std::uint64_t multi_cut_cutoffs{0};
    std::uint64_t internal_iterative_reductions{0};
    std::uint64_t check_extensions{0};

    friend bool operator==(const SearchExtensionStats& lhs,
                           const SearchExtensionStats& rhs) = default;
};

struct SearchLimits {
    Depth depth{Depth::plies(default_search_depth)};
    std::uint64_t nodes{0};
    std::chrono::steady_clock::time_point soft_deadline;
    std::chrono::steady_clock::time_point hard_deadline;
    std::atomic_bool* stop_requested{nullptr};
    TranspositionTable* transposition_table{nullptr};
    MoveOrderingState* move_ordering_state{nullptr};
    SearchIterationCallback iteration_callback{nullptr};
    void* iteration_callback_userdata{nullptr};
    SearchPruningOptions pruning;
    SearchPruningStats* pruning_stats{nullptr};
    SearchExtensionOptions extensions;
    SearchExtensionStats* extension_stats{nullptr};
    const NnueNetwork* nnue_network{nullptr};
    NnueInferenceBackend nnue_backend{NnueInferenceBackend::Scalar};
    int threads{default_search_threads};
    std::uint64_t poll_interval{default_search_poll_interval};
    bool has_soft_deadline{false};
    bool has_hard_deadline{false};
    bool iterative{false};
};

struct SearchResult {
    Move best_move{Move::none()};
    Score score;
    Depth depth;
    Ply seldepth;
    std::uint64_t nodes{0};
    std::uint64_t nps{0};
    std::int64_t elapsed_ms{0};
    int hashfull_permill{0};
    std::array<Move, max_search_ply> principal_variation{};
    std::size_t principal_variation_size{0};
    SearchScoreBound bound{SearchScoreBound::None};
    bool stopped{false};
    SearchStopReason stop_reason{SearchStopReason::None};
    std::uint32_t best_move_changes{0};
    std::uint32_t best_move_stability{0};
    std::uint32_t aspiration_fail_lows{0};
    std::uint32_t aspiration_fail_highs{0};
    SearchPruningStats pruning_stats;
    SearchExtensionStats extension_stats;
};

[[nodiscard]] SearchResult search(const Position& position, SearchLimits limits);
[[nodiscard]] SearchResult search(const Position& position, SearchLimits limits,
                                  std::span<const Move> root_moves);

} // namespace eng

#endif // ENG_SEARCH_HPP
