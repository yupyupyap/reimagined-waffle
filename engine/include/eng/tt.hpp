#ifndef ENG_TT_HPP
#define ENG_TT_HPP

#include "eng/move.hpp"
#include "eng/position.hpp"
#include "eng/types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace eng {

inline constexpr std::size_t tt_entry_size_bytes = 16;
inline constexpr std::size_t tt_bucket_size_bytes = 64;
inline constexpr std::size_t tt_entries_per_bucket = 4;
inline constexpr std::size_t tt_allocation_alignment = 4096;
inline constexpr int tt_mate_score = 30000;
inline constexpr int tt_mate_threshold = 29000;

enum class TranspositionBound : std::uint8_t { None = 0, Exact = 1, Lower = 2, Upper = 3 };

struct TranspositionEntry {
    Move move{Move::none()};
    Score score;
    Score static_eval;
    Depth depth;
    TranspositionBound bound{TranspositionBound::None};
    std::uint8_t age{0};
    bool is_pv{false};
};

[[nodiscard]] constexpr Score score_to_tt(Score score, Ply ply) noexcept {
    auto value = score.value();
    if (value > tt_mate_threshold) {
        value += ply.value();
    } else if (value < -tt_mate_threshold) {
        value -= ply.value();
    }
    return Score::centipawns(value);
}

[[nodiscard]] constexpr Score score_from_tt(Score score, Ply ply) noexcept {
    auto value = score.value();
    if (value > tt_mate_threshold) {
        value -= ply.value();
    } else if (value < -tt_mate_threshold) {
        value += ply.value();
    }
    return Score::centipawns(value);
}

class TranspositionTable {
  public:
    TranspositionTable() noexcept = default;
    explicit TranspositionTable(std::size_t hash_mb) {
        const auto resized = resize_mb(hash_mb);
        (void)resized;
    }
    ~TranspositionTable() noexcept;

    TranspositionTable(const TranspositionTable&) = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;
    TranspositionTable(TranspositionTable&&) = delete;
    TranspositionTable& operator=(TranspositionTable&&) = delete;

    [[nodiscard]] bool resize_mb(std::size_t hash_mb);
    void clear() noexcept;
    void new_search() noexcept;

    [[nodiscard]] std::optional<TranspositionEntry> probe(ZobristKey key, Ply ply) const noexcept;
    void save(ZobristKey key, Move move, Score score, Score static_eval, Depth depth,
              TranspositionBound bound, Ply ply, bool is_pv) noexcept;

    [[nodiscard]] int hashfull_permill() const noexcept;
    [[nodiscard]] std::size_t bucket_count() const noexcept { return bucket_count_; }
    [[nodiscard]] std::size_t size_bytes() const noexcept;
    [[nodiscard]] std::uint8_t age() const noexcept { return age_; }
    [[nodiscard]] const void* allocation_base() const noexcept { return buckets_; }
    [[nodiscard]] bool empty() const noexcept { return bucket_count_ == 0U; }

  private:
    struct Bucket;

    void release() noexcept;

    Bucket* buckets_{nullptr};
    std::size_t bucket_count_{0};
    std::uint8_t age_{0};
};

} // namespace eng

#endif // ENG_TT_HPP
