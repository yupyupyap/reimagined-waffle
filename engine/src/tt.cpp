#include <malloc.h>
#include <eng/tt.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>

namespace eng {
namespace {

constexpr int age_modulus = 32;
constexpr int age_replacement_penalty = 4;
constexpr int pv_replacement_bonus = 2;
constexpr int one_kib = 1024;
constexpr int one_mib = one_kib * one_kib;
constexpr int hashfull_sample_entries = 1000;
constexpr int permill_scale = 1000;
constexpr int max_packed_depth = 255;
constexpr std::uint64_t move_mask = 0xFFFFULL;
constexpr std::uint64_t depth_mask = 0xFFULL;
constexpr std::uint64_t bound_mask = 0x3ULL;
constexpr std::uint64_t age_mask = 0x1FULL;
constexpr std::uint64_t flag_mask = 0x1ULL;
constexpr std::uint64_t score_shift = 16;
constexpr std::uint64_t static_eval_shift = 32;
constexpr std::uint64_t depth_shift = 48;
constexpr std::uint64_t bound_shift = 56;
constexpr std::uint64_t age_shift = 58;
constexpr std::uint64_t pv_shift = 63;

struct PackedEntry {
    Move move{Move::none()};
    Score score;
    Score static_eval;
    Depth depth;
    TranspositionBound bound{TranspositionBound::None};
    std::uint8_t age{0};
    bool is_pv{false};
};

[[nodiscard]] constexpr int clamp_to_int16(int value) noexcept {
    return std::clamp(value, static_cast<int>(std::numeric_limits<std::int16_t>::min()),
                      static_cast<int>(std::numeric_limits<std::int16_t>::max()));
}

[[nodiscard]] constexpr std::uint16_t pack_signed_16(int value) noexcept {
    return static_cast<std::uint16_t>(static_cast<std::int16_t>(clamp_to_int16(value)));
}

[[nodiscard]] constexpr int unpack_signed_16(std::uint64_t value, std::uint64_t shift) noexcept {
    return static_cast<int>(static_cast<std::int16_t>((value >> shift) & move_mask));
}

[[nodiscard]] constexpr std::uint8_t clamp_depth_to_u8(Depth depth) noexcept {
    return static_cast<std::uint8_t>(std::clamp(depth.value(), 0, max_packed_depth));
}

[[nodiscard]] constexpr std::uint8_t normalized_age(std::uint8_t age) noexcept {
    return static_cast<std::uint8_t>(age & (age_modulus - 1));
}

[[nodiscard]] constexpr int age_distance(std::uint8_t current_age,
                                         std::uint8_t entry_age) noexcept {
    return (normalized_age(current_age) - normalized_age(entry_age) + age_modulus) &
           (age_modulus - 1);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] constexpr int replacement_priority(std::uint64_t data,
                                                 std::uint8_t current_age) noexcept {
    if (data == 0U) {
        return std::numeric_limits<int>::min();
    }

    const auto depth = static_cast<int>((data >> depth_shift) & depth_mask);
    const auto entry_age = static_cast<std::uint8_t>((data >> age_shift) & age_mask);
    const auto is_pv = ((data >> pv_shift) & flag_mask) != 0U;
    return depth - (age_distance(current_age, entry_age) * age_replacement_penalty) +
           (is_pv ? pv_replacement_bonus : 0);
}

[[nodiscard]] constexpr std::uint64_t pack_entry(PackedEntry entry) noexcept {
    const auto bound_bits = static_cast<std::uint64_t>(entry.bound) & bound_mask;
    return static_cast<std::uint64_t>(entry.move.raw()) |
           (static_cast<std::uint64_t>(pack_signed_16(entry.score.value())) << score_shift) |
           (static_cast<std::uint64_t>(pack_signed_16(entry.static_eval.value()))
            << static_eval_shift) |
           (static_cast<std::uint64_t>(clamp_depth_to_u8(entry.depth)) << depth_shift) |
           (bound_bits << bound_shift) |
           (static_cast<std::uint64_t>(normalized_age(entry.age)) << age_shift) |
           (entry.is_pv ? (1ULL << pv_shift) : 0ULL);
}

[[nodiscard]] constexpr PackedEntry unpack_entry(std::uint64_t data) noexcept {
    return PackedEntry{
        .move = Move::from_raw(static_cast<Move::Raw>(data & move_mask)),
        .score = Score::centipawns(unpack_signed_16(data, score_shift)),
        .static_eval = Score::centipawns(unpack_signed_16(data, static_eval_shift)),
        .depth = Depth::plies(static_cast<int>((data >> depth_shift) & depth_mask)),
        .bound = static_cast<TranspositionBound>((data >> bound_shift) & bound_mask),
        .age = static_cast<std::uint8_t>((data >> age_shift) & age_mask),
        .is_pv = ((data >> pv_shift) & flag_mask) != 0U,
    };
}

[[nodiscard]] std::size_t bucket_count_for_size_mb(std::size_t hash_mb) noexcept {
    if (hash_mb == 0U) {
        return 0;
    }

    const auto requested_bytes = hash_mb * static_cast<std::size_t>(one_mib);
    const auto requested_buckets =
        std::max<std::size_t>(requested_bytes / tt_bucket_size_bytes, static_cast<std::size_t>(1));
    return std::bit_floor(requested_buckets);
}

} // namespace

struct alignas(tt_bucket_size_bytes) TranspositionTable::Bucket {
    struct Slot {
        std::atomic<std::uint64_t> key_xor;
        std::atomic<std::uint64_t> data;

        Slot() noexcept : key_xor{0}, data{0} {}
    };

    std::array<Slot, tt_entries_per_bucket> slots;

    static_assert(sizeof(Slot) == tt_entry_size_bytes);
};

TranspositionTable::~TranspositionTable() noexcept { release(); }

bool TranspositionTable::resize_mb(std::size_t hash_mb) {
    static_assert(alignof(Bucket) == tt_bucket_size_bytes);
    static_assert(sizeof(Bucket) == tt_bucket_size_bytes);

    const auto new_bucket_count = bucket_count_for_size_mb(hash_mb);
    if (new_bucket_count == 0U) {
        release();
        age_ = 0;
        return true;
    }

    void* memory = nullptr;
    const auto allocation_size = new_bucket_count * sizeof(Bucket);
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
    memory = _aligned_malloc(allocation_size, tt_allocation_alignment); if (memory == nullptr) {
        return false;
    }

    auto* new_buckets = static_cast<Bucket*>(memory);
    for (std::size_t index = 0; index < new_bucket_count; ++index) {
        std::construct_at(new_buckets + index);
    }

    release();
    buckets_ = new_buckets;
    bucket_count_ = new_bucket_count;
    age_ = 0;
    clear();
    return true;
}

void TranspositionTable::clear() noexcept {
    for (std::size_t bucket_index = 0; bucket_index < bucket_count_; ++bucket_index) {
        for (auto& slot : buckets_[bucket_index].slots) {
            slot.data.store(0, std::memory_order_relaxed);
            slot.key_xor.store(0, std::memory_order_relaxed);
        }
    }
}

void TranspositionTable::new_search() noexcept {
    age_ = normalized_age(static_cast<std::uint8_t>(age_ + 1U));
}

std::optional<TranspositionEntry> TranspositionTable::probe(ZobristKey key,
                                                            Ply ply) const noexcept {
    if (bucket_count_ == 0U) {
        return std::nullopt;
    }

    const auto& bucket = buckets_[key & (bucket_count_ - 1U)];
    for (const auto& slot : bucket.slots) {
        const auto data = slot.data.load(std::memory_order_relaxed);
        const auto key_xor = slot.key_xor.load(std::memory_order_relaxed);
        if (data == 0U && key_xor == 0U) {
            continue;
        }
        if ((key_xor ^ data) != key) {
            continue;
        }

        const auto entry = unpack_entry(data);
        if (entry.bound == TranspositionBound::None) {
            continue;
        }
        return TranspositionEntry{
            .move = entry.move,
            .score = score_from_tt(entry.score, ply),
            .static_eval = entry.static_eval,
            .depth = entry.depth,
            .bound = entry.bound,
            .age = entry.age,
            .is_pv = entry.is_pv,
        };
    }

    return std::nullopt;
}

void TranspositionTable::save(ZobristKey key, Move move, Score score, Score static_eval,
                              Depth depth, TranspositionBound bound, Ply ply, bool is_pv) noexcept {
    if (bucket_count_ == 0U || bound == TranspositionBound::None) {
        return;
    }

    const auto packed = pack_entry(PackedEntry{
        .move = move,
        .score = score_to_tt(score, ply),
        .static_eval = static_eval,
        .depth = depth,
        .bound = bound,
        .age = age_,
        .is_pv = is_pv,
    });
    const auto packed_key = key ^ packed;
    auto& bucket = buckets_[key & (bucket_count_ - 1U)];

    auto* replacement = &bucket.slots.front();
    auto replacement_priority_value = std::numeric_limits<int>::max();
    for (auto& slot : bucket.slots) {
        const auto existing_data = slot.data.load(std::memory_order_relaxed);
        const auto existing_key_xor = slot.key_xor.load(std::memory_order_relaxed);
        if (existing_data == 0U && existing_key_xor == 0U) {
            replacement = &slot;
            break;
        }
        if ((existing_key_xor ^ existing_data) == key) {
            replacement = &slot;
            break;
        }

        const auto priority = replacement_priority(existing_data, age_);
        if (priority < replacement_priority_value) {
            replacement_priority_value = priority;
            replacement = &slot;
        }
    }

    replacement->data.store(packed, std::memory_order_relaxed);
    replacement->key_xor.store(packed_key, std::memory_order_relaxed);
}

int TranspositionTable::hashfull_permill() const noexcept {
    if (bucket_count_ == 0U) {
        return 0;
    }

    const auto sample_slots =
        std::min<std::size_t>(bucket_count_ * tt_entries_per_bucket, hashfull_sample_entries);
    std::size_t occupied = 0;
    for (std::size_t slot_index = 0; slot_index < sample_slots; ++slot_index) {
        const auto& bucket = buckets_[slot_index / tt_entries_per_bucket];
        const auto& slot = bucket.slots[slot_index % tt_entries_per_bucket];
        if (slot.data.load(std::memory_order_relaxed) != 0U ||
            slot.key_xor.load(std::memory_order_relaxed) != 0U) {
            ++occupied;
        }
    }

    return static_cast<int>((occupied * static_cast<std::size_t>(permill_scale)) / sample_slots);
}

std::size_t TranspositionTable::size_bytes() const noexcept {
    return bucket_count_ * sizeof(Bucket);
}

void TranspositionTable::release() noexcept {
    if (buckets_ == nullptr) {
        return;
    }

    for (std::size_t index = 0; index < bucket_count_; ++index) {
        std::destroy_at(buckets_ + index);
    }
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
    _aligned_free(buckets_);
    buckets_ = nullptr;
    bucket_count_ = 0;
}

} // namespace eng

