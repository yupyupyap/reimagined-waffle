#include <eng/bitboard.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace eng {
namespace {

constexpr int max_relevant_bits = 12;
constexpr std::size_t max_occupancy_count = std::size_t{1} << max_relevant_bits;
constexpr Bitboard unassigned_attack = std::numeric_limits<Bitboard>::max();
constexpr std::uint64_t rook_seed = 0x9D39247E33776D41ULL;
constexpr std::uint64_t bishop_seed = 0x2AF7398005AAA5C7ULL;
constexpr std::uint64_t square_seed_step = 0x9E3779B97F4A7C15ULL;
constexpr int magic_high_bit_threshold = 6;
constexpr Bitboard high_byte_mask = 0xFF00000000000000ULL;
constexpr int max_magic_attempts = 100000000;
constexpr std::uint64_t splitmix_increment = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t splitmix_multiplier_a = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t splitmix_multiplier_b = 0x94D049BB133111EBULL;
constexpr unsigned splitmix_shift_a = 30U;
constexpr unsigned splitmix_shift_b = 27U;
constexpr unsigned splitmix_shift_c = 31U;

class SplitMix64 {
  public:
    constexpr explicit SplitMix64(std::uint64_t seed) noexcept : state_{seed} {}

    [[nodiscard]] constexpr std::uint64_t next() noexcept {
        state_ += splitmix_increment;
        auto value = state_;
        value = (value ^ (value >> splitmix_shift_a)) * splitmix_multiplier_a;
        value = (value ^ (value >> splitmix_shift_b)) * splitmix_multiplier_b;
        return value ^ (value >> splitmix_shift_c);
    }

    [[nodiscard]] constexpr std::uint64_t sparse() noexcept { return next() & next() & next(); }

  private:
    std::uint64_t state_;
};

template <std::size_t DirectionCount>
[[nodiscard]] constexpr Bitboard
slow_slider_attacks(Square square, Bitboard occupancy,
                    const std::array<Direction, DirectionCount>& directions) noexcept {
    Bitboard attacks = empty_bitboard;
    for (const auto direction : directions) {
        auto current = bitboard_detail::step(square, direction);
        while (current.is_valid()) {
            attacks |= square_bb(current);
            if (has_square(occupancy, current)) {
                break;
            }
            current = bitboard_detail::step(current, direction);
        }
    }
    return attacks;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] constexpr Bitboard subset_from_index(Bitboard mask, std::size_t index) noexcept {
    Bitboard subset = empty_bitboard;
    auto remaining = mask;
    std::size_t bit_index = 0;
    while (remaining != empty_bitboard) {
        const auto square = pop_lsb(remaining);
        if ((index & (std::size_t{1} << bit_index)) != 0U) {
            subset |= square_bb(square);
        }
        ++bit_index;
    }
    return subset;
}

[[nodiscard]] constexpr std::size_t magic_local_index(const MagicEntry& entry,
                                                      Bitboard occupancy) noexcept {
    return static_cast<std::size_t>(((occupancy & entry.mask) * entry.magic) >> entry.shift);
}

[[nodiscard]] constexpr std::size_t magic_table_index(const MagicEntry& entry,
                                                      Bitboard occupancy) noexcept {
    return static_cast<std::size_t>(entry.offset) + magic_local_index(entry, occupancy);
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[nodiscard]] Bitboard find_magic(Bitboard mask, int relevant_bits,
                                  const std::array<Bitboard, max_occupancy_count>& occupancies,
                                  const std::array<Bitboard, max_occupancy_count>& references,
                                  std::size_t occupancy_count, SplitMix64& random) noexcept {
    std::array<Bitboard, max_occupancy_count> used{};
    const auto shift = static_cast<unsigned>(std::numeric_limits<Bitboard>::digits - relevant_bits);

    for (int attempt = 0; attempt < max_magic_attempts; ++attempt) {
        const auto magic = random.sparse();
        if (popcount((mask * magic) & high_byte_mask) < magic_high_bit_threshold) {
            continue;
        }

        used.fill(unassigned_attack);
        bool failed = false;

        for (std::size_t index = 0; index < occupancy_count; ++index) {
            const auto magic_index =
                static_cast<std::size_t>((occupancies[index] * magic) >> shift);
            const auto previous = used[magic_index];
            if (previous == unassigned_attack) {
                used[magic_index] = references[index];
            } else if (previous != references[index]) {
                failed = true;
                break;
            }
        }

        if (!failed) {
            return magic;
        }
    }

    std::abort();
}
// NOLINTEND(bugprone-easily-swappable-parameters)

template <std::size_t DirectionCount>
void init_magic_entry(MagicEntry& entry, std::array<Bitboard, magic_attack_table_size>& attacks,
                      std::uint32_t& next_offset, Square square, Bitboard mask,
                      const std::array<Direction, DirectionCount>& directions,
                      std::uint64_t seed) noexcept {
    std::array<Bitboard, max_occupancy_count> occupancies{};
    std::array<Bitboard, max_occupancy_count> references{};

    const auto relevant_bits = popcount(mask);
    const auto occupancy_count = std::size_t{1} << relevant_bits;

    for (std::size_t index = 0; index < occupancy_count; ++index) {
        const auto occupancy = subset_from_index(mask, index);
        occupancies[index] = occupancy;
        references[index] = slow_slider_attacks(square, occupancy, directions);
    }

    auto random =
        SplitMix64{seed + (square_seed_step * static_cast<std::uint64_t>(square.index()))};
    entry.mask = mask;
    entry.shift = static_cast<std::uint8_t>(std::numeric_limits<Bitboard>::digits - relevant_bits);
    entry.offset = next_offset;
    entry.magic = find_magic(mask, relevant_bits, occupancies, references, occupancy_count, random);

    const auto entry_table_size = static_cast<std::uint32_t>(std::uint32_t{1} << relevant_bits);
    next_offset += entry_table_size;

    for (std::size_t index = 0; index < occupancy_count; ++index) {
        attacks[magic_table_index(entry, occupancies[index])] = references[index];
    }
}

struct MagicTables {
    MagicTables() noexcept {
        std::uint32_t next_offset = 0;
        for (int square_index = 0; square_index < Square::count; ++square_index) {
            const auto square = Square::from_index(square_index);
            init_magic_entry(bishops[static_cast<std::size_t>(square_index)], attacks, next_offset,
                             square, bishop_relevant_occupancy(square),
                             bitboard_detail::bishop_directions, bishop_seed);
        }
        for (int square_index = 0; square_index < Square::count; ++square_index) {
            const auto square = Square::from_index(square_index);
            init_magic_entry(rooks[static_cast<std::size_t>(square_index)], attacks, next_offset,
                             square, rook_relevant_occupancy(square),
                             bitboard_detail::rook_directions, rook_seed);
        }

        if (next_offset != magic_attack_table_size) {
            std::abort();
        }
    }

    std::array<MagicEntry, Square::count> rooks{};
    std::array<MagicEntry, Square::count> bishops{};
    std::array<Bitboard, magic_attack_table_size> attacks{};
};

[[nodiscard]] const MagicTables& magic_tables() noexcept {
    static const MagicTables tables{};
    return tables;
}

[[nodiscard]] const MagicEntry& invalid_magic() noexcept {
    static constexpr MagicEntry entry{};
    return entry;
}

} // namespace

void init_bitboards() noexcept { static_cast<void>(magic_tables()); }

[[nodiscard]] const MagicEntry& rook_magic(Square square) noexcept {
    if (!square.is_valid()) {
        return invalid_magic();
    }
    return magic_tables().rooks[static_cast<std::size_t>(square.index())];
}

[[nodiscard]] const MagicEntry& bishop_magic(Square square) noexcept {
    if (!square.is_valid()) {
        return invalid_magic();
    }
    return magic_tables().bishops[static_cast<std::size_t>(square.index())];
}

[[nodiscard]] Bitboard rook_attacks(Square square, Bitboard occupancy) noexcept {
    if (!square.is_valid()) {
        return empty_bitboard;
    }
    const auto& tables = magic_tables();
    const auto& entry = tables.rooks[static_cast<std::size_t>(square.index())];
    return tables.attacks[magic_table_index(entry, occupancy)];
}

[[nodiscard]] Bitboard bishop_attacks(Square square, Bitboard occupancy) noexcept {
    if (!square.is_valid()) {
        return empty_bitboard;
    }
    const auto& tables = magic_tables();
    const auto& entry = tables.bishops[static_cast<std::size_t>(square.index())];
    return tables.attacks[magic_table_index(entry, occupancy)];
}

[[nodiscard]] Bitboard queen_attacks(Square square, Bitboard occupancy) noexcept {
    return rook_attacks(square, occupancy) | bishop_attacks(square, occupancy);
}

} // namespace eng
