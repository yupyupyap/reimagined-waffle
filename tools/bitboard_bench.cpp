#include <eng/bitboard.hpp>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace {

constexpr std::uint64_t splitmix_increment = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t splitmix_multiplier_a = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t splitmix_multiplier_b = 0x94D049BB133111EBULL;
constexpr unsigned splitmix_shift_a = 30U;
constexpr unsigned splitmix_shift_b = 27U;
constexpr unsigned splitmix_shift_c = 31U;
constexpr int iterations = 100000;
constexpr int attacks_per_square = 3;
constexpr std::uint64_t bench_seed = 0xA0761D6478BD642FULL;
constexpr int elapsed_precision = 6;

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

  private:
    std::uint64_t state_;
};

} // namespace

int main() {
    eng::init_bitboards();

    constexpr auto call_count = static_cast<std::uint64_t>(iterations) *
                                static_cast<std::uint64_t>(eng::Square::count) *
                                static_cast<std::uint64_t>(attacks_per_square);

    auto random = SplitMix64{bench_seed};
    eng::Bitboard checksum = eng::empty_bitboard;

    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int square_index = 0; square_index < eng::Square::count; ++square_index) {
            const auto square = eng::Square::from_index(square_index);
            const auto occupancy = random.next();
            checksum ^= eng::rook_attacks(square, occupancy);
            checksum ^= eng::bishop_attacks(square, occupancy) << 1U;
            checksum ^= eng::queen_attacks(square, occupancy) >> 1U;
        }
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const auto elapsed_seconds = std::chrono::duration<double>(elapsed).count();
    const auto calls_per_second = static_cast<double>(call_count) / elapsed_seconds;

    std::cout << "bitboard_bench calls " << call_count << " time_s " << std::fixed
              << std::setprecision(elapsed_precision) << elapsed_seconds << " calls_per_s "
              << static_cast<std::uint64_t>(calls_per_second) << " checksum 0x" << std::hex
              << checksum << std::dec << '\n';

    return 0;
}
