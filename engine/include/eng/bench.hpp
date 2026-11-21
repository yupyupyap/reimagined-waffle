#ifndef ENG_BENCH_HPP
#define ENG_BENCH_HPP

#include <cstdint>
#include <iosfwd>

namespace eng {

inline constexpr int bench_default_depth = 3;
inline constexpr int bench_position_count = 6;

struct BenchResult {
    int positions{0};
    int depth{0};
    std::uint64_t nodes{0};
    std::int64_t elapsed_ms{0};
    std::uint64_t nps{0};
};

[[nodiscard]] BenchResult run_bench(int depth = bench_default_depth);
void write_bench(std::ostream& output, const BenchResult& result);
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] int run_bench_cli(std::ostream& output, std::ostream& error);

} // namespace eng

#endif // ENG_BENCH_HPP
