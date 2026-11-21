#include <eng/bench.hpp>
#include <eng/uci.hpp>

#include <iostream>
#include <string_view>

// The engine executable is a stdin/stdout UCI process.
// NOLINTNEXTLINE(bugprone-exception-escape)
int main(int argc, char** argv) {
    constexpr int usage_error = 2;

    if (argc == 1) {
        return eng::run_uci(std::cin, std::cout);
    }
    if (argc == 2 && std::string_view{argv[1]} == "bench") {
        return eng::run_bench_cli(std::cout, std::cerr);
    }

    std::cerr << "usage:\n"
              << "  localchessengine\n"
              << "  localchessengine bench\n";
    return usage_error;
}
