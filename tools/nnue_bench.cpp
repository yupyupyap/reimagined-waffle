#include <eng/nnue.hpp>
#include <eng/nnue_data.hpp>

#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace {

constexpr int success = 0;
constexpr int usage_error = 2;
constexpr int network_error = 3;
constexpr int position_error = 4;
constexpr int equivalence_error = 5;
constexpr std::uint64_t default_iterations = 200000;
constexpr std::uint64_t nanoseconds_per_second = 1000000000ULL;
constexpr std::uint64_t nanoseconds_per_millisecond = 1000000ULL;
constexpr std::uint64_t checksum_seed = 14695981039346656037ULL;
constexpr std::uint64_t checksum_mix = 0x9E3779B97F4A7C15ULL;
constexpr unsigned checksum_left_shift = 6U;
constexpr unsigned checksum_right_shift = 2U;
constexpr int checksum_score_offset = 32768;

struct Checksum {
    std::uint64_t value{checksum_seed};
};

struct Config {
    std::filesystem::path network_path;
    std::filesystem::path dataset_path;
    std::uint64_t iterations{default_iterations};
    bool show_help{false};
};

struct BackendRun {
    eng::NnueInferenceBackend backend{eng::NnueInferenceBackend::Scalar};
    int eval_cp{0};
    std::uint64_t checksum{0};
    std::uint64_t elapsed_ns{0};
    std::uint64_t evals_per_second{0};
};

[[nodiscard]] std::string_view activation_name(eng::NnueNetworkActivation activation) noexcept {
    switch (activation) {
    case eng::NnueNetworkActivation::CReLU:
        return "CReLU";
    case eng::NnueNetworkActivation::SCReLU:
        return "SCReLU";
    }
    return "unknown";
}

void print_usage(std::ostream& output) {
    output << "usage: localchessengine_nnue_bench --network <path> [--iterations <count>]\n"
           << "       localchessengine_nnue_bench --network <path> --dataset <binpack>\n";
}

[[nodiscard]] bool parse_u64(std::string_view text, std::uint64_t& value) noexcept {
    const auto* begin = text.data();
    const auto* end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, value);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

[[nodiscard]] Config parse_args(std::span<char* const> args, std::ostream& error) {
    Config config;
    for (std::size_t index = 1; index < args.size(); ++index) {
        const auto arg = std::string_view{args[index]};
        if (arg == "--help" || arg == "-h") {
            config.show_help = true;
            return config;
        }
        if (arg == "--network" && index + 1 < args.size()) {
            ++index;
            config.network_path = args[index];
            continue;
        }
        if (arg == "--iterations" && index + 1 < args.size()) {
            ++index;
            if (!parse_u64(args[index], config.iterations) || config.iterations == 0) {
                error << "invalid --iterations value\n";
                config.show_help = true;
                return config;
            }
            continue;
        }
        if (arg == "--dataset" && index + 1 < args.size()) {
            ++index;
            config.dataset_path = args[index];
            continue;
        }
        error << "unknown or incomplete argument: " << arg << '\n';
        config.show_help = true;
        return config;
    }
    return config;
}

[[nodiscard]] Checksum mix_checksum(Checksum checksum, int score) noexcept {
    const auto value =
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(score + checksum_score_offset));
    checksum.value ^= value + checksum_mix + (checksum.value << checksum_left_shift) +
                      (checksum.value >> checksum_right_shift);
    return checksum;
}

[[nodiscard]] std::uint64_t evals_per_second(std::uint64_t iterations,
                                             std::uint64_t elapsed_ns) noexcept {
    if (elapsed_ns == 0) {
        return iterations;
    }
    return (iterations * nanoseconds_per_second) / elapsed_ns;
}

[[nodiscard]] BackendRun run_backend(const eng::NnueNetwork& network,
                                     std::span<const eng::NnueAccumulator> accumulators,
                                     eng::NnueInferenceBackend backend, std::uint64_t iterations) {
    auto checksum = Checksum{};
    int eval_cp = 0;
    const auto started_at = std::chrono::steady_clock::now();
    for (std::uint64_t iteration = 0; iteration < iterations; ++iteration) {
        const auto accumulator_index =
            static_cast<std::size_t>((iteration + checksum.value) % accumulators.size());
        eval_cp = network.evaluate_white_relative(accumulators[accumulator_index], backend);
        checksum = mix_checksum(checksum, eval_cp);
    }
    const auto elapsed_ns =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - started_at)
                                       .count());
    return BackendRun{
        .backend = backend,
        .eval_cp = eval_cp,
        .checksum = checksum.value,
        .elapsed_ns = elapsed_ns,
        .evals_per_second = evals_per_second(iterations, elapsed_ns),
    };
}

void print_run(std::ostream& output, const BackendRun& run) {
    output << "backend: " << eng::nnue_inference_backend_name(run.backend) << '\n';
    output << "eval cp: " << run.eval_cp << '\n';
    output << "checksum: " << run.checksum << '\n';
    output << "time ms: " << (run.elapsed_ns / nanoseconds_per_millisecond) << '\n';
    output << "evals/s: " << run.evals_per_second << '\n';
}

[[nodiscard]] std::vector<eng::NnueAccumulator> make_accumulators(const eng::NnueNetwork& network,
                                                                  std::ostream& error) {
    const std::vector<std::string_view> fens{
        eng::Position::startpos_fen,
        "4k3/8/8/8/8/8/4K3/R7 w - - 0 1",
        "4k3/8/8/8/8/8/4K3/R7 b - - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    };

    std::vector<eng::NnueAccumulator> accumulators;
    accumulators.reserve(fens.size());
    for (const auto fen : fens) {
        const auto position = eng::Position::from_fen(fen);
        if (!position.has_value()) {
            error << "failed to parse built-in bench FEN\n";
            return {};
        }
        accumulators.push_back(
            eng::refresh_nnue_accumulator(*position, network, eng::NnueInferenceBackend::Scalar));
    }
    return accumulators;
}

[[nodiscard]] BackendRun run_dataset_backend(const eng::NnueNetwork& network,
                                             const eng::NnueDataset& dataset,
                                             eng::NnueInferenceBackend backend,
                                             std::ostream& error) {
    auto checksum = Checksum{};
    int eval_cp = 0;
    const auto started_at = std::chrono::steady_clock::now();
    for (const auto& record : dataset.records) {
        const auto position = eng::Position::from_fen(record.fen);
        if (!position.has_value()) {
            error << "failed to parse dataset FEN\n";
            return BackendRun{.backend = backend};
        }
        const auto accumulator = eng::refresh_nnue_accumulator(*position, network, backend);
        eval_cp = network.evaluate_relative(accumulator, position->side_to_move(), backend);
        checksum = mix_checksum(checksum, eval_cp);
    }
    const auto elapsed_ns =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - started_at)
                                       .count());
    const auto records = static_cast<std::uint64_t>(dataset.records.size());
    return BackendRun{
        .backend = backend,
        .eval_cp = eval_cp,
        .checksum = checksum.value,
        .elapsed_ns = elapsed_ns,
        .evals_per_second = evals_per_second(records, elapsed_ns),
    };
}

[[nodiscard]] int run_dataset_mode(const eng::NnueNetwork& network, const Config& config,
                                   std::ostream& output, std::ostream& error) {
    const auto dataset = eng::read_nnue_dataset(config.dataset_path);
    if (!dataset.has_value()) {
        error << "failed to load dataset: " << dataset.error().message << '\n';
        return position_error;
    }

    output << "LocalChessEngine NNUE dataset bench\n";
    output << "network: " << config.network_path.string() << '\n';
    output << "dataset: " << config.dataset_path.string() << '\n';
    output << "dataset records: " << dataset->records.size() << '\n';
    output << "dataset checksum: 0x" << std::hex << dataset->checksum << std::dec << '\n';
    output << "neon available: " << (eng::nnue_neon_available() ? 1 : 0) << '\n';
    output << "hidden size: " << network.hidden_size() << '\n';
    output << "activation: " << activation_name(network.activation()) << '\n';

    const auto scalar =
        run_dataset_backend(network, *dataset, eng::NnueInferenceBackend::Scalar, error);
    print_run(output, scalar);

    if (!eng::nnue_neon_available()) {
        output << "backend: neon\n";
        output << "skipped: neon unavailable\n";
        return success;
    }

    const auto neon =
        run_dataset_backend(network, *dataset, eng::NnueInferenceBackend::Neon, error);
    print_run(output, neon);
    if (scalar.checksum != neon.checksum || scalar.eval_cp != neon.eval_cp) {
        error << "NEON dataset result does not match scalar reference\n";
        return equivalence_error;
    }
    return success;
}

} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
int main(int argc, char** argv) {
    const auto args = std::span<char* const>(argv, static_cast<std::size_t>(argc));
    const auto config = parse_args(args, std::cerr);
    if (config.show_help) {
        print_usage(std::cout);
        return usage_error;
    }
    if (config.network_path.empty()) {
        std::cerr << "missing --network path\n";
        print_usage(std::cerr);
        return usage_error;
    }

    const auto network = eng::NnueNetwork::load(config.network_path);
    if (!network.has_value()) {
        std::cerr << "failed to load network: " << network.error().message << '\n';
        return network_error;
    }
    if (!config.dataset_path.empty()) {
        return run_dataset_mode(*network, config, std::cout, std::cerr);
    }

    const auto accumulators = make_accumulators(*network, std::cerr);
    if (accumulators.empty()) {
        return position_error;
    }

    std::cout << "LocalChessEngine NNUE bench\n";
    std::cout << "network: " << config.network_path.string() << '\n';
    std::cout << "neon available: " << (eng::nnue_neon_available() ? 1 : 0) << '\n';
    std::cout << "hidden size: " << network->hidden_size() << '\n';
    std::cout << "activation: " << activation_name(network->activation()) << '\n';
    std::cout << "iterations: " << config.iterations << '\n';

    const auto scalar =
        run_backend(*network, accumulators, eng::NnueInferenceBackend::Scalar, config.iterations);
    print_run(std::cout, scalar);

    if (!eng::nnue_neon_available()) {
        std::cout << "backend: neon\n";
        std::cout << "skipped: neon unavailable\n";
        std::cout << "speedup neon/scalar: n/a\n";
        return success;
    }

    const auto neon =
        run_backend(*network, accumulators, eng::NnueInferenceBackend::Neon, config.iterations);
    print_run(std::cout, neon);
    if (scalar.checksum != neon.checksum || scalar.eval_cp != neon.eval_cp) {
        std::cerr << "NEON result does not match scalar reference\n";
        return equivalence_error;
    }

    const auto speedup =
        static_cast<double>(neon.evals_per_second) / static_cast<double>(scalar.evals_per_second);
    std::cout << "speedup neon/scalar: " << std::fixed << std::setprecision(3) << speedup << '\n';
    return success;
}
