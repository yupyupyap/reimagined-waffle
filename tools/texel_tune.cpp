#include <eng/texel.hpp>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

constexpr int checksum_hex_width = 16;
constexpr int floating_precision = 17;

struct FileConfig {
    std::filesystem::path config_path;
    std::filesystem::path dataset;
    std::filesystem::path output_parameters;
    std::filesystem::path report;
    eng::TexelTuneConfig tune;
};

[[nodiscard]] std::string_view trim_view(std::string_view text) noexcept {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t' || text.front() == '\r')) {
        text.remove_prefix(1);
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    return text;
}

[[nodiscard]] std::string trim_copy(std::string_view text) {
    const auto trimmed = trim_view(text);
    return std::string{trimmed};
}

[[nodiscard]] std::expected<int, std::string> parse_int(std::string_view text) {
    auto value = 0;
    const auto trimmed = trim_view(text);
    const auto* begin = trimmed.data();
    const auto* end = trimmed.data() + trimmed.size();
    const auto [ptr, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || ptr != end) {
        return std::unexpected("invalid integer");
    }
    return value;
}

[[nodiscard]] std::expected<double, std::string> parse_double(std::string_view text) {
    auto value = 0.0;
    const auto trimmed = trim_view(text);
    const auto* begin = trimmed.data();
    const auto* end = trimmed.data() + trimmed.size();
    const auto [ptr, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || ptr != end) {
        return std::unexpected("invalid number");
    }
    return value;
}

[[nodiscard]] std::expected<std::map<std::string, std::string>, std::string>
read_key_value_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        return std::unexpected("could not open config: " + path.string());
    }

    std::map<std::string, std::string> values;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto trimmed = trim_view(line);
        if (trimmed.empty() || trimmed.front() == '#') {
            continue;
        }

        const auto separator = trimmed.find('=');
        if (separator == std::string_view::npos) {
            return std::unexpected("invalid config line " + std::to_string(line_number));
        }

        values[trim_copy(trimmed.substr(0, separator))] = trim_copy(trimmed.substr(separator + 1));
    }
    return values;
}

[[nodiscard]] std::expected<FileConfig, std::string>
load_config(const std::filesystem::path& path) {
    const auto values = read_key_value_file(path);
    if (!values.has_value()) {
        return std::unexpected(values.error());
    }

    FileConfig config;
    config.config_path = path;
    const auto find_required =
        [&values](std::string_view key) -> std::expected<std::string, std::string> {
        const auto found = values->find(std::string{key});
        if (found == values->end() || found->second.empty()) {
            return std::unexpected(std::string{"missing required config key "} + std::string{key});
        }
        return found->second;
    };
    const auto find_int = [&values](std::string_view key,
                                    int fallback) -> std::expected<int, std::string> {
        const auto found = values->find(std::string{key});
        if (found == values->end()) {
            return fallback;
        }
        const auto parsed = parse_int(found->second);
        if (!parsed.has_value()) {
            return std::unexpected(std::string{parsed.error()} + " for " + std::string{key});
        }
        return *parsed;
    };

    const auto dataset = find_required("dataset");
    const auto output_parameters = find_required("output_parameters");
    const auto report = find_required("report");
    if (!dataset.has_value()) {
        return std::unexpected(dataset.error());
    }
    if (!output_parameters.has_value()) {
        return std::unexpected(output_parameters.error());
    }
    if (!report.has_value()) {
        return std::unexpected(report.error());
    }

    config.dataset = *dataset;
    config.output_parameters = *output_parameters;
    config.report = *report;

    const auto scale_min = find_int("scale_min", config.tune.scale_min);
    const auto scale_max = find_int("scale_max", config.tune.scale_max);
    const auto scale_step = find_int("scale_step", config.tune.scale_step);
    const auto iterations = find_int("iterations", config.tune.iterations);
    const auto initial_step = find_int("initial_step", config.tune.initial_step);
    const auto min_step = find_int("min_step", config.tune.min_step);
    for (const auto* parsed :
         {&scale_min, &scale_max, &scale_step, &iterations, &initial_step, &min_step}) {
        if (!parsed->has_value()) {
            return std::unexpected(parsed->error());
        }
    }

    config.tune.scale_min = *scale_min;
    config.tune.scale_max = *scale_max;
    config.tune.scale_step = *scale_step;
    config.tune.iterations = *iterations;
    config.tune.initial_step = *initial_step;
    config.tune.min_step = *min_step;
    return config;
}

[[nodiscard]] std::expected<std::vector<eng::TexelSample>, std::string>
load_samples(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        return std::unexpected("could not open dataset: " + path.string());
    }

    std::vector<eng::TexelSample> samples;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto trimmed = trim_view(line);
        if (trimmed.empty() || trimmed.front() == '#') {
            continue;
        }
        if (line_number == 1 && trimmed.starts_with("result,")) {
            continue;
        }

        const auto first = trimmed.find(',');
        const auto second =
            first == std::string_view::npos ? std::string_view::npos : trimmed.find(',', first + 1);
        if (first == std::string_view::npos || second == std::string_view::npos) {
            return std::unexpected("invalid dataset line " + std::to_string(line_number));
        }

        const auto result = parse_double(trimmed.substr(0, first));
        const auto weight = parse_double(trimmed.substr(first + 1, second - first - 1));
        const auto fen = trim_copy(trimmed.substr(second + 1));
        if (!result.has_value()) {
            return std::unexpected("line " + std::to_string(line_number) + ": " + result.error());
        }
        if (!weight.has_value()) {
            return std::unexpected("line " + std::to_string(line_number) + ": " + weight.error());
        }
        if (*result < 0.0 || *result > 1.0 || *weight <= 0.0) {
            return std::unexpected("line " + std::to_string(line_number) +
                                   ": result must be [0,1] and weight must be positive");
        }

        auto position = eng::Position::from_fen(fen);
        if (!position.has_value()) {
            return std::unexpected("line " + std::to_string(line_number) + ": invalid FEN");
        }
        samples.push_back(
            eng::TexelSample{.position = *position, .result = *result, .weight = *weight});
    }

    if (samples.empty()) {
        return std::unexpected("dataset contains no samples");
    }
    return samples;
}

[[nodiscard]] std::string hex_u64(std::uint64_t value) {
    std::ostringstream output;
    output << "0x" << std::hex << std::setw(checksum_hex_width) << std::setfill('0') << value;
    return output.str();
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

void write_array(std::ostream& output, std::span<const int> values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0U) {
            output << ", ";
        }
        output << values[index];
    }
    output << ']';
}

bool write_parameters(const std::filesystem::path& path, const FileConfig& config,
                      const eng::TexelTuneResult& result) {
    if (!ensure_parent_directory(path)) {
        return false;
    }

    std::ofstream output(path);
    if (!output) {
        return false;
    }

    const auto& parameters = result.final_parameters;
    output << "{\n";
    output << "  \"schema\": \"localchessengine.hce.parameters.v1\",\n";
    output << "  \"status\": \"experimental\",\n";
    output << "  \"source_task\": \"TASK-024\",\n";
    output << "  \"promotion\": \"not-enabled-by-default\",\n";
    output << R"(  "dataset": ")" << config.dataset.string() << "\",\n";
    output << R"(  "dataset_checksum": ")" << hex_u64(result.dataset_checksum) << "\",\n";
    output << R"(  "initial_parameter_checksum": ")" << hex_u64(result.initial_parameter_checksum)
           << "\",\n";
    output << R"(  "parameter_checksum": ")" << hex_u64(result.final_parameter_checksum) << "\",\n";
    output << "  \"scale\": " << result.scale << ",\n";
    output << "  \"loss\": {\n";
    output << "    \"initial\": " << std::setprecision(floating_precision) << result.initial_loss
           << ",\n";
    output << "    \"final\": " << std::setprecision(floating_precision) << result.final_loss
           << "\n";
    output << "  },\n";
    output << "  \"parameters\": {\n";
    output << "    \"middle_game_piece_values\": ";
    write_array(output, parameters.middle_game_piece_values);
    output << ",\n";
    output << "    \"end_game_piece_values\": ";
    write_array(output, parameters.end_game_piece_values);
    output << ",\n";
    output << "    \"middle_game_mobility_weights\": ";
    write_array(output, parameters.middle_game_mobility_weights);
    output << ",\n";
    output << "    \"end_game_mobility_weights\": ";
    write_array(output, parameters.end_game_mobility_weights);
    output << ",\n";
    output << "    \"middle_game_passed_pawn_bonus\": ";
    write_array(output, parameters.middle_game_passed_pawn_bonus);
    output << ",\n";
    output << "    \"end_game_passed_pawn_bonus\": ";
    write_array(output, parameters.end_game_passed_pawn_bonus);
    output << ",\n";
    output << "    \"middle_game_isolated_pawn_penalty\": "
           << parameters.middle_game_isolated_pawn_penalty << ",\n";
    output << "    \"end_game_isolated_pawn_penalty\": "
           << parameters.end_game_isolated_pawn_penalty << ",\n";
    output << "    \"middle_game_doubled_pawn_penalty\": "
           << parameters.middle_game_doubled_pawn_penalty << ",\n";
    output << "    \"end_game_doubled_pawn_penalty\": " << parameters.end_game_doubled_pawn_penalty
           << ",\n";
    output << "    \"middle_game_bishop_pair_bonus\": " << parameters.middle_game_bishop_pair_bonus
           << ",\n";
    output << "    \"end_game_bishop_pair_bonus\": " << parameters.end_game_bishop_pair_bonus
           << ",\n";
    output << "    \"middle_game_rook_open_file_bonus\": "
           << parameters.middle_game_rook_open_file_bonus << ",\n";
    output << "    \"end_game_rook_open_file_bonus\": " << parameters.end_game_rook_open_file_bonus
           << ",\n";
    output << "    \"middle_game_rook_semi_open_file_bonus\": "
           << parameters.middle_game_rook_semi_open_file_bonus << ",\n";
    output << "    \"end_game_rook_semi_open_file_bonus\": "
           << parameters.end_game_rook_semi_open_file_bonus << ",\n";
    output << "    \"king_shield_rank_one_bonus\": " << parameters.king_shield_rank_one_bonus
           << ",\n";
    output << "    \"king_shield_rank_two_bonus\": " << parameters.king_shield_rank_two_bonus
           << ",\n";
    output << "    \"king_semi_open_file_penalty\": " << parameters.king_semi_open_file_penalty
           << ",\n";
    output << "    \"king_open_file_penalty\": " << parameters.king_open_file_penalty << ",\n";
    output << "    \"king_zone_attack_penalty\": " << parameters.king_zone_attack_penalty << "\n";
    output << "  }\n";
    output << "}\n";
    return true;
}

bool write_report(const std::filesystem::path& path, const FileConfig& config,
                  const eng::TexelTuneResult& result) {
    if (!ensure_parent_directory(path)) {
        return false;
    }

    std::ofstream output(path);
    if (!output) {
        return false;
    }

    output << "# TASK-024 Texel sanity tuning report\n\n";
    output
        << "This report is a deterministic sanity artifact for the HCE tuning pipeline. It is not "
           "strength evidence and is not promoted into the default engine path.\n\n";
    output << "## Command\n\n";
    output << "```sh\n";
    output << "./build/debug/localchessengine_texel_tune --config " << config.config_path.string()
           << "\n";
    output << "```\n\n";
    output << "## Inputs\n\n";
    output << "- Dataset: `" << config.dataset.string() << "`\n";
    output << "- Samples: `" << result.samples << "`\n";
    output << "- Dataset checksum: `" << hex_u64(result.dataset_checksum) << "`\n";
    output << "- Scale: `" << result.scale << "`\n";
    output << "- Iterations: `" << config.tune.iterations << "`\n";
    output << "- Initial step: `" << config.tune.initial_step << "`\n";
    output << "- Min step: `" << config.tune.min_step << "`\n\n";
    output << "## Result\n\n";
    output << "- Initial loss: `" << std::setprecision(floating_precision) << result.initial_loss
           << "`\n";
    output << "- Final loss: `" << std::setprecision(floating_precision) << result.final_loss
           << "`\n";
    output << "- Initial parameter checksum: `" << hex_u64(result.initial_parameter_checksum)
           << "`\n";
    output << "- Final parameter checksum: `" << hex_u64(result.final_parameter_checksum) << "`\n";
    output << "- Parameter artifact: `" << config.output_parameters.string() << "`\n\n";
    output << "## Promotion status\n\n";
    output
        << "Experimental only. Default HCE parameters remain unchanged until `TASK-025`/`TASK-026` "
           "provide statistical non-regression or SPRT evidence.\n";
    return true;
}

void print_usage(std::ostream& output) {
    output << "usage: localchessengine_texel_tune --config <path>\n";
}

} // namespace

int run(int argc, char** argv) {
    if (argc != 3 || std::string_view{argv[1]} != "--config") {
        print_usage(std::cerr);
        return 2;
    }

    const auto config = load_config(argv[2]);
    if (!config.has_value()) {
        std::cerr << "error: " << config.error() << '\n';
        return 1;
    }

    const auto samples = load_samples(config->dataset);
    if (!samples.has_value()) {
        std::cerr << "error: " << samples.error() << '\n';
        return 1;
    }

    const auto result = eng::tune_texel(*samples, config->tune);
    if (!write_parameters(config->output_parameters, *config, result)) {
        std::cerr << "error: could not write parameters: " << config->output_parameters << '\n';
        return 1;
    }
    if (!write_report(config->report, *config, result)) {
        std::cerr << "error: could not write report: " << config->report << '\n';
        return 1;
    }

    std::cout << "samples " << result.samples << '\n';
    std::cout << "dataset checksum " << hex_u64(result.dataset_checksum) << '\n';
    std::cout << "scale " << result.scale << '\n';
    std::cout << "initial loss " << std::setprecision(floating_precision) << result.initial_loss
              << '\n';
    std::cout << "final loss " << std::setprecision(floating_precision) << result.final_loss
              << '\n';
    std::cout << "parameter checksum " << hex_u64(result.final_parameter_checksum) << '\n';
    return result.final_loss <= result.initial_loss ? 0 : 1;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
