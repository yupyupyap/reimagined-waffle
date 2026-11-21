#include <eng/nnue_data.hpp>

#include <eng/bitboard.hpp>
#include <eng/uci.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int success = 0;
constexpr int usage_error = 2;
constexpr int runtime_error = 1;
constexpr int checksum_hex_width = 16;
constexpr int decimal_base = 10;
constexpr int leap_year_divisor = 4;
constexpr int century_year_divisor = 100;
constexpr int quadricentennial_year_divisor = 400;
constexpr int leap_february_days = 29;
constexpr std::size_t utc_date_length = 20;
constexpr std::size_t utc_date_year_dash = 4;
constexpr std::size_t utc_date_month_offset = 5;
constexpr std::size_t utc_date_month_dash = 7;
constexpr std::size_t utc_date_day_offset = 8;
constexpr std::size_t utc_date_time_separator = 10;
constexpr std::size_t utc_date_hour_offset = 11;
constexpr std::size_t utc_date_hour_colon = 13;
constexpr std::size_t utc_date_minute_offset = 14;
constexpr std::size_t utc_date_minute_colon = 16;
constexpr std::size_t utc_date_second_offset = 17;
constexpr std::size_t utc_date_zone = 19;
constexpr std::size_t year_digits = 4;
constexpr std::size_t date_component_digits = 2;
constexpr int min_month = 1;
constexpr int max_month = 12;
constexpr int min_day = 1;
constexpr int max_hour = 23;
constexpr int max_minute_or_second = 59;
constexpr int diagnostic_equal_margin_cp = 50;
constexpr int diagnostic_decisive_cp = 1000;
constexpr std::uint16_t diagnostic_ply_bucket_15 = 15;
constexpr std::uint16_t diagnostic_ply_bucket_31 = 31;
constexpr std::uint16_t diagnostic_ply_bucket_63 = 63;
constexpr std::size_t source_fen_field_count = 4;
constexpr std::size_t full_fen_field_count = 6;
constexpr std::size_t fullmove_fen_field_index = 5;
constexpr int epd_id_width = 5;
constexpr std::uint32_t relabel_default_preview_limit = 256;
constexpr int material_bucket_8 = 8;
constexpr int material_bucket_16 = 16;
constexpr int material_bucket_24 = 24;
constexpr std::uint64_t nodes_bucket_1k = 1000;
constexpr std::uint64_t nodes_bucket_10k = 10000;
constexpr std::uint64_t nodes_bucket_100k = 100000;

struct GenerateCommand {
    std::filesystem::path output;
    std::filesystem::path manifest;
    std::filesystem::path preview;
    std::string dataset_id;
    std::string source_task{"TASK-027"};
    std::string notes{"Smoke dataset only; not training scale and not strength evidence."};
    std::string date;
    eng::NnueDataGenerationConfig config;
};

struct RelabelFensCommand {
    std::filesystem::path input;
    std::filesystem::path train_output;
    std::filesystem::path holdout_output;
    std::filesystem::path match_openings_output;
    std::filesystem::path manifest;
    std::filesystem::path train_manifest;
    std::filesystem::path holdout_manifest;
    std::filesystem::path train_preview;
    std::filesystem::path holdout_preview;
    std::filesystem::path match_preview;
    std::string dataset_id{"lce-task-034-local-relabel"};
    std::string source_task{"TASK-034"};
    std::string notes{
        "TASK-034 local relabel corpus; Lichess/HF rows provide FENs only, labels are local."};
    std::string date;
    eng::NnueFenRelabelConfig config;
    std::uint32_t preview_limit{relabel_default_preview_limit};
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

[[nodiscard]] std::expected<int, std::string> parse_int(std::string_view text) {
    int value = 0;
    const auto trimmed = trim_view(text);
    const auto* begin = trimmed.data();
    const auto* end = trimmed.data() + trimmed.size();
    const auto [ptr, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || ptr != end) {
        return std::unexpected("invalid integer: " + std::string{text});
    }
    return value;
}

[[nodiscard]] std::expected<std::uint64_t, std::string> parse_u64(std::string_view text) {
    std::uint64_t value = 0;
    const auto trimmed = trim_view(text);
    const auto* begin = trimmed.data();
    const auto* end = trimmed.data() + trimmed.size();
    const auto [ptr, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || ptr != end) {
        return std::unexpected("invalid unsigned integer: " + std::string{text});
    }
    return value;
}

[[nodiscard]] std::expected<eng::NnueDataPlayoutPolicy, std::string>
parse_playout_policy(std::string_view text) {
    const auto trimmed = trim_view(text);
    if (trimmed == "best") {
        return eng::NnueDataPlayoutPolicy::Best;
    }
    if (trimmed == "random") {
        return eng::NnueDataPlayoutPolicy::Random;
    }
    if (trimmed == "mixed") {
        return eng::NnueDataPlayoutPolicy::Mixed;
    }
    return std::unexpected("playout-policy must be one of: best, random, mixed");
}

[[nodiscard]] std::expected<std::uint16_t, std::string> parse_permille(std::string_view text) {
    const auto parsed = parse_int(text);
    constexpr int max_permille = static_cast<int>(eng::nnue_wdl_permille_total);
    if (!parsed.has_value() || *parsed < 0 || *parsed > max_permille) {
        return std::unexpected("permille value must be in [0, 1000]");
    }
    return static_cast<std::uint16_t>(*parsed);
}

[[nodiscard]] std::string hex_u64(std::uint64_t value) {
    std::ostringstream output;
    output << "0x" << std::hex << std::setw(checksum_hex_width) << std::setfill('0') << value;
    return output.str();
}

[[nodiscard]] std::string hex_u64_digits(std::uint64_t value) {
    std::ostringstream output;
    output << std::hex << std::setw(checksum_hex_width) << std::setfill('0') << value;
    return output.str();
}

[[nodiscard]] int decimal_digit(char value) noexcept {
    return value >= '0' && value <= '9' ? value - '0' : -1;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] bool parse_fixed_digits(std::string_view text, std::size_t offset, std::size_t count,
                                      int& value) noexcept {
    value = 0;
    if (offset + count > text.size()) {
        return false;
    }
    for (std::size_t index = offset; index < offset + count; ++index) {
        const auto digit = decimal_digit(text[index]);
        if (digit < 0) {
            return false;
        }
        value = (value * decimal_base) + digit;
    }
    return true;
}

[[nodiscard]] bool leap_year(int year) noexcept {
    return (year % leap_year_divisor == 0 && year % century_year_divisor != 0) ||
           year % quadricentennial_year_divisor == 0;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] int days_in_month(int year, int month) noexcept {
    constexpr std::array<int, 12> days_by_month{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2 && leap_year(year)) {
        return leap_february_days;
    }
    return days_by_month[month - 1];
}

[[nodiscard]] bool is_utc_rfc3339_seconds(std::string_view date) noexcept {
    if (date.size() != utc_date_length || date[utc_date_year_dash] != '-' ||
        date[utc_date_month_dash] != '-' || date[utc_date_time_separator] != 'T' ||
        date[utc_date_hour_colon] != ':' || date[utc_date_minute_colon] != ':' ||
        date[utc_date_zone] != 'Z') {
        return false;
    }

    int year = 0;
    int month = 0;
    int day = 0;
    int hour = 0;
    int minute = 0;
    int second = 0;
    if (!parse_fixed_digits(date, 0, year_digits, year) ||
        !parse_fixed_digits(date, utc_date_month_offset, date_component_digits, month) ||
        !parse_fixed_digits(date, utc_date_day_offset, date_component_digits, day) ||
        !parse_fixed_digits(date, utc_date_hour_offset, date_component_digits, hour) ||
        !parse_fixed_digits(date, utc_date_minute_offset, date_component_digits, minute) ||
        !parse_fixed_digits(date, utc_date_second_offset, date_component_digits, second)) {
        return false;
    }

    return month >= min_month && month <= max_month && day >= min_day &&
           day <= days_in_month(year, month) && hour >= 0 && hour <= max_hour && minute >= 0 &&
           minute <= max_minute_or_second && second >= 0 && second <= max_minute_or_second;
}

[[nodiscard]] std::expected<std::string, std::string> format_utc_epoch(std::uint64_t epoch) {
    if (epoch > static_cast<std::uint64_t>(std::numeric_limits<std::time_t>::max())) {
        return std::unexpected("SOURCE_DATE_EPOCH is too large for this platform");
    }

    const auto timestamp = static_cast<std::time_t>(epoch);
    std::tm utc{};
    if (gmtime_s(&utc, &timestamp) == nullptr) {
        return std::unexpected("could not convert epoch timestamp to UTC");
    }

    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

[[nodiscard]] std::expected<std::string, std::string> resolve_manifest_date() {
    if (const auto* source_date_epoch = std::getenv("SOURCE_DATE_EPOCH");
        source_date_epoch != nullptr && source_date_epoch[0] != '\0') {
        const auto parsed = parse_u64(source_date_epoch);
        if (!parsed.has_value()) {
            return std::unexpected("SOURCE_DATE_EPOCH must be an unsigned integer epoch second");
        }
        return format_utc_epoch(*parsed);
    }

    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    if (now < 0) {
        return std::unexpected("current system clock is before the Unix epoch");
    }
    return format_utc_epoch(static_cast<std::uint64_t>(now));
}

[[nodiscard]] std::string default_dataset_id(std::uint64_t checksum) {
    return "lce-nnue-v1-" + hex_u64_digits(checksum);
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

[[nodiscard]] std::string json_escape(std::string_view text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (const auto symbol : text) {
        if (symbol == '\\' || symbol == '"') {
            escaped.push_back('\\');
            escaped.push_back(symbol);
        } else if (symbol == '\n') {
            escaped += "\\n";
        } else {
            escaped.push_back(symbol);
        }
    }
    return escaped;
}

[[nodiscard]] std::string csv_escape(std::string_view text) {
    std::string escaped;
    escaped.push_back('"');
    for (const auto symbol : text) {
        if (symbol == '"') {
            escaped += "\"\"";
        } else {
            escaped.push_back(symbol);
        }
    }
    escaped.push_back('"');
    return escaped;
}

void print_usage(std::ostream& output) {
    output
        << "usage:\n"
        << "  localchessengine_nnue_data generate --output <dataset.binpack> "
           "--manifest <manifest.json> --preview <preview.csv> [options]\n\n"
        << "  localchessengine_nnue_data relabel-fens --input <fens.txt|jsonl|csv> "
           "--train-output <train.binpack> --holdout-output <holdout.binpack> "
           "--match-openings-output <openings.epd> --manifest <manifest.json> [options]\n\n"
        << "options:\n"
        << "  --positions <n>        accepted records to keep, default 16\n"
        << "  --games <n>            self-play games to try, default 4\n"
        << "  --max-plies <n>        plies per game before adjudication, default 48\n"
        << "  --depth <n>            search depth for labels, default 1\n"
        << "  --seed <n>             deterministic opening-playout seed, default 27027\n"
        << "  --playout-policy <p>   best, random or mixed; default best\n"
        << "  --random-playout-permille <n>  mixed random share in [0,1000], default 150\n"
        << "  --top-n-playout-permille <n>   mixed top-N share in [0,1000], default 350\n"
        << "  --top-n <n>            ordered candidate pool for mixed playout, default 4\n"
        << "  --max-records-per-game <n>  0 means unlimited, default 0\n"
        << "  --accepted-sample-stride <n>  keep every n-th accepted-by-filter record, default 1\n"
        << "  --dataset-id <id>      stable manifest dataset id, defaults from checksum\n"
        << "  --source-task <id>     manifest source task, default TASK-027\n"
        << "  --notes <text>         manifest notes text\n"
        << "  --date <timestamp>     UTC RFC3339 manifest date, e.g. 2026-05-11T00:00:00Z\n\n"
        << "relabel-fens options:\n"
        << "  --train-positions <n>      accepted train records, default 1000\n"
        << "  --holdout-positions <n>    accepted holdout records, default 200\n"
        << "  --match-openings <n>       accepted match-opening FENs, default 64\n"
        << "  --depth <n>                local search label depth, default 3\n"
        << "  --nodes <n>                optional local search node cap, default 0\n"
        << "  --hash-mb <n>              relabel TT hash size, default 16\n"
        << "  --holdout-permille <n>     deterministic split share, default 150\n"
        << "  --match-permille <n>       deterministic split share, default 100\n"
        << "  --stability-depth-delta <n>  optional second depth delta, default 0\n"
        << "  --max-stability-delta-cp <n> max accepted depth-delta eval change, default 150\n"
        << "  --preview-limit <n>        preview rows per split, default 256\n";
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::expected<GenerateCommand, std::string> parse_generate(int argc, char** argv) {
    GenerateCommand command;
    for (int index = 2; index < argc; ++index) {
        const std::string_view option{argv[index]};
        const auto require_value = [argc, argv, &index,
                                    option]() -> std::expected<std::string_view, std::string> {
            if (index + 1 >= argc) {
                return std::unexpected("missing value for " + std::string{option});
            }
            ++index;
            return std::string_view{argv[index]};
        };

        if (option == "--output") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            command.output = *value;
        } else if (option == "--manifest") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            command.manifest = *value;
        } else if (option == "--preview") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            command.preview = *value;
        } else if (option == "--positions") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed <= 0) {
                return std::unexpected("positions must be positive");
            }
            command.config.positions = static_cast<std::uint32_t>(*parsed);
        } else if (option == "--games") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed <= 0) {
                return std::unexpected("games must be positive");
            }
            command.config.games = static_cast<std::uint32_t>(*parsed);
        } else if (option == "--max-plies") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed <= 0) {
                return std::unexpected("max-plies must be positive");
            }
            command.config.max_plies = static_cast<std::uint32_t>(*parsed);
        } else if (option == "--depth") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed <= 0) {
                return std::unexpected("depth must be positive");
            }
            command.config.depth = eng::Depth::plies(*parsed);
        } else if (option == "--seed") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_u64(*value);
            if (!parsed.has_value()) {
                return std::unexpected(parsed.error());
            }
            command.config.seed = *parsed;
        } else if (option == "--playout-policy") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_playout_policy(*value);
            if (!parsed.has_value()) {
                return std::unexpected(parsed.error());
            }
            command.config.playout_policy = *parsed;
        } else if (option == "--random-playout-permille") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_permille(*value);
            if (!parsed.has_value()) {
                return std::unexpected(parsed.error());
            }
            command.config.random_playout_permille = *parsed;
        } else if (option == "--top-n-playout-permille") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_permille(*value);
            if (!parsed.has_value()) {
                return std::unexpected(parsed.error());
            }
            command.config.top_n_playout_permille = *parsed;
        } else if (option == "--top-n") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed <= 0) {
                return std::unexpected("top-n must be positive");
            }
            command.config.top_n_playout = static_cast<std::uint32_t>(*parsed);
        } else if (option == "--max-records-per-game") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed < 0) {
                return std::unexpected("max-records-per-game must be non-negative");
            }
            command.config.max_records_per_game = static_cast<std::uint32_t>(*parsed);
        } else if (option == "--accepted-sample-stride") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed <= 0) {
                return std::unexpected("accepted-sample-stride must be positive");
            }
            command.config.accepted_sample_stride = static_cast<std::uint32_t>(*parsed);
        } else if (option == "--dataset-id") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto trimmed = trim_view(*value);
            if (trimmed.empty()) {
                return std::unexpected("dataset-id must not be empty");
            }
            command.dataset_id = trimmed;
        } else if (option == "--source-task") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto trimmed = trim_view(*value);
            if (trimmed.empty()) {
                return std::unexpected("source-task must not be empty");
            }
            command.source_task = trimmed;
        } else if (option == "--notes") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            command.notes = trim_view(*value);
        } else if (option == "--date") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto trimmed = trim_view(*value);
            if (!is_utc_rfc3339_seconds(trimmed)) {
                return std::unexpected(
                    "date must use UTC RFC3339 seconds format YYYY-MM-DDTHH:MM:SSZ");
            }
            command.date = trimmed;
        } else {
            return std::unexpected("unknown option: " + std::string{option});
        }
    }

    if (command.output.empty() || command.manifest.empty() || command.preview.empty()) {
        return std::unexpected("--output, --manifest and --preview are required");
    }
    if (static_cast<int>(command.config.random_playout_permille) +
            static_cast<int>(command.config.top_n_playout_permille) >
        static_cast<int>(eng::nnue_wdl_permille_total)) {
        return std::unexpected("random-playout-permille + top-n-playout-permille must be <= 1000");
    }
    return command;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::expected<RelabelFensCommand, std::string> parse_relabel_fens(int argc,
                                                                                char** argv) {
    RelabelFensCommand command;
    for (int index = 2; index < argc; ++index) {
        const std::string_view option{argv[index]};
        const auto require_value = [argc, argv, &index,
                                    option]() -> std::expected<std::string_view, std::string> {
            if (index + 1 >= argc) {
                return std::unexpected("missing value for " + std::string{option});
            }
            ++index;
            return std::string_view{argv[index]};
        };

        if (option == "--input") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            command.input = *value;
        } else if (option == "--train-output") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            command.train_output = *value;
        } else if (option == "--holdout-output") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            command.holdout_output = *value;
        } else if (option == "--match-openings-output") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            command.match_openings_output = *value;
        } else if (option == "--manifest") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            command.manifest = *value;
        } else if (option == "--train-manifest") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            command.train_manifest = *value;
        } else if (option == "--holdout-manifest") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            command.holdout_manifest = *value;
        } else if (option == "--train-preview") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            command.train_preview = *value;
        } else if (option == "--holdout-preview") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            command.holdout_preview = *value;
        } else if (option == "--match-preview") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            command.match_preview = *value;
        } else if (option == "--dataset-id") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto trimmed = trim_view(*value);
            if (trimmed.empty()) {
                return std::unexpected("dataset-id must not be empty");
            }
            command.dataset_id = trimmed;
        } else if (option == "--source-task") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto trimmed = trim_view(*value);
            if (trimmed.empty()) {
                return std::unexpected("source-task must not be empty");
            }
            command.source_task = trimmed;
        } else if (option == "--notes") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            command.notes = trim_view(*value);
        } else if (option == "--date") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto trimmed = trim_view(*value);
            if (!is_utc_rfc3339_seconds(trimmed)) {
                return std::unexpected(
                    "date must use UTC RFC3339 seconds format YYYY-MM-DDTHH:MM:SSZ");
            }
            command.date = trimmed;
        } else if (option == "--train-positions") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed <= 0) {
                return std::unexpected("train-positions must be positive");
            }
            command.config.train_positions = static_cast<std::uint32_t>(*parsed);
        } else if (option == "--holdout-positions") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed < 0) {
                return std::unexpected("holdout-positions must be non-negative");
            }
            command.config.holdout_positions = static_cast<std::uint32_t>(*parsed);
        } else if (option == "--match-openings") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed < 0) {
                return std::unexpected("match-openings must be non-negative");
            }
            command.config.match_openings = static_cast<std::uint32_t>(*parsed);
        } else if (option == "--depth") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed <= 0) {
                return std::unexpected("depth must be positive");
            }
            command.config.depth = eng::Depth::plies(*parsed);
        } else if (option == "--nodes") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_u64(*value);
            if (!parsed.has_value()) {
                return std::unexpected(parsed.error());
            }
            command.config.nodes = *parsed;
        } else if (option == "--hash-mb") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed <= 0) {
                return std::unexpected("hash-mb must be positive");
            }
            command.config.hash_mb = static_cast<std::uint32_t>(*parsed);
        } else if (option == "--holdout-permille") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_permille(*value);
            if (!parsed.has_value()) {
                return std::unexpected(parsed.error());
            }
            command.config.holdout_permille = *parsed;
        } else if (option == "--match-permille") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_permille(*value);
            if (!parsed.has_value()) {
                return std::unexpected(parsed.error());
            }
            command.config.match_permille = *parsed;
        } else if (option == "--max-abs-eval-cp") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed <= 0) {
                return std::unexpected("max-abs-eval-cp must be positive");
            }
            command.config.max_abs_eval_cp = *parsed;
        } else if (option == "--min-pieces") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed <= 0) {
                return std::unexpected("min-pieces must be positive");
            }
            command.config.min_pieces = static_cast<std::uint32_t>(*parsed);
        } else if (option == "--max-pieces") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed <= 0) {
                return std::unexpected("max-pieces must be positive");
            }
            command.config.max_pieces = static_cast<std::uint32_t>(*parsed);
        } else if (option == "--stability-depth-delta") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed < 0) {
                return std::unexpected("stability-depth-delta must be non-negative");
            }
            command.config.stability_depth_delta = static_cast<std::uint32_t>(*parsed);
        } else if (option == "--max-stability-delta-cp") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed < 0) {
                return std::unexpected("max-stability-delta-cp must be non-negative");
            }
            command.config.max_stability_delta_cp = *parsed;
        } else if (option == "--preview-limit") {
            const auto value = require_value();
            if (!value.has_value()) {
                return std::unexpected(value.error());
            }
            const auto parsed = parse_int(*value);
            if (!parsed.has_value() || *parsed < 0) {
                return std::unexpected("preview-limit must be non-negative");
            }
            command.preview_limit = static_cast<std::uint32_t>(*parsed);
        } else if (option == "--skip-in-check") {
            command.config.skip_in_check = true;
        } else if (option == "--allow-in-check") {
            command.config.skip_in_check = false;
        } else if (option == "--skip-best-move-capture") {
            command.config.skip_best_move_capture = true;
        } else if (option == "--allow-best-move-capture") {
            command.config.skip_best_move_capture = false;
        } else {
            return std::unexpected("unknown option: " + std::string{option});
        }
    }

    if (command.input.empty() || command.train_output.empty() || command.manifest.empty()) {
        return std::unexpected("--input, --train-output and --manifest are required");
    }
    if (command.config.holdout_positions != 0U && command.holdout_output.empty()) {
        return std::unexpected("--holdout-output is required when holdout-positions > 0");
    }
    if (command.config.match_openings != 0U && command.match_openings_output.empty()) {
        return std::unexpected("--match-openings-output is required when match-openings > 0");
    }
    if (static_cast<int>(command.config.holdout_permille) +
            static_cast<int>(command.config.match_permille) >
        static_cast<int>(eng::nnue_wdl_permille_total)) {
        return std::unexpected("holdout-permille + match-permille must be <= 1000");
    }
    if (command.config.min_pieces > command.config.max_pieces) {
        return std::unexpected("min-pieces must not exceed max-pieces");
    }
    return command;
}

bool write_preview(const std::filesystem::path& path, const eng::NnueDataGenerationResult& result) {
    if (!ensure_parent_directory(path)) {
        return false;
    }
    std::ofstream output(path);
    if (!output) {
        return false;
    }

    output << "fen,bestmove,eval_cp,wdl_white,wdl_draw,wdl_black,ply,depth,nodes,zobrist_key,"
              "game_index\n";
    for (const auto& record : result.records) {
        output << csv_escape(record.fen) << ',' << eng::move_to_uci(record.best_move) << ','
               << record.eval_cp << ',' << record.wdl.white_win_permille << ','
               << record.wdl.draw_permille << ',' << record.wdl.black_win_permille << ','
               << record.ply << ',' << record.depth << ',' << record.nodes << ','
               << hex_u64(record.zobrist_key) << ',' << record.game_index << '\n';
    }
    return static_cast<bool>(output);
}

[[nodiscard]] std::vector<std::string> split_space_fields(std::string_view text) {
    std::vector<std::string> fields;
    text = trim_view(text);
    while (!text.empty()) {
        auto end = std::size_t{0};
        while (end < text.size() && text[end] != ' ' && text[end] != '\t') {
            ++end;
        }
        fields.emplace_back(text.substr(0, end));
        text.remove_prefix(end);
        text = trim_view(text);
    }
    return fields;
}

[[nodiscard]] std::expected<std::string, std::string> normalize_input_fen(std::string_view text) {
    const auto fields = split_space_fields(text);
    if (fields.size() == source_fen_field_count) {
        return fields[0] + " " + fields[1] + " " + fields[2] + " " + fields[3] + " 0 1";
    }
    if (fields.size() == full_fen_field_count) {
        return fields[0] + " " + fields[1] + " " + fields[2] + " " + fields[3] + " " + fields[4] +
               " " + fields[fullmove_fen_field_index];
    }
    return std::unexpected("FEN pool rows must contain 4-field or 6-field FENs");
}

[[nodiscard]] std::expected<std::string, std::string> first_csv_field(std::string_view line) {
    line = trim_view(line);
    if (line.empty()) {
        return std::unexpected("empty CSV row");
    }
    std::string value;
    if (line.front() == '"') {
        bool quoted = true;
        for (std::size_t index = 1; index < line.size(); ++index) {
            const auto symbol = line[index];
            if (symbol == '"' && index + 1 < line.size() && line[index + 1] == '"') {
                value.push_back('"');
                ++index;
            } else if (symbol == '"') {
                quoted = false;
                break;
            } else {
                value.push_back(symbol);
            }
        }
        if (quoted) {
            return std::unexpected("unterminated quoted CSV field");
        }
        return value;
    }
    const auto comma = line.find(',');
    return std::string{trim_view(line.substr(0, comma))};
}

[[nodiscard]] std::expected<std::string, std::string> json_fen_field(std::string_view line) {
    const auto key = line.find("\"fen\"");
    if (key == std::string_view::npos) {
        return std::unexpected("JSONL row does not contain a fen field");
    }
    const auto colon = line.find(':', key + 5U);
    if (colon == std::string_view::npos) {
        return std::unexpected("JSONL fen field is missing ':'");
    }
    auto cursor = colon + 1U;
    while (cursor < line.size() && (line[cursor] == ' ' || line[cursor] == '\t')) {
        ++cursor;
    }
    if (cursor >= line.size() || line[cursor] != '"') {
        return std::unexpected("JSONL fen field must be a string");
    }
    ++cursor;
    std::string value;
    bool escaped = false;
    for (; cursor < line.size(); ++cursor) {
        const auto symbol = line[cursor];
        if (escaped) {
            value.push_back(symbol);
            escaped = false;
        } else if (symbol == '\\') {
            escaped = true;
        } else if (symbol == '"') {
            return value;
        } else {
            value.push_back(symbol);
        }
    }
    return std::unexpected("JSONL fen string is unterminated");
}

[[nodiscard]] std::expected<std::optional<std::string>, std::string>
fen_from_pool_line(std::string_view line, std::uint64_t line_number) {
    line = trim_view(line);
    if (line.empty() || line.front() == '#') {
        return std::optional<std::string>{};
    }
    std::expected<std::string, std::string> extracted = std::string{line};
    if (line.front() == '{') {
        extracted = json_fen_field(line);
    } else if (line.contains(',')) {
        extracted = first_csv_field(line);
        if (extracted.has_value() && line_number == 1U && trim_view(*extracted) == "fen") {
            return std::optional<std::string>{};
        }
    }
    if (!extracted.has_value()) {
        return std::unexpected(extracted.error());
    }
    const auto normalized = normalize_input_fen(*extracted);
    if (!normalized.has_value()) {
        return std::unexpected(normalized.error());
    }
    return std::optional<std::string>{*normalized};
}

[[nodiscard]] std::expected<std::vector<std::string>, std::string>
read_fen_pool(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        return std::unexpected("could not open FEN pool input: " + path.string());
    }

    std::vector<std::string> fens;
    std::string line;
    std::uint64_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        auto fen = fen_from_pool_line(line, line_number);
        if (!fen.has_value()) {
            return std::unexpected(path.string() + ":" + std::to_string(line_number) + ": " +
                                   fen.error());
        }
        if (fen->has_value()) {
            fens.push_back(std::move(**fen));
        }
    }
    if (fens.empty()) {
        return std::unexpected("FEN pool input contained no usable FEN rows");
    }
    return fens;
}

bool write_records_preview(const std::filesystem::path& path,
                           std::span<const eng::NnueTrainingRecord> records,
                           std::uint32_t preview_limit) {
    if (path.empty()) {
        return true;
    }
    if (!ensure_parent_directory(path)) {
        return false;
    }
    std::ofstream output(path);
    if (!output) {
        return false;
    }
    output << "fen,bestmove,eval_cp,wdl_white,wdl_draw,wdl_black,ply,depth,nodes,zobrist_key,"
              "record_index\n";
    const auto limit = std::min<std::size_t>(records.size(), preview_limit);
    for (std::size_t index = 0; index < limit; ++index) {
        const auto& record = records[index];
        output << csv_escape(record.fen) << ',' << eng::move_to_uci(record.best_move) << ','
               << record.eval_cp << ',' << record.wdl.white_win_permille << ','
               << record.wdl.draw_permille << ',' << record.wdl.black_win_permille << ','
               << record.ply << ',' << record.depth << ',' << record.nodes << ','
               << hex_u64(record.zobrist_key) << ',' << index << '\n';
    }
    return static_cast<bool>(output);
}

bool write_match_openings(const std::filesystem::path& path,
                          std::span<const eng::NnueTrainingRecord> records) {
    if (path.empty()) {
        return true;
    }
    if (!ensure_parent_directory(path)) {
        return false;
    }
    std::ofstream output(path);
    if (!output) {
        return false;
    }
    for (std::size_t index = 0; index < records.size(); ++index) {
        const auto fields = split_space_fields(records[index].fen);
        if (fields.size() < 4U) {
            return false;
        }
        output << fields[0] << ' ' << fields[1] << ' ' << fields[2] << ' ' << fields[3] << " bm "
               << eng::move_to_uci(records[index].best_move) << "; id \"task034-"
               << std::setw(epd_id_width) << std::setfill('0') << (index + 1U) << "\";\n";
    }
    return static_cast<bool>(output);
}

[[nodiscard]] char side_to_move_from_fen(std::string_view fen) noexcept {
    const auto first_space = fen.find(' ');
    if (first_space == std::string_view::npos || first_space + 1 >= fen.size()) {
        return '?';
    }
    return fen[first_space + 1];
}

struct RecordDiagnostics {
    std::uint64_t eval_negative{0};
    std::uint64_t eval_near_equal{0};
    std::uint64_t eval_positive{0};
    std::uint64_t eval_decisive{0};
    std::uint64_t ply_8_15{0};
    std::uint64_t ply_16_31{0};
    std::uint64_t ply_32_63{0};
    std::uint64_t ply_64_plus{0};
    std::uint64_t wdl_white{0};
    std::uint64_t wdl_draw{0};
    std::uint64_t wdl_black{0};
    std::uint64_t side_to_move_white{0};
    std::uint64_t side_to_move_black{0};
    std::uint64_t games_with_records{0};
    std::uint64_t max_records_in_single_game{0};
};

struct RelabelDiagnostics {
    std::uint64_t eval_negative{0};
    std::uint64_t eval_near_equal{0};
    std::uint64_t eval_positive{0};
    std::uint64_t eval_decisive{0};
    std::uint64_t side_to_move_white{0};
    std::uint64_t side_to_move_black{0};
    std::uint64_t pieces_0_8{0};
    std::uint64_t pieces_9_16{0};
    std::uint64_t pieces_17_24{0};
    std::uint64_t pieces_25_32{0};
    std::uint64_t depth_1{0};
    std::uint64_t depth_2_3{0};
    std::uint64_t depth_4_plus{0};
    std::uint64_t nodes_0_999{0};
    std::uint64_t nodes_1k_9999{0};
    std::uint64_t nodes_10k_99999{0};
    std::uint64_t nodes_100k_plus{0};
};

[[nodiscard]] RelabelDiagnostics
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
diagnostics_for_relabel_records(std::span<const eng::NnueTrainingRecord> records) {
    RelabelDiagnostics diagnostics;
    for (const auto& record : records) {
        if (record.eval_cp < -diagnostic_equal_margin_cp) {
            ++diagnostics.eval_negative;
        } else if (record.eval_cp > diagnostic_equal_margin_cp) {
            ++diagnostics.eval_positive;
        } else {
            ++diagnostics.eval_near_equal;
        }
        if (std::abs(record.eval_cp) >= diagnostic_decisive_cp) {
            ++diagnostics.eval_decisive;
        }
        if (side_to_move_from_fen(record.fen) == 'w') {
            ++diagnostics.side_to_move_white;
        } else if (side_to_move_from_fen(record.fen) == 'b') {
            ++diagnostics.side_to_move_black;
        }

        const auto position = eng::Position::from_fen(record.fen);
        const auto pieces = position.has_value() ? eng::popcount(position->occupancy()) : 0;
        if (pieces <= material_bucket_8) {
            ++diagnostics.pieces_0_8;
        } else if (pieces <= material_bucket_16) {
            ++diagnostics.pieces_9_16;
        } else if (pieces <= material_bucket_24) {
            ++diagnostics.pieces_17_24;
        } else {
            ++diagnostics.pieces_25_32;
        }

        if (record.depth <= 1U) {
            ++diagnostics.depth_1;
        } else if (record.depth <= 3U) {
            ++diagnostics.depth_2_3;
        } else {
            ++diagnostics.depth_4_plus;
        }
        if (record.nodes < nodes_bucket_1k) {
            ++diagnostics.nodes_0_999;
        } else if (record.nodes < nodes_bucket_10k) {
            ++diagnostics.nodes_1k_9999;
        } else if (record.nodes < nodes_bucket_100k) {
            ++diagnostics.nodes_10k_99999;
        } else {
            ++diagnostics.nodes_100k_plus;
        }
    }
    return diagnostics;
}

void write_relabel_diagnostics(std::ostream& output,
                               std::span<const eng::NnueTrainingRecord> records,
                               std::string_view indent) {
    const auto diagnostics = diagnostics_for_relabel_records(records);
    output << indent << "\"eval_buckets\": {\n";
    output << indent << "  \"negative_lt_minus_50\": " << diagnostics.eval_negative << ",\n";
    output << indent << "  \"near_equal_minus_50_to_50\": " << diagnostics.eval_near_equal << ",\n";
    output << indent << "  \"positive_gt_50\": " << diagnostics.eval_positive << ",\n";
    output << indent << "  \"decisive_abs_ge_1000\": " << diagnostics.eval_decisive << "\n";
    output << indent << "},\n";
    output << indent << "\"side_to_move\": {\n";
    output << indent << "  \"white\": " << diagnostics.side_to_move_white << ",\n";
    output << indent << "  \"black\": " << diagnostics.side_to_move_black << "\n";
    output << indent << "},\n";
    output << indent << "\"material\": {\n";
    output << indent << "  \"pieces_0_8\": " << diagnostics.pieces_0_8 << ",\n";
    output << indent << "  \"pieces_9_16\": " << diagnostics.pieces_9_16 << ",\n";
    output << indent << "  \"pieces_17_24\": " << diagnostics.pieces_17_24 << ",\n";
    output << indent << "  \"pieces_25_32\": " << diagnostics.pieces_25_32 << "\n";
    output << indent << "},\n";
    output << indent << "\"depth\": {\n";
    output << indent << "  \"depth_1\": " << diagnostics.depth_1 << ",\n";
    output << indent << "  \"depth_2_3\": " << diagnostics.depth_2_3 << ",\n";
    output << indent << "  \"depth_4_plus\": " << diagnostics.depth_4_plus << "\n";
    output << indent << "},\n";
    output << indent << "\"nodes\": {\n";
    output << indent << "  \"nodes_0_999\": " << diagnostics.nodes_0_999 << ",\n";
    output << indent << "  \"nodes_1k_9999\": " << diagnostics.nodes_1k_9999 << ",\n";
    output << indent << "  \"nodes_10k_99999\": " << diagnostics.nodes_10k_99999 << ",\n";
    output << indent << "  \"nodes_100k_plus\": " << diagnostics.nodes_100k_plus << "\n";
    output << indent << "}\n";
}

[[nodiscard]] RecordDiagnostics
diagnostics_for_records(const eng::NnueDataGenerationResult& result) {
    RecordDiagnostics diagnostics;
    std::vector<std::uint32_t> records_per_game(
        static_cast<std::size_t>(result.stats.games_started));
    for (const auto& record : result.records) {
        if (record.eval_cp < -diagnostic_equal_margin_cp) {
            ++diagnostics.eval_negative;
        } else if (record.eval_cp > diagnostic_equal_margin_cp) {
            ++diagnostics.eval_positive;
        } else {
            ++diagnostics.eval_near_equal;
        }
        if (std::abs(record.eval_cp) >= diagnostic_decisive_cp) {
            ++diagnostics.eval_decisive;
        }

        if (record.ply <= diagnostic_ply_bucket_15) {
            ++diagnostics.ply_8_15;
        } else if (record.ply <= diagnostic_ply_bucket_31) {
            ++diagnostics.ply_16_31;
        } else if (record.ply <= diagnostic_ply_bucket_63) {
            ++diagnostics.ply_32_63;
        } else {
            ++diagnostics.ply_64_plus;
        }

        if (record.wdl.white_win_permille == eng::nnue_wdl_permille_total) {
            ++diagnostics.wdl_white;
        } else if (record.wdl.black_win_permille == eng::nnue_wdl_permille_total) {
            ++diagnostics.wdl_black;
        } else {
            ++diagnostics.wdl_draw;
        }
        if (side_to_move_from_fen(record.fen) == 'w') {
            ++diagnostics.side_to_move_white;
        } else if (side_to_move_from_fen(record.fen) == 'b') {
            ++diagnostics.side_to_move_black;
        }

        if (record.game_index < records_per_game.size()) {
            auto& count = records_per_game[static_cast<std::size_t>(record.game_index)];
            ++count;
            diagnostics.max_records_in_single_game =
                std::max<std::uint64_t>(diagnostics.max_records_in_single_game, count);
        }
    }
    diagnostics.games_with_records =
        static_cast<std::uint64_t>(std::count_if(records_per_game.begin(), records_per_game.end(),
                                                 [](std::uint32_t count) { return count != 0U; }));
    return diagnostics;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
bool write_manifest(const std::filesystem::path& path, const GenerateCommand& command,
                    const eng::NnueDataGenerationResult& result,
                    const std::filesystem::path& dataset_path,
                    const std::filesystem::path& preview_path, std::string_view dataset_id,
                    std::string_view date) {
    if (!ensure_parent_directory(path)) {
        return false;
    }
    std::ofstream output(path);
    if (!output) {
        return false;
    }

    const auto& stats = result.stats;
    const auto diagnostics = diagnostics_for_records(result);
    output << "{\n";
    output << R"(  "event": ")" << eng::nnue_dataset_manifest_event << "\",\n";
    output << "  \"schema\": \"localchessengine.nnue.dataset.manifest.v1\",\n";
    output << R"(  "dataset_id": ")" << json_escape(dataset_id) << "\",\n";
    output << R"(  "date": ")" << json_escape(date) << "\",\n";
    output << R"(  "source_task": ")" << json_escape(command.source_task) << "\",\n";
    output << R"(  "dataset_format": ")" << eng::nnue_dataset_format << "\",\n";
    output << R"(  "dataset": ")" << json_escape(dataset_path.string()) << "\",\n";
    output << R"(  "preview": ")" << json_escape(preview_path.string()) << "\",\n";
    output << "  \"positions\": " << result.records.size() << ",\n";
    output << R"(  "checksum": ")" << hex_u64(result.checksum) << "\",\n";
    output << R"(  "dataset_checksum": ")" << hex_u64(result.checksum) << "\",\n";
    output << "  \"external_data\": false,\n";
    output << "  \"primary_signal\": \"local_hce_search\",\n";
    output << "  \"strength_evidence\": false,\n";
    output << "  \"source_engine\": {\n";
    output << "    \"name\": \"LocalChessEngine\",\n";
    output << "    \"label_source\": \"own_engine_search_and_hce\",\n";
    output << "    \"search_depth\": " << command.config.depth.value() << "\n";
    output << "  },\n";
    output << "  \"generation\": {\n";
    output << "    \"positions_requested\": " << command.config.positions << ",\n";
    output << "    \"positions_accepted\": " << result.records.size() << ",\n";
    output << "    \"games_requested\": " << command.config.games << ",\n";
    output << "    \"games_started\": " << stats.games_started << ",\n";
    output << "    \"games_completed\": " << stats.games_completed << ",\n";
    output << "    \"max_plies\": " << command.config.max_plies << ",\n";
    output << "    \"plies_played\": " << stats.plies_played << ",\n";
    output << "    \"seed\": " << command.config.seed << ",\n";
    output << R"(    "playout_policy": ")"
           << eng::nnue_data_playout_policy_name(command.config.playout_policy) << "\",\n";
    output << "    \"random_playout_permille\": " << command.config.random_playout_permille
           << ",\n";
    output << "    \"top_n_playout_permille\": " << command.config.top_n_playout_permille << ",\n";
    output << "    \"top_n_playout\": " << command.config.top_n_playout << ",\n";
    output << "    \"max_records_per_game\": " << command.config.max_records_per_game << ",\n";
    output << "    \"accepted_sample_stride\": " << command.config.accepted_sample_stride << "\n";
    output << "  },\n";
    output << "  \"filters\": {\n";
    output << "    \"min_ply\": " << command.config.min_ply << ",\n";
    output << "    \"max_abs_eval_cp\": " << command.config.max_abs_eval_cp << ",\n";
    output << "    \"skip_in_check\": true,\n";
    output << "    \"skip_best_move_capture\": true,\n";
    output << "    \"positions_considered\": " << stats.positions_considered << ",\n";
    output << "    \"accepted_by_filter\": " << stats.accepted << ",\n";
    output << "    \"filtered_total\": " << stats.filtered() << ",\n";
    output << "    \"sampled_out_by_stride\": " << stats.sampled_out_by_stride << ",\n";
    output << "    \"sampled_out_by_game_limit\": " << stats.sampled_out_by_game_limit << ",\n";
    output << "    \"reasons\": {\n";
    output << "      \"opening_ply\": " << stats.filtered_opening_ply << ",\n";
    output << "      \"in_check\": " << stats.filtered_in_check << ",\n";
    output << "      \"best_move_capture\": " << stats.filtered_best_move_capture << ",\n";
    output << "      \"eval_out_of_range\": " << stats.filtered_eval_out_of_range << ",\n";
    output << "      \"terminal_or_no_bestmove\": " << stats.filtered_terminal_or_no_bestmove
           << "\n";
    output << "    }\n";
    output << "  },\n";
    output << "  \"wdl\": {\n";
    output << "    \"orientation\": \"white_relative\",\n";
    output << "    \"unit\": \"permille\",\n";
    output << "    \"adjudication_draw_cp\": " << command.config.adjudication_draw_cp << "\n";
    output << "  },\n";
    output << "  \"diagnostics\": {\n";
    output << "    \"eval_buckets\": {\n";
    output << "      \"negative_lt_minus_50\": " << diagnostics.eval_negative << ",\n";
    output << "      \"near_equal_minus_50_to_50\": " << diagnostics.eval_near_equal << ",\n";
    output << "      \"positive_gt_50\": " << diagnostics.eval_positive << ",\n";
    output << "      \"decisive_abs_ge_1000\": " << diagnostics.eval_decisive << "\n";
    output << "    },\n";
    output << "    \"ply_buckets\": {\n";
    output << "      \"8_15\": " << diagnostics.ply_8_15 << ",\n";
    output << "      \"16_31\": " << diagnostics.ply_16_31 << ",\n";
    output << "      \"32_63\": " << diagnostics.ply_32_63 << ",\n";
    output << "      \"64_plus\": " << diagnostics.ply_64_plus << "\n";
    output << "    },\n";
    output << "    \"wdl_buckets\": {\n";
    output << "      \"white_win\": " << diagnostics.wdl_white << ",\n";
    output << "      \"draw\": " << diagnostics.wdl_draw << ",\n";
    output << "      \"black_win\": " << diagnostics.wdl_black << "\n";
    output << "    },\n";
    output << "    \"side_to_move\": {\n";
    output << "      \"white\": " << diagnostics.side_to_move_white << ",\n";
    output << "      \"black\": " << diagnostics.side_to_move_black << "\n";
    output << "    },\n";
    output << "    \"games_with_records\": " << diagnostics.games_with_records << ",\n";
    output << "    \"max_records_in_single_game\": " << diagnostics.max_records_in_single_game
           << "\n";
    output << "  },\n";
    output << R"(  "notes": ")" << json_escape(command.notes) << "\"\n";
    output << "}\n";
    return static_cast<bool>(output);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
bool write_relabel_split_manifest(const std::filesystem::path& path,
                                  const RelabelFensCommand& command,
                                  std::span<const eng::NnueTrainingRecord> records,
                                  const std::filesystem::path& dataset_path,
                                  std::string_view dataset_id, std::string_view split_name,
                                  std::uint64_t checksum, std::string_view date) {
    if (path.empty()) {
        return true;
    }
    if (!ensure_parent_directory(path)) {
        return false;
    }
    std::ofstream output(path);
    if (!output) {
        return false;
    }

    output << "{\n";
    output << R"(  "event": ")" << eng::nnue_dataset_manifest_event << "\",\n";
    output << "  \"schema\": \"localchessengine.nnue.dataset.manifest.v1\",\n";
    output << R"(  "dataset_id": ")" << json_escape(dataset_id) << "\",\n";
    output << R"(  "date": ")" << json_escape(date) << "\",\n";
    output << R"(  "source_task": ")" << json_escape(command.source_task) << "\",\n";
    output << R"(  "dataset_format": ")" << eng::nnue_dataset_format << "\",\n";
    output << R"(  "dataset": ")" << json_escape(dataset_path.string()) << "\",\n";
    output << "  \"positions\": " << records.size() << ",\n";
    output << R"(  "checksum": ")" << hex_u64(checksum) << "\",\n";
    output << R"(  "dataset_checksum": ")" << hex_u64(checksum) << "\",\n";
    output << "  \"external_data\": false,\n";
    output << "  \"external_position_source\": true,\n";
    output << "  \"strength_evidence\": false,\n";
    output << "  \"source\": {\n";
    output << "    \"kind\": \"local_relabel_from_lichess_fen_pool\",\n";
    output << R"(    "fen_pool": ")" << json_escape(command.input.string()) << "\",\n";
    output << "    \"stockfish_labels_present\": false,\n";
    output << "    \"stockfish_labels_used\": false,\n";
    output << "    \"policy\": \"FENs only; labels regenerated locally to preserve PRD BR-005.\"\n";
    output << "  },\n";
    output << "  \"source_engine\": {\n";
    output << "    \"name\": \"LocalChessEngine\",\n";
    output << "    \"label_source\": \"own_engine_search_and_hce\",\n";
    output << "    \"search_depth\": " << command.config.depth.value() << ",\n";
    output << "    \"search_nodes\": " << command.config.nodes << ",\n";
    output << "    \"hash_mb\": " << command.config.hash_mb << ",\n";
    output << "    \"threads\": 1,\n";
    output << "    \"use_nnue\": false\n";
    output << "  },\n";
    output << R"(  "split": ")" << json_escape(split_name) << "\",\n";
    output << "  \"generation\": {\n";
    output << "    \"positions_requested\": " << records.size() << ",\n";
    output << "    \"positions_accepted\": " << records.size() << ",\n";
    output << "    \"source_kind\": \"local_relabel_from_lichess_fen_pool\",\n";
    output << "    \"label_source\": \"local_engine_search\"\n";
    output << "  },\n";
    output << "  \"filters\": {\n";
    output << "    \"min_ply\": 0,\n";
    output << "    \"max_abs_eval_cp\": " << command.config.max_abs_eval_cp << ",\n";
    output << "    \"skip_in_check\": " << (command.config.skip_in_check ? "true" : "false")
           << ",\n";
    output << "    \"skip_best_move_capture\": "
           << (command.config.skip_best_move_capture ? "true" : "false") << ",\n";
    output << "    \"skip_terminal_or_no_bestmove\": true,\n";
    output << "    \"min_pieces\": " << command.config.min_pieces << ",\n";
    output << "    \"max_pieces\": " << command.config.max_pieces << ",\n";
    output << "    \"stability_depth_delta\": " << command.config.stability_depth_delta << ",\n";
    output << "    \"max_stability_delta_cp\": " << command.config.max_stability_delta_cp << "\n";
    output << "  },\n";
    output << "  \"wdl\": {\n";
    output << "    \"orientation\": \"white_relative\",\n";
    output << "    \"unit\": \"permille\",\n";
    output << "    \"source\": \"derived_from_local_eval_cp_logistic\"\n";
    output << "  },\n";
    output << "  \"diagnostics\": {\n";
    write_relabel_diagnostics(output, records, "    ");
    output << "  },\n";
    output << R"(  "notes": ")" << json_escape(command.notes) << "\"\n";
    output << "}\n";
    return static_cast<bool>(output);
}

bool write_relabel_manifest(const std::filesystem::path& path, const RelabelFensCommand& command,
                            const eng::NnueFenRelabelResult& result, std::string_view date) {
    if (!ensure_parent_directory(path)) {
        return false;
    }
    std::ofstream output(path);
    if (!output) {
        return false;
    }
    const auto& stats = result.stats;
    output << "{\n";
    output << R"(  "event": ")" << eng::nnue_dataset_manifest_event << "\",\n";
    output << "  \"schema\": \"localchessengine.nnue.local_relabel.manifest.v2\",\n";
    output << R"(  "dataset_id": ")" << json_escape(command.dataset_id) << "\",\n";
    output << R"(  "date": ")" << json_escape(date) << "\",\n";
    output << R"(  "source_task": ")" << json_escape(command.source_task) << "\",\n";
    output << R"(  "dataset_format": ")" << eng::nnue_dataset_format << "\",\n";
    output << "  \"external_data\": false,\n";
    output << "  \"external_position_source\": true,\n";
    output << "  \"strength_evidence\": false,\n";
    output << "  \"source\": {\n";
    output << "    \"kind\": \"local_relabel_from_lichess_fen_pool\",\n";
    output << R"(    "fen_pool": ")" << json_escape(command.input.string()) << "\",\n";
    output << "    \"stockfish_labels_present\": false,\n";
    output << "    \"stockfish_labels_used\": false,\n";
    output << "    \"policy\": \"FENs only; labels regenerated locally to preserve PRD BR-005.\"\n";
    output << "  },\n";
    output << "  \"source_engine\": {\n";
    output << "    \"name\": \"LocalChessEngine\",\n";
    output << "    \"label_source\": \"own_engine_search_and_hce\",\n";
    output << "    \"search_depth\": " << command.config.depth.value() << ",\n";
    output << "    \"search_nodes\": " << command.config.nodes << ",\n";
    output << "    \"hash_mb\": " << command.config.hash_mb << ",\n";
    output << "    \"threads\": 1,\n";
    output << "    \"use_nnue\": false\n";
    output << "  },\n";
    output << "  \"artifacts\": {\n";
    output << R"(    "train": {"path": ")" << json_escape(command.train_output.string())
           << R"(", "positions": )" << result.train_records.size() << R"(, "checksum": ")"
           << hex_u64(result.train_checksum) << R"("},)" << '\n';
    output << R"(    "holdout": {"path": ")" << json_escape(command.holdout_output.string())
           << R"(", "positions": )" << result.holdout_records.size() << R"(, "checksum": ")"
           << hex_u64(result.holdout_checksum) << R"("},)" << '\n';
    output << R"(    "match_openings": {"path": ")"
           << json_escape(command.match_openings_output.string()) << R"(", "positions": )"
           << result.match_opening_records.size() << R"(, "checksum": ")"
           << hex_u64(result.match_openings_checksum) << R"("})" << '\n';
    output << "  },\n";
    output << "  \"filters\": {\n";
    output << "    \"max_abs_eval_cp\": " << command.config.max_abs_eval_cp << ",\n";
    output << "    \"skip_in_check\": " << (command.config.skip_in_check ? "true" : "false")
           << ",\n";
    output << "    \"skip_best_move_capture\": "
           << (command.config.skip_best_move_capture ? "true" : "false") << ",\n";
    output << "    \"skip_terminal_or_no_bestmove\": true,\n";
    output << "    \"min_pieces\": " << command.config.min_pieces << ",\n";
    output << "    \"max_pieces\": " << command.config.max_pieces << ",\n";
    output << "    \"holdout_permille\": " << command.config.holdout_permille << ",\n";
    output << "    \"match_permille\": " << command.config.match_permille << ",\n";
    output << "    \"stability_depth_delta\": " << command.config.stability_depth_delta << ",\n";
    output << "    \"max_stability_delta_cp\": " << command.config.max_stability_delta_cp << "\n";
    output << "  },\n";
    output << "  \"stats\": {\n";
    output << "    \"source_fens_seen\": " << stats.source_fens_seen << ",\n";
    output << "    \"accepted_train\": " << stats.accepted_train << ",\n";
    output << "    \"accepted_holdout\": " << stats.accepted_holdout << ",\n";
    output << "    \"accepted_match_openings\": " << stats.accepted_match_openings << ",\n";
    output << "    \"filtered_total\": " << stats.filtered() << ",\n";
    output << "    \"filtered_invalid_fen\": " << stats.filtered_invalid_fen << ",\n";
    output << "    \"filtered_duplicate_fen\": " << stats.filtered_duplicate_fen << ",\n";
    output << "    \"filtered_split_full\": " << stats.filtered_split_full << ",\n";
    output << "    \"filtered_material\": " << stats.filtered_material << ",\n";
    output << "    \"filtered_in_check\": " << stats.filtered_in_check << ",\n";
    output << "    \"filtered_best_move_capture\": " << stats.filtered_best_move_capture << ",\n";
    output << "    \"filtered_eval_out_of_range\": " << stats.filtered_eval_out_of_range << ",\n";
    output << "    \"filtered_terminal_or_no_bestmove\": " << stats.filtered_terminal_or_no_bestmove
           << ",\n";
    output << "    \"filtered_unstable_eval\": " << stats.filtered_unstable_eval << "\n";
    output << "  },\n";
    output << "  \"diagnostics\": {\n";
    output << "    \"train\": {\n";
    write_relabel_diagnostics(output, result.train_records, "      ");
    output << "    },\n";
    output << "    \"holdout\": {\n";
    write_relabel_diagnostics(output, result.holdout_records, "      ");
    output << "    },\n";
    output << "    \"match_openings\": {\n";
    write_relabel_diagnostics(output, result.match_opening_records, "      ");
    output << "    }\n";
    output << "  },\n";
    output << R"(  "notes": ")" << json_escape(command.notes) << "\"\n";
    output << "}\n";
    return static_cast<bool>(output);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

int run_generate(const GenerateCommand& command) {
    const auto result = eng::generate_nnue_dataset(command.config);
    if (result.records.empty()) {
        std::cerr << "error: generation produced no accepted records\n";
        return runtime_error;
    }

    const auto written = eng::write_nnue_dataset(command.output, result.records);
    if (!written.has_value()) {
        std::cerr << "error: " << written.error().message << '\n';
        return runtime_error;
    }

    const auto read_back = eng::read_nnue_dataset(command.output);
    if (!read_back.has_value()) {
        std::cerr << "error: generated dataset failed readback: " << read_back.error().message
                  << '\n';
        return runtime_error;
    }
    if (read_back->checksum != result.checksum) {
        std::cerr << "error: generated dataset checksum changed during readback\n";
        return runtime_error;
    }

    const auto dataset_id =
        command.dataset_id.empty() ? default_dataset_id(result.checksum) : command.dataset_id;
    const auto resolved_date = command.date.empty()
                                   ? resolve_manifest_date()
                                   : std::expected<std::string, std::string>{command.date};
    if (!resolved_date.has_value()) {
        std::cerr << "error: " << resolved_date.error() << '\n';
        return runtime_error;
    }

    if (!write_preview(command.preview, result)) {
        std::cerr << "error: could not write preview: " << command.preview << '\n';
        return runtime_error;
    }
    if (!write_manifest(command.manifest, command, result, command.output, command.preview,
                        dataset_id, *resolved_date)) {
        std::cerr << "error: could not write manifest: " << command.manifest << '\n';
        return runtime_error;
    }

    std::cout << "dataset id " << dataset_id << '\n';
    std::cout << "manifest date " << *resolved_date << '\n';
    std::cout << "records " << result.records.size() << '\n';
    std::cout << "positions considered " << result.stats.positions_considered << '\n';
    std::cout << "filtered " << result.stats.filtered() << '\n';
    std::cout << "dataset checksum " << hex_u64(result.checksum) << '\n';
    return success;
}

int run_relabel_fens(const RelabelFensCommand& command) {
    const auto fens = read_fen_pool(command.input);
    if (!fens.has_value()) {
        std::cerr << "error: " << fens.error() << '\n';
        return runtime_error;
    }

    const auto result = eng::relabel_nnue_fens(*fens, command.config);
    if (result.train_records.size() < command.config.train_positions ||
        result.holdout_records.size() < command.config.holdout_positions ||
        result.match_opening_records.size() < command.config.match_openings) {
        std::cerr << "error: relabel-fens produced insufficient records"
                  << " train=" << result.train_records.size() << "/"
                  << command.config.train_positions << " holdout=" << result.holdout_records.size()
                  << "/" << command.config.holdout_positions
                  << " openings=" << result.match_opening_records.size() << "/"
                  << command.config.match_openings << " filtered=" << result.stats.filtered()
                  << '\n';
        return runtime_error;
    }

    const auto resolved_date = command.date.empty()
                                   ? resolve_manifest_date()
                                   : std::expected<std::string, std::string>{command.date};
    if (!resolved_date.has_value()) {
        std::cerr << "error: " << resolved_date.error() << '\n';
        return runtime_error;
    }

    const auto write_train = eng::write_nnue_dataset(command.train_output, result.train_records);
    if (!write_train.has_value()) {
        std::cerr << "error: " << write_train.error().message << '\n';
        return runtime_error;
    }
    if (!command.holdout_output.empty()) {
        const auto write_holdout =
            eng::write_nnue_dataset(command.holdout_output, result.holdout_records);
        if (!write_holdout.has_value()) {
            std::cerr << "error: " << write_holdout.error().message << '\n';
            return runtime_error;
        }
    }
    if (!write_match_openings(command.match_openings_output, result.match_opening_records)) {
        std::cerr << "error: could not write match openings: " << command.match_openings_output
                  << '\n';
        return runtime_error;
    }
    if (!write_records_preview(command.train_preview, result.train_records,
                               command.preview_limit) ||
        !write_records_preview(command.holdout_preview, result.holdout_records,
                               command.preview_limit) ||
        !write_records_preview(command.match_preview, result.match_opening_records,
                               command.preview_limit)) {
        std::cerr << "error: could not write relabel preview\n";
        return runtime_error;
    }
    if (!write_relabel_manifest(command.manifest, command, result, *resolved_date)) {
        std::cerr << "error: could not write relabel manifest: " << command.manifest << '\n';
        return runtime_error;
    }
    if (!write_relabel_split_manifest(command.train_manifest, command, result.train_records,
                                      command.train_output, command.dataset_id + "-train", "train",
                                      result.train_checksum, *resolved_date) ||
        !write_relabel_split_manifest(command.holdout_manifest, command, result.holdout_records,
                                      command.holdout_output, command.dataset_id + "-holdout",
                                      "holdout", result.holdout_checksum, *resolved_date)) {
        std::cerr << "error: could not write split relabel manifest\n";
        return runtime_error;
    }

    std::cout << "dataset id " << command.dataset_id << '\n';
    std::cout << "manifest date " << *resolved_date << '\n';
    std::cout << "source fens " << fens->size() << '\n';
    std::cout << "train records " << result.train_records.size() << " checksum "
              << hex_u64(result.train_checksum) << '\n';
    std::cout << "holdout records " << result.holdout_records.size() << " checksum "
              << hex_u64(result.holdout_checksum) << '\n';
    std::cout << "match openings " << result.match_opening_records.size() << " checksum "
              << hex_u64(result.match_openings_checksum) << '\n';
    std::cout << "filtered " << result.stats.filtered() << '\n';
    return success;
}

int run(int argc, char** argv) {
    if (argc < 2) {
        print_usage(std::cerr);
        return usage_error;
    }

    const std::string_view mode{argv[1]};
    if (mode == "generate") {
        const auto command = parse_generate(argc, argv);
        if (!command.has_value()) {
            std::cerr << "error: " << command.error() << "\n\n";
            print_usage(std::cerr);
            return usage_error;
        }
        return run_generate(*command);
    }
    if (mode == "relabel-fens") {
        const auto command = parse_relabel_fens(argc, argv);
        if (!command.has_value()) {
            std::cerr << "error: " << command.error() << "\n\n";
            print_usage(std::cerr);
            return usage_error;
        }
        return run_relabel_fens(*command);
    }

    std::cerr << "error: unknown mode: " << mode << "\n\n";
    print_usage(std::cerr);
    return usage_error;
}

} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
int main(int argc, char** argv) { return run(argc, argv); }
