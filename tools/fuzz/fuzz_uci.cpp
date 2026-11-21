#include <eng/bitboard.hpp>
#include <eng/uci.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

namespace {

constexpr std::size_t max_input_bytes = 4096;
constexpr std::size_t max_line_bytes = 512;
constexpr int max_lines_per_input = 32;
constexpr std::size_t max_output_bytes = std::size_t{64} * std::size_t{1024};

[[nodiscard]] bool is_ascii_space(char symbol) noexcept {
    return symbol == ' ' || symbol == '\t' || symbol == '\n' || symbol == '\r' || symbol == '\f' ||
           symbol == '\v';
}

[[nodiscard]] char to_lower_ascii(char symbol) noexcept {
    if (symbol >= 'A' && symbol <= 'Z') {
        return static_cast<char>(symbol - 'A' + 'a');
    }
    return symbol;
}

[[nodiscard]] std::string_view trim(std::string_view text) noexcept {
    while (!text.empty() && is_ascii_space(text.front())) {
        text.remove_prefix(1);
    }
    while (!text.empty() && is_ascii_space(text.back())) {
        text.remove_suffix(1);
    }
    return text;
}

[[nodiscard]] std::string lower_copy(std::string_view text) {
    std::string lowered;
    lowered.reserve(text.size());
    for (const auto symbol : text) {
        lowered.push_back(to_lower_ascii(symbol));
    }
    return lowered;
}

[[nodiscard]] std::string_view first_token(std::string_view line) noexcept {
    line = trim(line);
    const auto end = line.find_first_of(" \t\r\n\f\v");
    if (end == std::string_view::npos) {
        return line;
    }
    line.remove_suffix(line.size() - end);
    return line;
}

[[nodiscard]] bool equals_ci(std::string_view lhs, std::string_view rhs) {
    return lhs.size() == rhs.size() &&
           std::ranges::equal(lhs, rhs, [](char left, char right) noexcept {
               return to_lower_ascii(left) == to_lower_ascii(right);
           });
}

[[nodiscard]] bool contains_ci(std::string_view haystack, std::string_view needle) {
    return lower_copy(haystack).contains(needle);
}

void clear_output(std::ostringstream& output) {
    output.str(std::string{});
    output.clear();
}

void dispatch_bounded_line(eng::UciSession& session, std::ostringstream& output,
                           std::string_view line) {
    line = trim(line);
    if (line.empty() || line.size() > max_line_bytes) {
        return;
    }

    const auto command = first_token(line);
    if (equals_ci(command, "go")) {
        return;
    }
    if (equals_ci(command, "setoption")) {
        if (contains_ci(line, "hash")) {
            session.handle_line("setoption name Hash value 1");
            return;
        }
        if (contains_ci(line, "evalfile")) {
            session.handle_line(
                "setoption name EvalFile value /tmp/localchessengine-fuzz-missing.nnue");
            return;
        }
    }

    session.handle_line(line);
    if (output.tellp() > static_cast<std::streampos>(max_output_bytes)) {
        clear_output(output);
    }
}

} // namespace

// NOLINTNEXTLINE(readability-non-const-parameter): libFuzzer requires this signature.
extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
    (void)argc;
    (void)argv;
    eng::init_bitboards();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (data == nullptr || size == 0U || size > max_input_bytes) {
        return 0;
    }

    static std::istringstream input;
    static std::ostringstream output;
    static eng::UciSession session{input, output};
    static bool initialized = false;
    if (!initialized) {
        session.handle_line("setoption name Hash value 1");
        initialized = true;
    }

    const std::string text{reinterpret_cast<const char*>(data), size};
    std::size_t offset = 0;
    int lines = 0;
    while (offset <= text.size() && lines < max_lines_per_input) {
        const auto end = text.find('\n', offset);
        const auto line_end = end == std::string::npos ? text.size() : end;
        dispatch_bounded_line(session, output,
                              std::string_view{text}.substr(offset, line_end - offset));
        ++lines;
        if (end == std::string::npos) {
            break;
        }
        offset = end + 1U;
    }

    return 0;
}
