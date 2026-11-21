#include <eng/perft.hpp>
#include <eng/position.hpp>

#include <charconv>
#include <iostream>
#include <string>
#include <string_view>

namespace {

constexpr int success = 0;
constexpr int usage_error = 2;
constexpr int input_error = 3;
constexpr int perft_command_index = 1;
constexpr int depth_index = 2;
constexpr int fen_start_index = 3;

[[nodiscard]] std::string_view usage_text() noexcept {
    return "usage:\n"
           "  localchessengine_perft perft <depth> [fen...]\n"
           "  localchessengine_perft divide <depth> [fen...]\n";
}

[[nodiscard]] bool parse_depth(std::string_view text, int& depth) noexcept {
    const auto* first = text.data();
    const auto* last = text.data() + text.size();
    int parsed = 0;
    const auto [ptr, error] = std::from_chars(first, last, parsed);
    if (error != std::errc{} || ptr != last || parsed < 0) {
        return false;
    }

    depth = parsed;
    return true;
}

[[nodiscard]] std::string parse_fen(int argc, char** argv) {
    if (argc <= fen_start_index) {
        return std::string{eng::Position::startpos_fen};
    }

    std::string fen;
    for (int index = fen_start_index; index < argc; ++index) {
        if (!fen.empty()) {
            fen.push_back(' ');
        }
        fen += argv[index];
    }
    return fen;
}

[[nodiscard]] char promotion_char(eng::PieceType type) noexcept {
    switch (type) {
    case eng::PieceType::Knight:
        return 'n';
    case eng::PieceType::Bishop:
        return 'b';
    case eng::PieceType::Rook:
        return 'r';
    case eng::PieceType::Queen:
        return 'q';
    case eng::PieceType::Pawn:
    case eng::PieceType::King:
    case eng::PieceType::None:
        return '\0';
    }
    return '\0';
}

[[nodiscard]] std::string square_name(eng::Square square) {
    if (!square.is_valid()) {
        return "??";
    }

    std::string result;
    result.push_back(static_cast<char>('a' + square.file().index()));
    result.push_back(static_cast<char>('1' + square.rank().index()));
    return result;
}

[[nodiscard]] std::string move_name(eng::Move move) {
    auto result = square_name(move.from()) + square_name(move.to());
    if (move.is_promotion()) {
        result.push_back(promotion_char(move.promotion_piece()));
    }
    return result;
}

} // namespace

// Command-line diagnostics may allocate strings; report failures through exit codes.
// NOLINTNEXTLINE(bugprone-exception-escape)
int main(int argc, char** argv) {
    if (argc < fen_start_index) {
        std::cerr << usage_text();
        return usage_error;
    }

    const auto command = std::string_view{argv[perft_command_index]};
    int depth = 0;
    if (!parse_depth(argv[depth_index], depth)) {
        std::cerr << "invalid depth: " << argv[depth_index] << '\n';
        return usage_error;
    }
    if (command == "divide" && depth == 0) {
        std::cerr << "divide depth must be positive\n";
        return usage_error;
    }

    const auto fen = parse_fen(argc, argv);
    auto parsed_position = eng::Position::from_fen(fen);
    if (!parsed_position.has_value()) {
        std::cerr << "invalid FEN: " << parsed_position.error().message << '\n';
        return input_error;
    }

    auto position = *parsed_position;
    if (command == "perft") {
        std::cout << "fen: " << fen << '\n';
        std::cout << "depth: " << depth << '\n';
        std::cout << "nodes: " << eng::perft(position, depth) << '\n';
        return success;
    }

    if (command == "divide") {
        const auto entries = eng::divide(position, depth);
        eng::PerftNodeCount total = 0;
        for (const auto& entry : entries) {
            std::cout << move_name(entry.move) << ": " << entry.nodes << '\n';
            total += entry.nodes;
        }
        std::cout << "total: " << total << '\n';
        return success;
    }

    std::cerr << "unknown command: " << command << '\n';
    std::cerr << usage_text();
    return usage_error;
}
