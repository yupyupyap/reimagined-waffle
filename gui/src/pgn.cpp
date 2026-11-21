#include "pgn.hpp"

#include "eng/movegen.hpp"
#include "eng/rules.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <optional>
#include <span>
#include <string>
#include <utility>

namespace gui::pgn {
namespace {

constexpr int kPgnWrapColumn = 80;
constexpr std::array<std::string_view, 7> kSevenTagRoster{
    "Event", "Site", "Date", "Round", "White", "Black", "Result",
};

[[nodiscard]] constexpr bool is_ascii_space(char symbol) noexcept {
    return symbol == ' ' || symbol == '\t' || symbol == '\n' || symbol == '\r' || symbol == '\f' ||
           symbol == '\v';
}

[[nodiscard]] constexpr bool is_ascii_digit(char symbol) noexcept {
    return symbol >= '0' && symbol <= '9';
}

[[nodiscard]] constexpr bool is_ascii_alpha(char symbol) noexcept {
    return (symbol >= 'A' && symbol <= 'Z') || (symbol >= 'a' && symbol <= 'z');
}

[[nodiscard]] constexpr char to_lower_ascii(char symbol) noexcept {
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

[[nodiscard]] Error make_error(ErrorCode code, std::string message, std::size_t offset) {
    return Error{.code = code, .message = std::move(message), .offset = offset};
}

[[nodiscard]] bool is_result_token(std::string_view token) noexcept {
    return token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*";
}

[[nodiscard]] bool is_tag_name_char(char symbol) noexcept {
    return is_ascii_alpha(symbol) || is_ascii_digit(symbol) || symbol == '_' || symbol == '-' ||
           symbol == ':' || symbol == '+' || symbol == '#' || symbol == '=';
}

[[nodiscard]] bool is_symbol_boundary(char symbol) noexcept {
    return is_ascii_space(symbol) || symbol == '{' || symbol == '}' || symbol == '(' ||
           symbol == ')' || symbol == ';' || symbol == '[' || symbol == ']';
}

[[nodiscard]] bool is_move_number_token(std::string_view token) noexcept {
    bool saw_dot = false;
    bool saw_digit = false;
    for (const auto symbol : token) {
        if (symbol == '.') {
            saw_dot = true;
        } else if (is_ascii_digit(symbol)) {
            saw_digit = true;
        } else {
            return false;
        }
    }
    return saw_dot && saw_digit;
}

[[nodiscard]] std::string strip_move_number_prefix(std::string token) {
    std::size_t index = 0;
    while (index < token.size() && is_ascii_digit(token[index])) {
        ++index;
    }
    if (index == 0 || index >= token.size() || token[index] != '.') {
        return token;
    }
    while (index < token.size() && token[index] == '.') {
        ++index;
    }
    return token.substr(index);
}

[[nodiscard]] std::string sanitize_san_token(std::string_view token) {
    token = trim(token);
    std::string normalized;
    normalized.reserve(token.size());
    for (const auto symbol : token) {
        normalized.push_back(symbol == '0' ? 'O' : symbol);
    }

    while (!normalized.empty() && (normalized.back() == '!' || normalized.back() == '?' ||
                                   is_ascii_space(normalized.back()))) {
        normalized.pop_back();
    }

    const auto lower_copy = [&normalized]() {
        std::string lower;
        lower.reserve(normalized.size());
        for (const auto symbol : normalized) {
            lower.push_back(to_lower_ascii(symbol));
        }
        return lower;
    };

    auto lower = lower_copy();
    if (lower.ends_with("e.p.")) {
        normalized.resize(normalized.size() - 4U);
    } else if (lower.ends_with("ep")) {
        normalized.resize(normalized.size() - 2U);
    }

    while (!normalized.empty() && is_ascii_space(normalized.back())) {
        normalized.pop_back();
    }
    return normalized;
}

[[nodiscard]] std::string strip_check_suffix(std::string text) {
    while (!text.empty() && (text.back() == '+' || text.back() == '#')) {
        text.pop_back();
    }
    return text;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] bool san_matches(std::string_view input, std::string_view generated) {
    const std::string normalized_input = sanitize_san_token(input);
    const std::string normalized_generated = sanitize_san_token(generated);
    return normalized_input == normalized_generated ||
           strip_check_suffix(normalized_input) == strip_check_suffix(normalized_generated);
}

[[nodiscard]] std::string square_name(eng::Square square) {
    std::string result;
    result.push_back(static_cast<char>('a' + square.file().index()));
    result.push_back(static_cast<char>('1' + square.rank().index()));
    return result;
}

[[nodiscard]] char file_char(eng::Square square) noexcept {
    return static_cast<char>('a' + square.file().index());
}

[[nodiscard]] char rank_char(eng::Square square) noexcept {
    return static_cast<char>('1' + square.rank().index());
}

[[nodiscard]] char piece_letter(eng::PieceType type) noexcept {
    switch (type) {
    case eng::PieceType::Knight:
        return 'N';
    case eng::PieceType::Bishop:
        return 'B';
    case eng::PieceType::Rook:
        return 'R';
    case eng::PieceType::Queen:
        return 'Q';
    case eng::PieceType::King:
        return 'K';
    case eng::PieceType::Pawn:
    case eng::PieceType::None:
        break;
    }
    return '\0';
}

[[nodiscard]] bool is_capture(const eng::Position& position, eng::Move move) {
    return move.is_en_passant() || position.piece_on(move.to()).is_valid();
}

[[nodiscard]] bool same_header_name(std::string_view lhs, std::string_view rhs) noexcept {
    return lhs == rhs;
}

void upsert_header(std::vector<Header>& headers, std::string name, std::string value) {
    const auto found = std::ranges::find_if(headers, [&name](const Header& header) noexcept {
        return same_header_name(header.name, name);
    });
    if (found == headers.end()) {
        headers.push_back(Header{.name = std::move(name), .value = std::move(value)});
    } else {
        found->value = std::move(value);
    }
}

[[nodiscard]] std::vector<Header> merge_with_defaults(std::span<const Header> input_headers,
                                                      std::string_view result) {
    auto merged = default_headers(result);
    for (const auto& header : input_headers) {
        upsert_header(merged, header.name, header.value);
    }
    upsert_header(merged, "Result", std::string{result});
    return merged;
}

[[nodiscard]] bool is_roster_tag(std::string_view name) noexcept {
    return std::ranges::find(kSevenTagRoster, name) != kSevenTagRoster.end();
}

[[nodiscard]] std::string escape_tag_value(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const auto symbol : value) {
        if (symbol == '\\' || symbol == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(symbol);
    }
    return escaped;
}

void append_wrapped_token(std::string& output, int& column, std::string_view token) {
    if (column == 0) {
        output.append(token);
        column = static_cast<int>(token.size());
        return;
    }
    if (column + 1 + static_cast<int>(token.size()) > kPgnWrapColumn) {
        output.push_back('\n');
        output.append(token);
        column = static_cast<int>(token.size());
    } else {
        output.push_back(' ');
        output.append(token);
        column += 1 + static_cast<int>(token.size());
    }
}

[[nodiscard]] std::optional<eng::Move> resolve_san(const eng::Position& position,
                                                   std::string_view token, Error& error) {
    const auto legal_moves = eng::generate_legal_moves(position);
    std::optional<eng::Move> matched;
    for (const auto move : legal_moves) {
        const auto san = san_for_move(position, move);
        if (!san.has_value()) {
            continue;
        }
        if (!san_matches(token, *san)) {
            continue;
        }
        if (matched.has_value()) {
            error = make_error(ErrorCode::AmbiguousMove,
                               "PGN move is ambiguous after SAN normalization.", 0);
            return std::nullopt;
        }
        matched = move;
    }
    if (!matched.has_value()) {
        error =
            make_error(ErrorCode::IllegalMove, "PGN move is not legal in the current position.", 0);
    }
    return matched;
}

class Parser final {
  public:
    explicit Parser(std::string_view text) : text_{text} {}

    // NOLINTNEXTLINE(readability-function-cognitive-complexity)
    [[nodiscard]] ParseResult parse() {
        if (text_.empty()) {
            return std::unexpected{
                make_error(ErrorCode::EmptyInput, "PGN input is empty.", offset_)};
        }
        if (text_.size() > max_pgn_bytes) {
            return std::unexpected{make_error(ErrorCode::InputTooLarge,
                                              "PGN input exceeds the supported size limit.", 0)};
        }

        while (true) {
            const auto skipped = skip_space_and_comments();
            if (!skipped.has_value()) {
                return std::unexpected{skipped.error()};
            }
            if (peek() != '[') {
                break;
            }
            auto tag = parse_tag();
            if (!tag.has_value()) {
                return std::unexpected{tag.error()};
            }
            upsert_header(headers_, std::move(tag->name), std::move(tag->value));
        }

        auto setup = std::string{header_value(headers_, "SetUp")};
        const auto fen = header_value(headers_, "FEN");
        if (!setup.empty() && setup != "0" && setup != "1") {
            return std::unexpected{
                make_error(ErrorCode::MalformedTag, "PGN SetUp header must be 0 or 1.", 0)};
        }
        if (setup == "1" && fen.empty()) {
            return std::unexpected{
                make_error(ErrorCode::MissingFen, "PGN SetUp header requires a FEN header.", 0)};
        }
        if (setup == "0" && !fen.empty()) {
            return std::unexpected{
                make_error(ErrorCode::MalformedTag, "PGN FEN header conflicts with SetUp 0.", 0)};
        }

        eng::Position position = eng::Position::startpos();
        if (!fen.empty()) {
            const auto parsed_position = eng::Position::from_fen(fen);
            if (!parsed_position.has_value()) {
                return std::unexpected{
                    make_error(ErrorCode::InvalidFen, parsed_position.error().message, 0)};
            }
            position = *parsed_position;
        }
        initial_position_ = position;

        const auto header_result = header_value(headers_, "Result");
        if (!header_result.empty() && !is_result_token(header_result)) {
            return std::unexpected{make_error(ErrorCode::InvalidResult,
                                              "PGN Result header has an unsupported value.", 0)};
        }
        result_ = header_result.empty() ? "*" : std::string{header_result};

        bool saw_anything = !headers_.empty();
        bool saw_movetext_result = false;
        while (offset_ < text_.size()) {
            const auto skipped = skip_space_comments_and_variations();
            if (!skipped.has_value()) {
                return std::unexpected{skipped.error()};
            }
            if (offset_ >= text_.size()) {
                break;
            }
            if (peek() == '[') {
                return std::unexpected{make_error(ErrorCode::UnsupportedMultiGame,
                                                  "Only one PGN game is supported.", offset_)};
            }
            if (peek() == ')') {
                return std::unexpected{make_error(
                    ErrorCode::UnsupportedToken, "Unmatched PGN variation close marker.", offset_)};
            }

            const auto token_offset = offset_;
            auto token = read_token();
            if (token.empty()) {
                return std::unexpected{make_error(ErrorCode::UnsupportedToken,
                                                  "Unsupported PGN token.", token_offset)};
            }
            saw_anything = true;

            if (token.starts_with('$')) {
                if (token.size() == 1U || !std::ranges::all_of(token.substr(1), is_ascii_digit)) {
                    return std::unexpected{make_error(ErrorCode::UnsupportedToken,
                                                      "Malformed PGN NAG token.", token_offset)};
                }
                continue;
            }

            token = strip_move_number_prefix(std::move(token));
            if (token.empty() || is_move_number_token(token)) {
                continue;
            }

            if (is_result_token(token)) {
                if (!header_result.empty() && header_result != token) {
                    return std::unexpected{make_error(
                        ErrorCode::InvalidResult,
                        "PGN movetext result does not match Result header.", token_offset)};
                }
                result_ = token;
                saw_movetext_result = true;
                continue;
            }

            if (saw_movetext_result) {
                return std::unexpected{make_error(ErrorCode::UnsupportedMultiGame,
                                                  "PGN has movetext after the result marker.",
                                                  token_offset)};
            }

            Error resolve_error;
            auto move = resolve_san(position, token, resolve_error);
            if (!move.has_value()) {
                resolve_error.offset = token_offset;
                resolve_error.message.append(" Token: ");
                resolve_error.message.append(token);
                return std::unexpected{resolve_error};
            }

            eng::StateInfo state;
            if (!position.make_move(*move, state)) {
                return std::unexpected{make_error(ErrorCode::InvalidMoveList,
                                                  "PGN legal move failed during application.",
                                                  token_offset)};
            }
            moves_.push_back(*move);
        }

        if (!saw_anything) {
            return std::unexpected{
                make_error(ErrorCode::EmptyInput, "PGN input does not contain a game.", 0)};
        }

        Game game;
        game.initial_position = initial_position_;
        game.moves = std::move(moves_);
        game.result = result_;
        game.headers = merge_with_defaults(headers_, result_);
        if (game.initial_position.to_fen() != eng::Position::startpos().to_fen()) {
            upsert_header(game.headers, "SetUp", "1");
            upsert_header(game.headers, "FEN", game.initial_position.to_fen());
        }
        game.warnings = std::move(warnings_);
        return game;
    }

  private:
    [[nodiscard]] char peek() const noexcept {
        if (offset_ >= text_.size()) {
            return '\0';
        }
        return text_[offset_];
    }

    [[nodiscard]] std::expected<void, Error> skip_space_and_comments() {
        while (offset_ < text_.size()) {
            if (is_ascii_space(peek())) {
                ++offset_;
            } else if (peek() == '{') {
                auto skipped = skip_brace_comment();
                if (!skipped.has_value()) {
                    return skipped;
                }
            } else if (peek() == ';') {
                skip_semicolon_comment();
            } else {
                break;
            }
        }
        return {};
    }

    [[nodiscard]] std::expected<void, Error> skip_space_comments_and_variations() {
        while (offset_ < text_.size()) {
            if (is_ascii_space(peek())) {
                ++offset_;
            } else if (peek() == '{') {
                auto skipped = skip_brace_comment();
                if (!skipped.has_value()) {
                    return skipped;
                }
            } else if (peek() == ';') {
                skip_semicolon_comment();
            } else if (peek() == '(') {
                auto skipped = skip_variation();
                if (!skipped.has_value()) {
                    return skipped;
                }
            } else {
                break;
            }
        }
        return {};
    }

    // NOLINTNEXTLINE(readability-function-cognitive-complexity)
    [[nodiscard]] std::expected<Header, Error> parse_tag() {
        const auto tag_offset = offset_;
        ++offset_;
        while (offset_ < text_.size() && (text_[offset_] == ' ' || text_[offset_] == '\t')) {
            ++offset_;
        }

        const auto name_begin = offset_;
        while (offset_ < text_.size() && is_tag_name_char(text_[offset_])) {
            ++offset_;
        }
        if (offset_ == name_begin) {
            return std::unexpected{
                make_error(ErrorCode::MalformedTag, "PGN tag name is missing.", tag_offset)};
        }
        std::string name{text_.substr(name_begin, offset_ - name_begin)};
        while (offset_ < text_.size() && (text_[offset_] == ' ' || text_[offset_] == '\t')) {
            ++offset_;
        }
        if (offset_ >= text_.size() || text_[offset_] != '"') {
            return std::unexpected{
                make_error(ErrorCode::MalformedTag, "PGN tag value must be quoted.", offset_)};
        }
        ++offset_;

        std::string value;
        while (offset_ < text_.size()) {
            const auto symbol = text_[offset_++];
            if (symbol == '"') {
                while (offset_ < text_.size() &&
                       (text_[offset_] == ' ' || text_[offset_] == '\t')) {
                    ++offset_;
                }
                if (offset_ >= text_.size() || text_[offset_] != ']') {
                    return std::unexpected{make_error(
                        ErrorCode::MalformedTag, "PGN tag is missing closing bracket.", offset_)};
                }
                ++offset_;
                return Header{.name = std::move(name), .value = std::move(value)};
            }
            if (symbol == '\\') {
                if (offset_ >= text_.size()) {
                    return std::unexpected{make_error(ErrorCode::MalformedTag,
                                                      "PGN tag escape is incomplete.", offset_)};
                }
                value.push_back(text_[offset_++]);
            } else {
                value.push_back(symbol);
            }
        }
        return std::unexpected{
            make_error(ErrorCode::MalformedTag, "PGN tag value is unterminated.", tag_offset)};
    }

    [[nodiscard]] std::expected<void, Error> skip_brace_comment() {
        const auto comment_offset = offset_;
        ++offset_;
        while (offset_ < text_.size() && text_[offset_] != '}') {
            ++offset_;
        }
        if (offset_ >= text_.size()) {
            return std::unexpected{make_error(ErrorCode::UnterminatedComment,
                                              "PGN brace comment is unterminated.",
                                              comment_offset)};
        }
        ++offset_;
        return {};
    }

    void skip_semicolon_comment() noexcept {
        while (offset_ < text_.size() && text_[offset_] != '\n') {
            ++offset_;
        }
    }

    [[nodiscard]] std::expected<void, Error> skip_variation() {
        const auto variation_offset = offset_;
        int depth = 0;
        warnings_.push_back(Warning{.message = "Skipped PGN variation."});
        while (offset_ < text_.size()) {
            const auto symbol = text_[offset_];
            if (symbol == '{') {
                auto skipped = skip_brace_comment();
                if (!skipped.has_value()) {
                    return skipped;
                }
                continue;
            }
            if (symbol == ';') {
                skip_semicolon_comment();
                continue;
            }
            if (symbol == '(') {
                ++depth;
            } else if (symbol == ')') {
                --depth;
                if (depth == 0) {
                    ++offset_;
                    return {};
                }
            }
            ++offset_;
        }
        return std::unexpected{make_error(ErrorCode::UnterminatedVariation,
                                          "PGN variation is unterminated.", variation_offset)};
    }

    [[nodiscard]] std::string read_token() {
        const auto begin = offset_;
        while (offset_ < text_.size() && !is_symbol_boundary(text_[offset_])) {
            ++offset_;
        }
        return std::string{text_.substr(begin, offset_ - begin)};
    }

    std::string_view text_;
    std::size_t offset_{0};
    std::vector<Header> headers_;
    eng::Position initial_position_{eng::Position::startpos()};
    std::vector<eng::Move> moves_;
    std::string result_{"*"};
    std::vector<Warning> warnings_;
};

} // namespace

std::vector<Header> default_headers(std::string_view result) {
    std::vector<Header> headers;
    headers.reserve(kSevenTagRoster.size());
    headers.push_back(Header{.name = "Event", .value = "LocalChessEngine Game"});
    headers.push_back(Header{.name = "Site", .value = "?"});
    headers.push_back(Header{.name = "Date", .value = "????.??.??"});
    headers.push_back(Header{.name = "Round", .value = "-"});
    headers.push_back(Header{.name = "White", .value = "White"});
    headers.push_back(Header{.name = "Black", .value = "Black"});
    headers.push_back(Header{.name = "Result", .value = std::string{result}});
    return headers;
}

std::string_view header_value(const std::vector<Header>& headers, std::string_view name) noexcept {
    for (std::size_t index = headers.size(); index > 0U; --index) {
        const auto& header = headers[index - 1U];
        if (header.name == name) {
            return header.value;
        }
    }
    return {};
}

ParseResult parse_game(std::string_view text) { return Parser{text}.parse(); }

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TextResult san_for_move(const eng::Position& position, eng::Move move) {
    const auto moving_piece = position.piece_on(move.from());
    if (!move.is_valid() || move.is_null() || moving_piece.is_none()) {
        return std::unexpected{
            make_error(ErrorCode::InvalidMoveList, "Cannot format an invalid move.", 0)};
    }

    std::string san;
    if (move.is_castling()) {
        san = move.to().file().index() > move.from().file().index() ? "O-O" : "O-O-O";
    } else {
        const auto moving_type = moving_piece.type();
        const bool capture = is_capture(position, move);
        if (moving_type != eng::PieceType::Pawn) {
            san.push_back(piece_letter(moving_type));

            const auto legal_moves = eng::generate_legal_moves(position);
            std::vector<eng::Move> ambiguous_moves;
            for (const auto other : legal_moves) {
                if (other == move || other.to() != move.to()) {
                    continue;
                }
                const auto other_piece = position.piece_on(other.from());
                if (other_piece.color() == moving_piece.color() &&
                    other_piece.type() == moving_type) {
                    ambiguous_moves.push_back(other);
                }
            }
            if (!ambiguous_moves.empty()) {
                const bool same_file =
                    std::ranges::any_of(ambiguous_moves, [move](eng::Move other) {
                        return other.from().file() == move.from().file();
                    });
                const bool same_rank =
                    std::ranges::any_of(ambiguous_moves, [move](eng::Move other) {
                        return other.from().rank() == move.from().rank();
                    });
                if (!same_file) {
                    san.push_back(file_char(move.from()));
                } else if (!same_rank) {
                    san.push_back(rank_char(move.from()));
                } else {
                    san.push_back(file_char(move.from()));
                    san.push_back(rank_char(move.from()));
                }
            }
        } else if (capture) {
            san.push_back(file_char(move.from()));
        }

        if (capture) {
            san.push_back('x');
        }
        san.append(square_name(move.to()));
        if (move.is_promotion()) {
            san.push_back('=');
            san.push_back(piece_letter(move.promotion_piece()));
        }
    }

    auto next = position;
    eng::StateInfo state;
    if (!next.make_move(move, state)) {
        return std::unexpected{
            make_error(ErrorCode::InvalidMoveList, "Cannot format an illegal move.", 0)};
    }
    if (eng::is_checkmate(next)) {
        san.push_back('#');
    } else if (eng::side_to_move_in_check(next)) {
        san.push_back('+');
    }
    return san;
}

TextResult write_game(const Game& game) {
    if (!is_result_token(game.result)) {
        return std::unexpected{
            make_error(ErrorCode::InvalidResult, "Cannot export unsupported PGN result.", 0)};
    }

    const bool custom_start = game.initial_position.to_fen() != eng::Position::startpos().to_fen();
    auto headers = merge_with_defaults(game.headers, game.result);
    if (custom_start) {
        upsert_header(headers, "SetUp", "1");
        upsert_header(headers, "FEN", game.initial_position.to_fen());
    }

    std::string output;
    for (const auto name : kSevenTagRoster) {
        output.push_back('[');
        output.append(name);
        output.append(" \"");
        output.append(escape_tag_value(header_value(headers, name)));
        output.append("\"]\n");
    }
    if (custom_start) {
        output.append("[SetUp \"1\"]\n");
        output.append("[FEN \"");
        output.append(escape_tag_value(game.initial_position.to_fen()));
        output.append("\"]\n");
    }
    for (const auto& header : headers) {
        if (is_roster_tag(header.name) || header.name == "SetUp" || header.name == "FEN") {
            continue;
        }
        output.push_back('[');
        output.append(header.name);
        output.append(" \"");
        output.append(escape_tag_value(header.value));
        output.append("\"]\n");
    }
    output.push_back('\n');

    eng::Position position = game.initial_position;
    int column = 0;
    bool first_move = true;
    for (const auto move : game.moves) {
        if (position.side_to_move() == eng::Color::White) {
            append_wrapped_token(output, column, std::to_string(position.fullmove_number()) + ".");
        } else if (first_move) {
            append_wrapped_token(output, column,
                                 std::to_string(position.fullmove_number()) + "...");
        }

        const auto san = san_for_move(position, move);
        if (!san.has_value()) {
            return std::unexpected{san.error()};
        }
        append_wrapped_token(output, column, *san);

        eng::StateInfo state;
        if (!position.make_move(move, state)) {
            return std::unexpected{make_error(ErrorCode::InvalidMoveList,
                                              "Cannot export an invalid PGN move list.", 0)};
        }
        first_move = false;
    }
    append_wrapped_token(output, column, game.result);
    output.push_back('\n');
    return output;
}

} // namespace gui::pgn
