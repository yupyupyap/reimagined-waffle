#ifndef GUI_PGN_HPP
#define GUI_PGN_HPP

#include "eng/move.hpp"
#include "eng/position.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace gui::pgn {

inline constexpr std::size_t max_pgn_bytes = std::size_t{2} * 1024U * 1024U;

enum class ErrorCode : std::uint8_t {
    EmptyInput,
    InputTooLarge,
    MalformedTag,
    UnterminatedComment,
    UnterminatedVariation,
    InvalidResult,
    InvalidFen,
    MissingFen,
    UnsupportedMultiGame,
    UnsupportedToken,
    IllegalMove,
    AmbiguousMove,
    InvalidMoveList
};

struct Error {
    ErrorCode code{ErrorCode::UnsupportedToken};
    std::string message;
    std::size_t offset{0};
};

struct Warning {
    std::string message;
};

struct Header {
    std::string name;
    std::string value;
};

struct Game {
    std::vector<Header> headers;
    eng::Position initial_position{eng::Position::startpos()};
    std::vector<eng::Move> moves;
    std::string result{"*"};
    std::vector<Warning> warnings;
};

using ParseResult = std::expected<Game, Error>;
using TextResult = std::expected<std::string, Error>;

[[nodiscard]] std::vector<Header> default_headers(std::string_view result = "*");
[[nodiscard]] std::string_view header_value(const std::vector<Header>& headers,
                                            std::string_view name) noexcept;

[[nodiscard]] ParseResult parse_game(std::string_view text);
[[nodiscard]] TextResult write_game(const Game& game);
[[nodiscard]] TextResult san_for_move(const eng::Position& position, eng::Move move);

} // namespace gui::pgn

#endif // GUI_PGN_HPP
