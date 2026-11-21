#include <eng/rules.hpp>

#include <doctest/doctest.h>

#include <array>
#include <string_view>

namespace {

[[nodiscard]] eng::Position parse(std::string_view fen) {
    const auto position = eng::Position::from_fen(fen);
    REQUIRE(position.has_value());
    return *position;
}

} // namespace

TEST_CASE("fifty move rule draw uses halfmove clock threshold") {
    CHECK(eng::is_fifty_move_rule_draw(parse("4k3/8/8/8/8/8/8/4K3 w - - 100 75")));
    CHECK(!eng::is_fifty_move_rule_draw(parse("4k3/8/8/8/8/8/8/4K3 w - - 99 75")));
}

TEST_CASE("threefold repetition uses external history inside the halfmove window") {
    const auto current = parse("4k3/8/8/8/8/8/8/4K3 w - - 8 1");
    const auto other = parse("4k3/8/8/8/8/8/8/3K4 b - - 0 1");
    const std::array history{
        other.zobrist_key(),
        current.zobrist_key(),
        other.zobrist_key(),
        current.zobrist_key(),
    };

    CHECK(eng::is_threefold_repetition(current, history));

    const auto narrow_window = parse("4k3/8/8/8/8/8/8/4K3 w - - 1 1");
    CHECK(!eng::is_threefold_repetition(narrow_window, history));
}

TEST_CASE("insufficient material detects practical dead-material cases") {
    CHECK(eng::is_insufficient_material(parse("4k3/8/8/8/8/8/8/4K3 w - - 0 1")));
    CHECK(eng::is_insufficient_material(parse("4k3/8/8/8/8/8/8/2B1K3 w - - 0 1")));
    CHECK(eng::is_insufficient_material(parse("4k3/8/8/8/8/8/8/3NK3 w - - 0 1")));
    CHECK(eng::is_insufficient_material(parse("4k3/8/7B/8/8/8/8/2B1K3 w - - 0 1")));

    CHECK(!eng::is_insufficient_material(parse("4k3/8/8/8/8/8/5NN1/4K3 w - - 0 1")));
    CHECK(!eng::is_insufficient_material(parse("4k3/8/8/7b/8/8/8/2B1K3 w - - 0 1")));
    CHECK(!eng::is_insufficient_material(parse("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")));
}
