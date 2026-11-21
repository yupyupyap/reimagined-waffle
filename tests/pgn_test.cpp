#include "game_state.hpp"
#include "pgn.hpp"

#include <eng/uci.hpp>

#include <doctest/doctest.h>

#include <optional>
#include <string_view>

namespace {

constexpr int kFileE = 4;
constexpr int kRank2 = 1;
constexpr int kRank4 = 3;

[[nodiscard]] constexpr eng::Square square(int file_index, int rank_index) {
    return eng::Square::from_file_rank(eng::File::from_index(file_index),
                                       eng::Rank::from_index(rank_index));
}

[[nodiscard]] gui::pgn::Game require_parse(std::string_view text) {
    auto parsed = gui::pgn::parse_game(text);
    REQUIRE(parsed.has_value());
    return *parsed;
}

[[nodiscard]] eng::Move require_uci(const eng::Position& position, std::string_view notation) {
    auto move = eng::move_from_uci(position, notation);
    if (!move.has_value()) {
        FAIL("expected legal UCI move");
        return eng::Move::none();
    }
    return *move;
}

void require_apply(GuiGameState& game, std::string_view notation) {
    const auto move = require_uci(game.position(), notation);
    const auto promotion =
        move.is_promotion() ? std::optional<eng::PieceType>{move.promotion_piece()} : std::nullopt;
    const auto attempt = game.try_move(move.from(), move.to(), promotion);
    REQUIRE(attempt.kind == GuiGameState::MoveAttemptKind::Applied);
}

} // namespace

TEST_CASE("PGN parses simple SAN and writes a round-trippable game") {
    const auto game = require_parse("1. e4 e5 2. Nf3 Nc6 *\n");
    CHECK(game.moves.size() == 4U);
    CHECK(game.result == "*");
    CHECK(gui::pgn::header_value(game.headers, "Event") == "LocalChessEngine Game");

    const auto exported = gui::pgn::write_game(game);
    REQUIRE(exported.has_value());
    CHECK(exported->contains("[Result \"*\"]"));
    CHECK(exported->contains("1. e4 e5 2. Nf3 Nc6 *"));

    const auto reparsed = gui::pgn::parse_game(*exported);
    REQUIRE(reparsed.has_value());
    CHECK(reparsed->moves == game.moves);
    CHECK(reparsed->result == "*");
}

TEST_CASE("PGN supports captures, castling, disambiguation, promotion and mate SAN") {
    auto capture = require_parse("1. e4 d5 2. exd5 *");
    CHECK(capture.moves.size() == 3U);

    auto castling = require_parse("[SetUp \"1\"]\n"
                                  "[FEN \"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1\"]\n\n"
                                  "1. O-O O-O-O *");
    REQUIRE(castling.moves.size() == 2U);
    const auto castling_export = gui::pgn::write_game(castling);
    REQUIRE(castling_export.has_value());
    CHECK(castling_export->contains("1. O-O O-O-O *"));

    auto disambiguation = require_parse("[SetUp \"1\"]\n"
                                        "[FEN \"4k3/8/8/8/8/8/8/1N2KN2 w - - 0 1\"]\n\n"
                                        "1. Nbd2 *");
    CHECK(disambiguation.moves.size() == 1U);

    auto promotion = require_parse("[SetUp \"1\"]\n"
                                   "[FEN \"4k3/6P1/8/8/8/8/8/4K3 w - - 0 1\"]\n\n"
                                   "1. g8=Q+ *");
    CHECK(promotion.moves.size() == 1U);

    auto mate = require_parse("1. f3 e5 2. g4 Qh4# 0-1");
    CHECK(mate.result == "0-1");
    const auto mate_export = gui::pgn::write_game(mate);
    REQUIRE(mate_export.has_value());
    CHECK(mate_export->contains("Qh4# 0-1"));
}

TEST_CASE("PGN handles SetUp FEN headers and preserves them on export") {
    constexpr std::string_view pgn = "[Event \"Promotion Smoke\"]\n"
                                     "[SetUp \"1\"]\n"
                                     "[FEN \"4k3/6P1/8/8/8/8/8/4K3 w - - 4 9\"]\n\n"
                                     "9. g8=Q+ *";

    const auto game = require_parse(pgn);
    CHECK(game.initial_position.to_fen() == "4k3/6P1/8/8/8/8/8/4K3 w - - 4 9");

    const auto exported = gui::pgn::write_game(game);
    REQUIRE(exported.has_value());
    CHECK(exported->contains("[SetUp \"1\"]"));
    CHECK(exported->contains("[FEN \"4k3/6P1/8/8/8/8/8/4K3 w - - 4 9\"]"));
    CHECK(exported->contains("9. g8=Q+ *"));
}

TEST_CASE("PGN ignores comments NAGs and variations with warnings") {
    const auto game = require_parse("1. e4 {main move} (1. d4 d5) e5 $1 2. Nf3 ; line comment\n"
                                    "2... Nc6 *");
    CHECK(game.moves.size() == 4U);
    REQUIRE(!game.warnings.empty());
    CHECK(game.warnings.front().message == "Skipped PGN variation.");
}

TEST_CASE("PGN rejects malformed or unsupported input") {
    auto invalid_tag = gui::pgn::parse_game("[Event bad]\n1. e4 *");
    REQUIRE(!invalid_tag.has_value());
    CHECK(invalid_tag.error().code == gui::pgn::ErrorCode::MalformedTag);

    auto invalid_fen = gui::pgn::parse_game("[SetUp \"1\"]\n[FEN \"not a fen\"]\n\n*");
    REQUIRE(!invalid_fen.has_value());
    CHECK(invalid_fen.error().code == gui::pgn::ErrorCode::InvalidFen);

    auto illegal_move = gui::pgn::parse_game("1. e5 *");
    REQUIRE(!illegal_move.has_value());
    CHECK(illegal_move.error().code == gui::pgn::ErrorCode::IllegalMove);

    auto malformed_promotion = gui::pgn::parse_game("[SetUp \"1\"]\n"
                                                    "[FEN \"4k3/6P1/8/8/8/8/8/4K3 w - - 0 1\"]\n\n"
                                                    "1. g8Q *");
    REQUIRE(!malformed_promotion.has_value());
    CHECK(malformed_promotion.error().code == gui::pgn::ErrorCode::IllegalMove);

    auto multi_game = gui::pgn::parse_game("[Event \"A\"]\n\n1. e4 *\n\n[Event \"B\"]\n\n1. d4 *");
    REQUIRE(!multi_game.has_value());
    CHECK(multi_game.error().code == gui::pgn::ErrorCode::UnsupportedMultiGame);
}

TEST_CASE("GUI game state loads PGN mainline and navigates without mutating on invalid PGN") {
    GuiGameState state;
    const auto before = state.position().to_fen();
    const auto invalid = gui::pgn::parse_game("1. e5 *");
    REQUIRE(!invalid.has_value());
    CHECK(state.position().to_fen() == before);

    const auto game = require_parse("1. e4 e5 2. Nf3 *");
    REQUIRE(state.load_pgn_game(game));
    CHECK(state.move_count() == 3U);
    CHECK(state.current_ply() == 3U);

    REQUIRE(state.go_to_ply(1));
    CHECK(state.position().to_fen() ==
          "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    CHECK(!state.is_at_latest_ply());
    auto attempt = state.try_move(square(kFileE, kRank2), square(kFileE, kRank4));
    CHECK(attempt.kind == GuiGameState::MoveAttemptKind::Invalid);
    CHECK(state.move_count() == 3U);

    REQUIRE(state.go_to_ply(state.move_count()));
    require_apply(state, "b8c6");
    CHECK(state.move_count() == 4U);

    const auto exported = gui::pgn::write_game(state.to_pgn_game());
    REQUIRE(exported.has_value());
    CHECK(exported->contains("1. e4 e5 2. Nf3 Nc6 *"));
}
