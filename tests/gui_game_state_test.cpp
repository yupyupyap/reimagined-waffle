#include "game_state.hpp"

#include <eng/uci.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <optional>
#include <string_view>

// Square names such as e2 and g8 are clearer than expanded identifiers in chess tests.
// NOLINTBEGIN(readability-identifier-length)

namespace {

constexpr int file_d = 3;
constexpr int file_e = 4;
constexpr int file_f = 5;
constexpr int file_g = 6;
constexpr int file_h = 7;
constexpr int rank_1 = 0;
constexpr int rank_2 = 1;
constexpr int rank_3 = 2;
constexpr int rank_4 = 3;
constexpr int rank_5 = 4;
constexpr int rank_6 = 5;
constexpr int rank_7 = 6;
constexpr int rank_8 = 7;

[[nodiscard]] constexpr eng::Square square(int file_index, int rank_index) {
    return eng::Square::from_file_rank(eng::File::from_index(file_index),
                                       eng::Rank::from_index(rank_index));
}

[[nodiscard]] eng::Position parse_position(std::string_view fen) {
    const auto position = eng::Position::from_fen(fen);
    REQUIRE(position.has_value());
    return *position;
}

[[nodiscard]] bool contains_square(const std::vector<eng::Square>& squares, eng::Square square) {
    return std::ranges::find(squares, square) != squares.end();
}

[[nodiscard]] GuiGameState::MoveAttempt apply_uci(GuiGameState& game, std::string_view notation) {
    const auto move = eng::move_from_uci(game.position(), notation);
    if (!move.has_value()) {
        FAIL_CHECK("expected parseable legal UCI move");
        return {};
    }
    const auto parsed_move = move.value();
    std::optional<eng::PieceType> promotion;
    if (parsed_move.is_promotion()) {
        promotion = parsed_move.promotion_piece();
    }
    return game.try_move(parsed_move.from(), parsed_move.to(), promotion);
}

void require_applied(GuiGameState& game, std::string_view notation) {
    const auto attempt = apply_uci(game, notation);
    REQUIRE(attempt.kind == GuiGameState::MoveAttemptKind::Applied);
}

} // namespace

TEST_CASE("GUI game state exposes legal hints from the starting position") {
    GuiGameState game;

    REQUIRE(game.select_square(square(file_e, rank_2)));
    CHECK(contains_square(game.legal_destinations(), square(file_e, rank_3)));
    CHECK(contains_square(game.legal_destinations(), square(file_e, rank_4)));

    REQUIRE(game.select_square(square(file_g, rank_1)));
    CHECK(contains_square(game.legal_destinations(), square(file_f, rank_3)));
    CHECK(contains_square(game.legal_destinations(), square(file_h, rank_3)));
    CHECK(!contains_square(game.legal_destinations(), square(file_e, rank_2)));
}

TEST_CASE("GUI game state rejects illegal moves without mutating position") {
    GuiGameState game;
    const auto before_fen = game.position().to_fen();
    const auto before_key = game.position().zobrist_key();

    const auto attempt = game.try_move(square(file_e, rank_2), square(file_e, rank_5));
    CHECK(attempt.kind == GuiGameState::MoveAttemptKind::Invalid);
    CHECK(game.position().to_fen() == before_fen);
    CHECK(game.position().zobrist_key() == before_key);
}

TEST_CASE("GUI game state applies castling and en passant through engine legality") {
    GuiGameState castling;
    castling.set_position(parse_position("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"));

    auto attempt = castling.try_move(square(file_e, rank_1), square(file_g, rank_1));
    REQUIRE(attempt.kind == GuiGameState::MoveAttemptKind::Applied);
    CHECK(castling.position().piece_on(square(file_g, rank_1)) ==
          eng::Piece::make(eng::Color::White, eng::PieceType::King));
    CHECK(castling.position().piece_on(square(file_f, rank_1)) ==
          eng::Piece::make(eng::Color::White, eng::PieceType::Rook));

    GuiGameState en_passant;
    en_passant.set_position(parse_position("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 3"));

    attempt = en_passant.try_move(square(file_e, rank_5), square(file_d, rank_6));
    REQUIRE(attempt.kind == GuiGameState::MoveAttemptKind::Applied);
    CHECK(en_passant.position().piece_on(square(file_d, rank_6)) ==
          eng::Piece::make(eng::Color::White, eng::PieceType::Pawn));
    CHECK(en_passant.position().piece_on(square(file_d, rank_5)).is_none());
}

TEST_CASE("GUI game state requires explicit promotion choice and applies queen promotion") {
    GuiGameState game;
    game.set_position(parse_position("4k3/6P1/8/8/8/8/8/4K3 w - - 4 9"));
    const auto before_fen = game.position().to_fen();

    auto attempt = game.try_move(square(file_g, rank_7), square(file_g, rank_8));
    REQUIRE(attempt.kind == GuiGameState::MoveAttemptKind::PromotionRequired);
    CHECK(attempt.promotion_count == 4);
    CHECK(attempt.promotion_pieces[0] == eng::PieceType::Queen);
    CHECK(game.position().to_fen() == before_fen);

    attempt = game.try_move(square(file_g, rank_7), square(file_g, rank_8), eng::PieceType::Queen);
    REQUIRE(attempt.kind == GuiGameState::MoveAttemptKind::Applied);
    CHECK(game.position().piece_on(square(file_g, rank_8)) ==
          eng::Piece::make(eng::Color::White, eng::PieceType::Queen));
}

TEST_CASE("GUI game state reports fool's mate checkmate with black winner") {
    GuiGameState game;
    require_applied(game, "f2f3");
    require_applied(game, "e7e5");
    require_applied(game, "g2g4");
    require_applied(game, "d8h4");

    CHECK(game.status().kind == GuiGameState::StatusKind::Checkmate);
    CHECK(game.status().terminal);
    CHECK(game.status().winner == eng::Color::Black);
    CHECK(game.check_square() == square(file_e, rank_1));
}

TEST_CASE("GUI game state reports draw and check statuses in rules order") {
    GuiGameState game;

    game.set_position(parse_position("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"));
    CHECK(game.status().kind == GuiGameState::StatusKind::Stalemate);

    game.set_position(parse_position("4k3/8/8/8/8/8/8/4K3 w - - 100 75"));
    CHECK(game.status().kind == GuiGameState::StatusKind::DrawFiftyMove);

    game.set_position(parse_position("4k3/8/8/8/8/8/8/3NK3 w - - 0 1"));
    CHECK(game.status().kind == GuiGameState::StatusKind::DrawInsufficientMaterial);
}

TEST_CASE("GUI game state detects threefold repetition from current game history") {
    GuiGameState game;
    require_applied(game, "g1f3");
    require_applied(game, "g8f6");
    require_applied(game, "f3g1");
    require_applied(game, "f6g8");
    CHECK(game.status().kind == GuiGameState::StatusKind::Playing);

    require_applied(game, "g1f3");
    require_applied(game, "g8f6");
    require_applied(game, "f3g1");
    require_applied(game, "f6g8");
    CHECK(game.status().kind == GuiGameState::StatusKind::DrawRepetition);
    CHECK(game.status().terminal);
}

// NOLINTEND(readability-identifier-length)
