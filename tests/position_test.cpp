#include <eng/bitboard.hpp>
#include <eng/position.hpp>

#include <doctest/doctest.h>

#include <array>
#include <string_view>

// Square names such as e1 and h8 are clearer than expanded identifiers in chess tests.
// NOLINTBEGIN(readability-identifier-length)

namespace {

constexpr int file_a = 0;
constexpr int file_e = 4;
constexpr int file_h = 7;
constexpr int rank_1 = 0;
constexpr int rank_4 = 3;
constexpr int rank_8 = 7;

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] constexpr eng::Square square(int file_index, int rank_index) {
    return eng::Square::from_file_rank(eng::File::from_index(file_index),
                                       eng::Rank::from_index(rank_index));
}

struct InvalidFenCase {
    std::string_view fen;
    eng::FenErrorCode code;
};

} // namespace

TEST_CASE("start position FEN parses round-trips and builds board invariants") {
    const auto parsed = eng::Position::from_fen(eng::Position::startpos_fen);
    REQUIRE(parsed.has_value());

    const auto& position = *parsed;
    CHECK(position.to_fen() == eng::Position::startpos_fen);
    CHECK(eng::Position::startpos().to_fen() == eng::Position::startpos_fen);
    CHECK(position.has_consistent_occupancies());
    CHECK(position.zobrist_key() == position.recompute_zobrist_key());

    CHECK(position.side_to_move() == eng::Color::White);
    CHECK(position.castling_rights() == eng::CastlingRights::all());
    CHECK(!position.en_passant_square().is_valid());
    CHECK(position.halfmove_clock() == 0);
    CHECK(position.fullmove_number() == 1);

    CHECK(eng::popcount(position.occupancy(eng::Color::White)) == 16);
    CHECK(eng::popcount(position.occupancy(eng::Color::Black)) == 16);
    CHECK(eng::popcount(position.occupancy()) == 32);
    CHECK(eng::popcount(position.pieces(eng::Color::White, eng::PieceType::Pawn)) == 8);
    CHECK(eng::popcount(position.pieces(eng::Color::Black, eng::PieceType::Pawn)) == 8);

    CHECK(position.piece_on(square(file_e, rank_1)) ==
          eng::Piece::make(eng::Color::White, eng::PieceType::King));
    CHECK(position.piece_on(square(file_e, rank_8)) ==
          eng::Piece::make(eng::Color::Black, eng::PieceType::King));
    CHECK(position.piece_on(square(file_a, rank_1)) ==
          eng::Piece::make(eng::Color::White, eng::PieceType::Rook));
    CHECK(position.piece_on(square(file_h, rank_8)) ==
          eng::Piece::make(eng::Color::Black, eng::PieceType::Rook));
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("FEN parser accepts standard CPW sample positions and canonicalizes output") {
    constexpr std::array<std::string_view, 3> samples{{
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    }};

    for (const auto fen : samples) {
        const auto parsed = eng::Position::from_fen(fen);
        REQUIRE(parsed.has_value());
        CHECK(parsed->to_fen() == fen);
        CHECK(parsed->has_consistent_occupancies());
        CHECK(parsed->zobrist_key() == parsed->recompute_zobrist_key());
    }
}

TEST_CASE("FEN parser preserves counters en-passant and empty castling rights") {
    constexpr std::string_view fen = "4k3/8/8/8/4P3/8/8/4K3 b - e3 42 57";
    const auto parsed = eng::Position::from_fen(fen);
    REQUIRE(parsed.has_value());

    CHECK(parsed->to_fen() == fen);
    CHECK(parsed->side_to_move() == eng::Color::Black);
    CHECK(parsed->castling_rights().empty());
    CHECK(parsed->en_passant_square() == square(file_e, rank_1 + 2));
    CHECK(parsed->halfmove_clock() == 42);
    CHECK(parsed->fullmove_number() == 57);
    CHECK(parsed->piece_on(square(file_e, rank_4)) ==
          eng::Piece::make(eng::Color::White, eng::PieceType::Pawn));
}

TEST_CASE("Zobrist keys are deterministic and cover board side castling and en-passant") {
    const auto start_a = eng::Position::from_fen(eng::Position::startpos_fen);
    const auto start_b = eng::Position::from_fen(eng::Position::startpos_fen);
    REQUIRE(start_a.has_value());
    REQUIRE(start_b.has_value());
    CHECK(start_a->zobrist_key() == start_b->zobrist_key());
    CHECK(start_a->zobrist_key() == start_a->recompute_zobrist_key());

    const auto empty_white = eng::Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    const auto empty_black = eng::Position::from_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 1");
    const auto white_knight = eng::Position::from_fen("4k3/8/8/8/8/8/8/4KN2 w - - 0 1");
    const auto no_castling =
        eng::Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");
    const auto no_ep = eng::Position::from_fen("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1");
    const auto with_ep = eng::Position::from_fen("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1");
    const auto changed_counters = eng::Position::from_fen("4k3/8/8/8/4P3/8/8/4K3 b - - 7 12");
    REQUIRE(empty_white.has_value());
    REQUIRE(empty_black.has_value());
    REQUIRE(white_knight.has_value());
    REQUIRE(no_castling.has_value());
    REQUIRE(no_ep.has_value());
    REQUIRE(with_ep.has_value());
    REQUIRE(changed_counters.has_value());

    CHECK(empty_white->zobrist_key() != empty_black->zobrist_key());
    CHECK(empty_white->zobrist_key() != white_knight->zobrist_key());
    CHECK(start_a->zobrist_key() != no_castling->zobrist_key());
    CHECK(no_ep->zobrist_key() != with_ep->zobrist_key());
    CHECK(no_ep->zobrist_key() == changed_counters->zobrist_key());
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("FEN parser reports structured errors for malformed input") {
    constexpr std::array<InvalidFenCase, 14> invalid_cases{{
        {.fen = "4k3/8/8/8/8/8/8/4K3 w - - 0", .code = eng::FenErrorCode::MissingField},
        {.fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra", .code = eng::FenErrorCode::TooManyFields},
        {.fen = "4k3/8/8/8/8/8/8/4X3 w - - 0 1", .code = eng::FenErrorCode::InvalidPiecePlacement},
        {.fen = "4k3/8/8/8/8/8/8/5K3 w - - 0 1", .code = eng::FenErrorCode::InvalidPiecePlacement},
        {.fen = "4k3/8/8/8/8/8/8/4K3 x - - 0 1", .code = eng::FenErrorCode::InvalidSideToMove},
        {.fen = "r3k2r/8/8/8/8/8/8/R3K2R w KK - 0 1",
         .code = eng::FenErrorCode::InvalidCastlingRights},
        {.fen = "r3k2r/8/8/8/8/8/8/R3K2R w QK - 0 1",
         .code = eng::FenErrorCode::InvalidCastlingRights},
        {.fen = "4k3/8/8/8/8/8/8/4K3 w K - 0 1", .code = eng::FenErrorCode::InvalidCastlingRights},
        {.fen = "4k3/8/8/8/4P3/8/8/4K3 b - e4 0 1",
         .code = eng::FenErrorCode::InvalidEnPassantSquare},
        {.fen = "4k3/8/8/8/8/8/8/4K3 b - e3 0 1",
         .code = eng::FenErrorCode::InvalidEnPassantSquare},
        {.fen = "8/8/8/8/8/8/8/8 w - - 0 1", .code = eng::FenErrorCode::InvalidBoardState},
        {.fen = "4k3/8/8/8/8/8/8/4K2K w - - 0 1", .code = eng::FenErrorCode::InvalidBoardState},
        {.fen = "4k2P/8/8/8/8/8/8/4K3 w - - 0 1", .code = eng::FenErrorCode::InvalidBoardState},
        {.fen = "4k3/8/8/8/8/8/8/4K3 w - - -1 0", .code = eng::FenErrorCode::InvalidHalfmoveClock},
    }};

    for (const auto& invalid_case : invalid_cases) {
        const auto parsed = eng::Position::from_fen(invalid_case.fen);
        REQUIRE(!parsed.has_value());
        CHECK(parsed.error().code == invalid_case.code);
    }

    const auto invalid_fullmove = eng::Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 0");
    REQUIRE(!invalid_fullmove.has_value());
    CHECK(invalid_fullmove.error().code == eng::FenErrorCode::InvalidFullmoveNumber);
}

// NOLINTEND(readability-identifier-length)
