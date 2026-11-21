#include <eng/movegen.hpp>
#include <eng/movepick.hpp>

#include <doctest/doctest.h>

#include <array>
#include <span>
#include <string_view>

// Square names such as e2 and f3 are clearer than expanded identifiers in chess tests.
// NOLINTBEGIN(readability-identifier-length)

namespace {

constexpr int file_b = 1;
constexpr int file_c = 2;
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
constexpr int history_saturation_updates = 64;
constexpr int continuation_test_bonus = 1000;

[[nodiscard]] constexpr eng::Square square(int file_index, int rank_index) {
    return eng::Square::from_file_rank(eng::File::from_index(file_index),
                                       eng::Rank::from_index(rank_index));
}

[[nodiscard]] eng::Position parse(std::string_view fen) {
    const auto position = eng::Position::from_fen(fen);
    REQUIRE(position.has_value());
    return *position;
}

} // namespace

// doctest assertion macros intentionally inflate this test's measured complexity.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("move picker returns tt move first without duplicating it") {
    const auto position = eng::Position::startpos();
    const auto moves = eng::generate_moves(position);
    const auto tt_move = eng::Move::normal(square(file_g, rank_1), square(file_f, rank_3));

    eng::MovePicker picker(position, moves, eng::MovePickerContext{.tt_move = tt_move});

    int tt_occurrences = 0;
    for (auto move = picker.next(); move.is_valid(); move = picker.next()) {
        if (move == tt_move) {
            ++tt_occurrences;
            CHECK(tt_occurrences == 1);
        }
        if (tt_occurrences == 0) {
            CHECK(move == tt_move);
        }
    }

    CHECK(tt_occurrences == 1);
}

TEST_CASE("move picker orders tactical moves before quiet moves") {
    const auto position = parse("4k3/8/8/8/4q3/8/4R3/4K3 w - - 0 1");
    const auto queen_capture = eng::Move::normal(square(file_e, rank_2), square(file_e, rank_4));

    eng::MovePicker picker(position, eng::generate_moves(position), {});

    CHECK(picker.next() == queen_capture);
}

TEST_CASE("move picker can use see before mvv-lva for capture ordering") {
    const auto position = parse("4r1k1/8/8/6b1/4r3/5N2/4Q3/6K1 w - - 0 1");
    const auto losing_queen_capture =
        eng::Move::normal(square(file_e, rank_2), square(file_e, rank_4));
    const auto winning_knight_capture =
        eng::Move::normal(square(file_f, rank_3), square(file_g, rank_5));
    const std::array moves{losing_queen_capture, winning_knight_capture};

    eng::MovePicker picker(position, std::span<const eng::Move>{moves.data(), moves.size()},
                           eng::MovePickerContext{.score_captures_with_see = true});

    CHECK(picker.next() == winning_knight_capture);
    CHECK(picker.next() == losing_queen_capture);
}

TEST_CASE("move picker scores equal see captures ahead of losing see captures") {
    const auto position = parse("8/8/8/7Q/4r1b1/5k2/4R3/6K1 w - - 0 1");
    const auto equal_rook_trade = eng::Move::normal(square(file_e, rank_2), square(file_e, rank_4));
    const auto losing_queen_capture =
        eng::Move::normal(square(file_h, rank_5), square(file_g, rank_4));
    const std::array moves{losing_queen_capture, equal_rook_trade};

    eng::MovePicker picker(position, std::span<const eng::Move>{moves.data(), moves.size()},
                           eng::MovePickerContext{.score_captures_with_see = true});

    CHECK(picker.next() == equal_rook_trade);
    CHECK(picker.next() == losing_queen_capture);
}

TEST_CASE("move picker keeps tt move first when see capture scoring is enabled") {
    const auto position = parse("4r1k1/8/8/6b1/4r3/5N2/4Q3/6K1 w - - 0 1");
    const auto tt_capture = eng::Move::normal(square(file_e, rank_2), square(file_e, rank_4));
    const auto winning_capture = eng::Move::normal(square(file_f, rank_3), square(file_g, rank_5));
    const std::array moves{winning_capture, tt_capture};

    eng::MovePicker picker(
        position, std::span<const eng::Move>{moves.data(), moves.size()},
        eng::MovePickerContext{.tt_move = tt_capture, .score_captures_with_see = true});

    CHECK(picker.next() == tt_capture);
    CHECK(picker.next() == winning_capture);
}

TEST_CASE("killer moves outrank quiet history moves") {
    const auto position = eng::Position::startpos();
    const auto killer = eng::Move::normal(square(file_b, rank_1), square(file_c, rank_3));
    const auto history_move = eng::Move::normal(square(file_g, rank_1), square(file_f, rank_3));
    eng::MoveOrderingState ordering;

    ordering.update_quiet_move(position, history_move, {}, eng::move_ordering_history_limit);
    ordering.add_killer(eng::Ply::from_root(0), killer);

    eng::MovePicker picker(position, eng::generate_moves(position),
                           eng::MovePickerContext{.ordering = &ordering});

    CHECK(picker.next() == killer);
}

TEST_CASE("history gravity update saturates and applies malus") {
    const auto knight = eng::Piece::make(eng::Color::White, eng::PieceType::Knight);
    const auto target = square(file_f, rank_3);
    eng::MoveOrderingState ordering;

    for (int count = 0; count < history_saturation_updates; ++count) {
        ordering.update_quiet(knight, target, {}, eng::move_ordering_history_limit);
    }

    const auto saturated = ordering.history_score(knight, target);
    CHECK(saturated <= eng::move_ordering_history_limit);
    CHECK(saturated > eng::move_ordering_history_limit - 32);

    ordering.update_quiet(knight, target, {}, -eng::move_ordering_history_limit);

    CHECK(ordering.history_score(knight, target) < saturated);
}

TEST_CASE("continuation history is keyed by previous move") {
    const auto previous =
        eng::MoveContinuationKey{.piece = eng::Piece::make(eng::Color::White, eng::PieceType::Pawn),
                                 .to = square(file_e, rank_4)};
    const auto other_previous =
        eng::MoveContinuationKey{.piece = eng::Piece::make(eng::Color::White, eng::PieceType::Pawn),
                                 .to = square(file_d, rank_4)};
    const auto knight = eng::Piece::make(eng::Color::White, eng::PieceType::Knight);
    const auto target = square(file_f, rank_3);
    eng::MoveOrderingState ordering;

    ordering.update_quiet(knight, target, previous, continuation_test_bonus);

    CHECK(ordering.continuation_history_score(previous, knight, target) > 0);
    CHECK(ordering.continuation_history_score(other_previous, knight, target) == 0);
    CHECK(ordering.quiet_score(knight, target, previous) >
          ordering.quiet_score(knight, target, other_previous));
}

// NOLINTEND(readability-identifier-length)
