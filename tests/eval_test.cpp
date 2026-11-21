#include <eng/eval.hpp>
#include <eng/position.hpp>

#include <doctest/doctest.h>

#include <string_view>

namespace {

[[nodiscard]] eng::Position parse(std::string_view fen) {
    const auto position = eng::Position::from_fen(fen);
    REQUIRE(position.has_value());
    return *position;
}

[[nodiscard]] constexpr eng::EvaluationOptions no_eval_terms() noexcept {
    return {
        .material_and_psqt = false,
        .mobility = false,
        .pawn_structure = false,
        .king_safety = false,
        .bishop_pair = false,
        .rook_files = false,
    };
}

[[nodiscard]] constexpr eng::EvaluationOptions only_mobility() noexcept {
    auto options = no_eval_terms();
    options.mobility = true;
    return options;
}

[[nodiscard]] constexpr eng::EvaluationOptions only_pawn_structure() noexcept {
    auto options = no_eval_terms();
    options.pawn_structure = true;
    return options;
}

[[nodiscard]] constexpr eng::EvaluationOptions only_king_safety() noexcept {
    auto options = no_eval_terms();
    options.king_safety = true;
    return options;
}

[[nodiscard]] constexpr eng::EvaluationOptions only_bishop_pair() noexcept {
    auto options = no_eval_terms();
    options.bishop_pair = true;
    return options;
}

[[nodiscard]] constexpr eng::EvaluationOptions only_rook_files() noexcept {
    auto options = no_eval_terms();
    options.rook_files = true;
    return options;
}

} // namespace

TEST_CASE("evaluation is neutral for symmetric material") {
    const auto start = eng::Position::startpos();
    CHECK(eng::evaluate(start).value() == 0);
    CHECK(eng::evaluate(start).value() == eng::evaluate(start).value());

    CHECK(eng::evaluate(parse("4k3/8/8/8/8/8/8/4K3 w - - 0 1")).value() == 0);
    CHECK(eng::evaluate(parse("4k3/8/8/8/8/8/8/4K3 b - - 0 1")).value() == 0);
}

TEST_CASE("evaluation is relative to side to move") {
    const auto white_to_move = parse("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1");
    const auto black_to_move = parse("4k3/8/8/8/8/8/8/4KQ2 b - - 0 1");

    CHECK(eng::evaluate(white_to_move).value() > 0);
    CHECK(eng::evaluate(black_to_move).value() < 0);
    CHECK(eng::evaluate(white_to_move).value() == -eng::evaluate(black_to_move).value());
}

TEST_CASE("evaluation is symmetric after color and board mirroring") {
    const auto white_knight = parse("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1");
    const auto black_knight = parse("4k3/8/8/3n4/8/8/8/4K3 b - - 0 1");

    CHECK(eng::evaluate(white_knight).value() == eng::evaluate(black_knight).value());
}

TEST_CASE("evaluation piece-square tables distinguish placement with equal material") {
    const auto central_knight = parse("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1");
    const auto rim_knight = parse("4k3/8/8/8/8/8/8/N3K3 w - - 0 1");

    CHECK(eng::evaluate(central_knight).value() > eng::evaluate(rim_knight).value());
}

TEST_CASE("evaluation tapered score changes as game phase changes") {
    const auto endgame_pawn = parse("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1");
    const auto middlegame_pawn = parse("3qk3/8/8/8/4P3/8/8/3QK3 w - - 0 1");

    CHECK(eng::evaluate(endgame_pawn).value() > 0);
    CHECK(eng::evaluate(middlegame_pawn).value() > 0);
    CHECK(eng::evaluate(endgame_pawn).value() != eng::evaluate(middlegame_pawn).value());
}

TEST_CASE("evaluation terms can be disabled for ablation") {
    const auto position = parse("4k3/8/4P3/8/2BB4/8/5PPP/R5K1 w - - 0 1");

    CHECK(eng::evaluate(position, no_eval_terms()).value() == 0);
    CHECK(eng::evaluate(position).value() != eng::evaluate(position, no_eval_terms()).value());
}

TEST_CASE("evaluation default parameters preserve default score") {
    const auto parameters = eng::default_evaluation_parameters();
    for (const auto fen : {
             eng::Position::startpos_fen,
             std::string_view{"4k3/8/4P3/8/2BB4/8/5PPP/R5K1 w - - 0 1"},
             std::string_view{"q3k3/8/8/8/8/8/5PPP/Q5K1 b - - 0 1"},
         }) {
        const auto position = parse(fen);
        CHECK(eng::evaluate(position).value() ==
              eng::evaluate(position, eng::EvaluationOptions{}, parameters).value());
    }
}

TEST_CASE("evaluation parameter checksum is deterministic") {
    auto parameters = eng::default_evaluation_parameters();
    const auto checksum = eng::evaluation_parameters_checksum(parameters);

    CHECK(checksum == eng::evaluation_parameters_checksum(eng::default_evaluation_parameters()));
    CHECK(eng::set_evaluation_parameter(parameters, "bishop_pair.mg.bonus", 42));
    CHECK(checksum != eng::evaluation_parameters_checksum(parameters));
}

TEST_CASE("evaluation parameter changes are term scoped") {
    const auto bishop_pair = parse("4k3/8/8/8/2BB4/8/8/4K3 w - - 0 1");
    const auto open_bishop = parse("4k3/8/8/8/3B4/8/8/4K3 w - - 0 1");
    auto parameters = eng::default_evaluation_parameters();
    CHECK(eng::set_evaluation_parameter(parameters, "bishop_pair.mg.bonus", 90));
    CHECK(eng::set_evaluation_parameter(parameters, "bishop_pair.eg.bonus", 90));

    CHECK(eng::evaluate(bishop_pair, only_bishop_pair(), parameters).value() >
          eng::evaluate(bishop_pair, only_bishop_pair()).value());
    CHECK(eng::evaluate(open_bishop, only_mobility(), parameters).value() ==
          eng::evaluate(open_bishop, only_mobility()).value());
}

TEST_CASE("evaluation mobility rewards available pseudo-attacks") {
    const auto open_bishop = parse("4k3/8/8/8/3B4/8/8/4K3 w - - 0 1");
    const auto blocked_bishop = parse("4k3/8/8/2P1P3/3B4/2P1P3/8/4K3 w - - 0 1");

    CHECK(eng::evaluate(open_bishop, only_mobility()).value() >
          eng::evaluate(blocked_bishop, only_mobility()).value());
}

TEST_CASE("evaluation pawn structure rewards passed pawns and penalizes weak pawns") {
    const auto passed_pawn = parse("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1");
    const auto stopped_pawn = parse("4k3/3p4/4P3/8/8/8/8/4K3 w - - 0 1");
    const auto connected_pawns = parse("4k3/8/8/8/2PP4/8/8/4K3 w - - 0 1");
    const auto doubled_isolated_pawns = parse("4k3/8/8/8/2P5/2P5/8/4K3 w - - 0 1");

    CHECK(eng::evaluate(passed_pawn, only_pawn_structure()).value() >
          eng::evaluate(stopped_pawn, only_pawn_structure()).value());
    CHECK(eng::evaluate(connected_pawns, only_pawn_structure()).value() >
          eng::evaluate(doubled_isolated_pawns, only_pawn_structure()).value());
}

TEST_CASE("evaluation bishop pair is independently rewarded") {
    const auto one_bishop = parse("4k3/8/8/8/3B4/8/8/4K3 w - - 0 1");
    const auto bishop_pair = parse("4k3/8/8/8/2BB4/8/8/4K3 w - - 0 1");

    CHECK(eng::evaluate(bishop_pair, only_bishop_pair()).value() >
          eng::evaluate(one_bishop, only_bishop_pair()).value());
}

TEST_CASE("evaluation rooks prefer open and semi-open files") {
    const auto open_file = parse("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    const auto closed_file = parse("4k3/8/8/8/8/8/P7/R3K3 w - - 0 1");

    CHECK(eng::evaluate(open_file, only_rook_files()).value() >
          eng::evaluate(closed_file, only_rook_files()).value());
}

TEST_CASE("evaluation king safety rewards pawn shield over exposed king") {
    const auto shielded_king = parse("q3k3/8/8/8/8/8/5PPP/Q5K1 w - - 0 1");
    const auto exposed_king = parse("q3k3/8/8/8/8/8/8/Q5K1 w - - 0 1");

    CHECK(eng::evaluate(shielded_king, only_king_safety()).value() >
          eng::evaluate(exposed_king, only_king_safety()).value());
}
