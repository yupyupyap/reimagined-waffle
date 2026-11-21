#include <eng/bitboard.hpp>

#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <limits>

// Square names such as a1 and d4 are clearer than expanded identifiers in chess tests.
// NOLINTBEGIN(readability-identifier-length)

namespace {

constexpr int file_a = 0;
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
constexpr int rank_6 = 5;
constexpr int rank_7 = 6;
constexpr int rank_8 = 7;

[[nodiscard]] constexpr eng::Square square(int file_index, int rank_index) {
    return eng::Square::from_file_rank(eng::File::from_index(file_index),
                                       eng::Rank::from_index(rank_index));
}

template <typename... Squares> [[nodiscard]] constexpr eng::Bitboard squares(Squares... values) {
    return (eng::empty_bitboard | ... | eng::square_bb(values));
}

template <std::size_t DirectionCount>
[[nodiscard]] constexpr eng::Bitboard
slow_slider_attacks(eng::Square from_square, eng::Bitboard occupancy,
                    const std::array<eng::Direction, DirectionCount>& directions) {
    eng::Bitboard attacks = eng::empty_bitboard;
    for (const auto direction : directions) {
        auto current = eng::bitboard_detail::step(from_square, direction);
        while (current.is_valid()) {
            attacks |= eng::square_bb(current);
            if (eng::has_square(occupancy, current)) {
                break;
            }
            current = eng::bitboard_detail::step(current, direction);
        }
    }
    return attacks;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] constexpr eng::Bitboard subset_from_index(eng::Bitboard mask, std::size_t index) {
    eng::Bitboard subset = eng::empty_bitboard;
    auto remaining = mask;
    std::size_t bit_index = 0;
    while (remaining != eng::empty_bitboard) {
        const auto current = eng::pop_lsb(remaining);
        if ((index & (std::size_t{1} << bit_index)) != 0U) {
            subset |= eng::square_bb(current);
        }
        ++bit_index;
    }
    return subset;
}

} // namespace

TEST_CASE("bitboard helpers manipulate square masks") {
    auto bitboard = eng::empty_bitboard;
    const auto a1 = square(file_a, rank_1);
    const auto h8 = square(file_h, rank_8);

    CHECK(eng::popcount(bitboard) == 0);
    CHECK(!eng::lsb(bitboard).is_valid());

    eng::set_square(bitboard, h8);
    eng::set_square(bitboard, a1);
    CHECK(eng::popcount(bitboard) == 2);
    CHECK(eng::has_square(bitboard, a1));
    CHECK(eng::has_square(bitboard, h8));
    CHECK(eng::lsb(bitboard) == a1);
    CHECK(eng::pop_lsb(bitboard) == a1);
    CHECK(!eng::has_square(bitboard, a1));
    CHECK(eng::pop_lsb(bitboard) == h8);
    CHECK(bitboard == eng::empty_bitboard);

    eng::set_square(bitboard, h8);
    eng::clear_square(bitboard, h8);
    CHECK(bitboard == eng::empty_bitboard);
}

TEST_CASE("king knight and pawn attack tables match known squares") {
    constexpr auto a1 = square(file_a, rank_1);
    constexpr auto d4 = square(file_d, rank_4);
    constexpr auto e2 = square(file_e, rank_2);
    constexpr auto e7 = square(file_e, rank_7);
    constexpr auto h8 = square(file_h, rank_8);

    static_assert(eng::king_attacks(a1) ==
                  squares(square(file_b, rank_1), square(file_a, rank_2), square(file_b, rank_2)));
    CHECK(eng::king_attacks(h8) ==
          squares(square(file_g, rank_8), square(file_h, rank_7), square(file_g, rank_7)));
    CHECK(eng::king_attacks(d4) == squares(square(file_c, rank_3), square(file_d, rank_3),
                                           square(file_e, rank_3), square(file_c, rank_4),
                                           square(file_e, rank_4), square(file_c, rank_5),
                                           square(file_d, rank_5), square(file_e, rank_5)));

    static_assert(eng::knight_attacks(a1) ==
                  squares(square(file_b, rank_3), square(file_c, rank_2)));
    CHECK(eng::knight_attacks(d4) == squares(square(file_b, rank_3), square(file_b, rank_5),
                                             square(file_c, rank_2), square(file_c, rank_6),
                                             square(file_e, rank_2), square(file_e, rank_6),
                                             square(file_f, rank_3), square(file_f, rank_5)));

    CHECK(eng::pawn_attacks(eng::Color::White, e2) ==
          squares(square(file_d, rank_3), square(file_f, rank_3)));
    CHECK(eng::pawn_attacks(eng::Color::Black, e7) ==
          squares(square(file_d, rank_6), square(file_f, rank_6)));
    CHECK(eng::pawn_attacks(eng::Color::White, square(file_a, rank_2)) ==
          squares(square(file_b, rank_3)));
    CHECK(eng::pawn_attacks(eng::Color::Black, square(file_h, rank_7)) ==
          squares(square(file_g, rank_6)));
    CHECK(eng::pawn_attacks(eng::Color::None, e2) == eng::empty_bitboard);
}

TEST_CASE("ray between and line tables preserve a1 based geometry") {
    constexpr auto a1 = square(file_a, rank_1);
    constexpr auto h1 = square(file_h, rank_1);
    constexpr auto c1 = square(file_c, rank_1);
    constexpr auto f4 = square(file_f, rank_4);
    constexpr auto b3 = square(file_b, rank_3);
    constexpr auto d4 = square(file_d, rank_4);

    CHECK(eng::ray(a1, eng::Direction::North) ==
          squares(square(file_a, rank_2), square(file_a, rank_3), square(file_a, rank_4),
                  square(file_a, rank_5), square(file_a, rank_6), square(file_a, rank_7),
                  square(file_a, rank_8)));
    CHECK(eng::ray(a1, eng::Direction::East) ==
          squares(square(file_b, rank_1), square(file_c, rank_1), square(file_d, rank_1),
                  square(file_e, rank_1), square(file_f, rank_1), square(file_g, rank_1),
                  square(file_h, rank_1)));
    CHECK(eng::ray(a1, eng::Direction::NorthEast) ==
          squares(square(file_b, rank_2), square(file_c, rank_3), square(file_d, rank_4),
                  square(file_e, rank_5), square(file_f, rank_6), square(file_g, rank_7),
                  square(file_h, rank_8)));

    CHECK(eng::between(a1, h1) == squares(square(file_b, rank_1), square(file_c, rank_1),
                                          square(file_d, rank_1), square(file_e, rank_1),
                                          square(file_f, rank_1), square(file_g, rank_1)));
    CHECK(eng::between(c1, f4) == squares(square(file_d, rank_2), square(file_e, rank_3)));
    CHECK(eng::between(a1, b3) == eng::empty_bitboard);
    CHECK(eng::between(a1, a1) == eng::empty_bitboard);

    CHECK(eng::line(a1, h1) == squares(square(file_a, rank_1), square(file_b, rank_1),
                                       square(file_c, rank_1), square(file_d, rank_1),
                                       square(file_e, rank_1), square(file_f, rank_1),
                                       square(file_g, rank_1), square(file_h, rank_1)));
    CHECK(eng::line(c1, f4) == squares(square(file_c, rank_1), square(file_d, rank_2),
                                       square(file_e, rank_3), square(file_f, rank_4),
                                       square(file_g, rank_5), square(file_h, rank_6)));
    CHECK(eng::line(a1, b3) == eng::empty_bitboard);
    CHECK(eng::line(d4, d4) == eng::square_bb(d4));
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("magic metadata matches relevant occupancy masks") {
    eng::init_bitboards();

    for (int index = 0; index < eng::Square::count; ++index) {
        const auto current = eng::Square::from_index(index);
        const auto rook_entry = eng::rook_magic(current);
        const auto bishop_entry = eng::bishop_magic(current);

        CHECK(rook_entry.mask == eng::rook_relevant_occupancy(current));
        CHECK(bishop_entry.mask == eng::bishop_relevant_occupancy(current));
        CHECK(rook_entry.magic != eng::empty_bitboard);
        CHECK(bishop_entry.magic != eng::empty_bitboard);
        CHECK(rook_entry.shift ==
              std::numeric_limits<eng::Bitboard>::digits - eng::popcount(rook_entry.mask));
        CHECK(bishop_entry.shift ==
              std::numeric_limits<eng::Bitboard>::digits - eng::popcount(bishop_entry.mask));
    }
}

TEST_CASE("sliding attacks match slow references for representative blockers") {
    eng::init_bitboards();

    constexpr auto d4_square = square(file_d, rank_4);
    const auto occupancy =
        squares(square(file_d, rank_6), square(file_d, rank_2), square(file_b, rank_4),
                square(file_f, rank_4), square(file_b, rank_2), square(file_f, rank_6));

    CHECK(
        eng::rook_attacks(d4_square, eng::empty_bitboard) ==
        slow_slider_attacks(d4_square, eng::empty_bitboard, eng::bitboard_detail::rook_directions));
    CHECK(eng::bishop_attacks(d4_square, eng::empty_bitboard) ==
          slow_slider_attacks(d4_square, eng::empty_bitboard,
                              eng::bitboard_detail::bishop_directions));
    CHECK(eng::rook_attacks(d4_square, occupancy) ==
          slow_slider_attacks(d4_square, occupancy, eng::bitboard_detail::rook_directions));
    CHECK(eng::bishop_attacks(d4_square, occupancy) ==
          slow_slider_attacks(d4_square, occupancy, eng::bitboard_detail::bishop_directions));
    CHECK(eng::queen_attacks(d4_square, occupancy) ==
          (slow_slider_attacks(d4_square, occupancy, eng::bitboard_detail::rook_directions) |
           slow_slider_attacks(d4_square, occupancy, eng::bitboard_detail::bishop_directions)));
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("rook and bishop magics match slow attacks for all relevant occupancies") {
    eng::init_bitboards();

    for (int index = 0; index < eng::Square::count; ++index) {
        const auto current = eng::Square::from_index(index);

        const auto rook_mask = eng::rook_relevant_occupancy(current);
        const auto rook_occupancy_count = std::size_t{1} << eng::popcount(rook_mask);
        for (std::size_t subset = 0; subset < rook_occupancy_count; ++subset) {
            const auto occupancy = subset_from_index(rook_mask, subset);
            CHECK(eng::rook_attacks(current, occupancy) ==
                  slow_slider_attacks(current, occupancy, eng::bitboard_detail::rook_directions));
        }

        const auto bishop_mask = eng::bishop_relevant_occupancy(current);
        const auto bishop_occupancy_count = std::size_t{1} << eng::popcount(bishop_mask);
        for (std::size_t subset = 0; subset < bishop_occupancy_count; ++subset) {
            const auto occupancy = subset_from_index(bishop_mask, subset);
            CHECK(eng::bishop_attacks(current, occupancy) ==
                  slow_slider_attacks(current, occupancy, eng::bitboard_detail::bishop_directions));
        }
    }
}

// NOLINTEND(readability-identifier-length)
