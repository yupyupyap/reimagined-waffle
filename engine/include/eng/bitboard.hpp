#ifndef ENG_BITBOARD_HPP
#define ENG_BITBOARD_HPP

#include "eng/types.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <limits>

namespace eng {

enum class Direction : std::uint8_t {
    North = 0,
    South = 1,
    East = 2,
    West = 3,
    NorthEast = 4,
    NorthWest = 5,
    SouthEast = 6,
    SouthWest = 7
};

struct MagicEntry {
    Bitboard mask{0};
    Bitboard magic{0};
    std::uint8_t shift{0};
    std::uint32_t offset{0};

    friend constexpr bool operator==(MagicEntry lhs, MagicEntry rhs) noexcept = default;
};

inline constexpr int direction_count = 8;
inline constexpr Bitboard empty_bitboard = 0;
inline constexpr Bitboard full_bitboard = std::numeric_limits<Bitboard>::max();

[[nodiscard]] constexpr int popcount(Bitboard bitboard) noexcept { return std::popcount(bitboard); }

[[nodiscard]] constexpr Square lsb(Bitboard bitboard) noexcept {
    if (bitboard == empty_bitboard) {
        return Square{};
    }
    return Square::from_index(std::countr_zero(bitboard));
}

[[nodiscard]] constexpr Square pop_lsb(Bitboard& bitboard) noexcept {
    const auto square = lsb(bitboard);
    if (square.is_valid()) {
        bitboard &= bitboard - 1U;
    }
    return square;
}

[[nodiscard]] constexpr bool has_square(Bitboard bitboard, Square square) noexcept {
    return (square_bb(square) & bitboard) != empty_bitboard;
}

constexpr void set_square(Bitboard& bitboard, Square square) noexcept {
    bitboard |= square_bb(square);
}

constexpr void clear_square(Bitboard& bitboard, Square square) noexcept {
    bitboard &= ~square_bb(square);
}

namespace bitboard_detail {

struct Offset {
    int file_delta;
    int rank_delta;
};

inline constexpr std::array<Offset, direction_count> direction_offsets{{
    {.file_delta = 0, .rank_delta = 1},   // North
    {.file_delta = 0, .rank_delta = -1},  // South
    {.file_delta = 1, .rank_delta = 0},   // East
    {.file_delta = -1, .rank_delta = 0},  // West
    {.file_delta = 1, .rank_delta = 1},   // NorthEast
    {.file_delta = -1, .rank_delta = 1},  // NorthWest
    {.file_delta = 1, .rank_delta = -1},  // SouthEast
    {.file_delta = -1, .rank_delta = -1}, // SouthWest
}};

[[nodiscard]] constexpr int direction_index(Direction direction) noexcept {
    return static_cast<int>(direction);
}

[[nodiscard]] constexpr Direction opposite(Direction direction) noexcept {
    switch (direction) {
    case Direction::North:
        return Direction::South;
    case Direction::South:
        return Direction::North;
    case Direction::East:
        return Direction::West;
    case Direction::West:
        return Direction::East;
    case Direction::NorthEast:
        return Direction::SouthWest;
    case Direction::NorthWest:
        return Direction::SouthEast;
    case Direction::SouthEast:
        return Direction::NorthWest;
    case Direction::SouthWest:
        return Direction::NorthEast;
    }
    return Direction::North;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] constexpr Square offset_square(Square square, int file_delta,
                                             int rank_delta) noexcept {
    if (!square.is_valid()) {
        return Square{};
    }

    const auto file = square.file().index() + file_delta;
    const auto rank = square.rank().index() + rank_delta;
    return Square::from_file_rank(File::from_index(file), Rank::from_index(rank));
}

[[nodiscard]] constexpr Square step(Square square, Direction direction) noexcept {
    const auto offset = direction_offsets[direction_index(direction)];
    return offset_square(square, offset.file_delta, offset.rank_delta);
}

[[nodiscard]] constexpr int sign(int value) noexcept {
    if (value > 0) {
        return 1;
    }
    if (value < 0) {
        return -1;
    }
    return 0;
}

[[nodiscard]] constexpr int abs_value(int value) noexcept { return value < 0 ? -value : value; }

[[nodiscard]] constexpr bool is_on_board(int file, int rank) noexcept {
    return file >= 0 && file < File::count && rank >= 0 && rank < Rank::count;
}

[[nodiscard]] constexpr bool are_aligned(Square from_square, Square to_square) noexcept {
    if (!from_square.is_valid() || !to_square.is_valid()) {
        return false;
    }

    const auto file_delta = to_square.file().index() - from_square.file().index();
    const auto rank_delta = to_square.rank().index() - from_square.rank().index();
    return file_delta == 0 || rank_delta == 0 || abs_value(file_delta) == abs_value(rank_delta);
}

[[nodiscard]] constexpr Direction direction_between(Square from_square, Square to_square) noexcept {
    const auto file_delta = sign(to_square.file().index() - from_square.file().index());
    const auto rank_delta = sign(to_square.rank().index() - from_square.rank().index());

    if (file_delta == 0 && rank_delta > 0) {
        return Direction::North;
    }
    if (file_delta == 0 && rank_delta < 0) {
        return Direction::South;
    }
    if (file_delta > 0 && rank_delta == 0) {
        return Direction::East;
    }
    if (file_delta < 0 && rank_delta == 0) {
        return Direction::West;
    }
    if (file_delta > 0 && rank_delta > 0) {
        return Direction::NorthEast;
    }
    if (file_delta < 0 && rank_delta > 0) {
        return Direction::NorthWest;
    }
    if (file_delta > 0 && rank_delta < 0) {
        return Direction::SouthEast;
    }
    return Direction::SouthWest;
}

[[nodiscard]] consteval Bitboard make_king_attacks(Square square) {
    Bitboard attacks = empty_bitboard;
    for (int file_delta = -1; file_delta <= 1; ++file_delta) {
        for (int rank_delta = -1; rank_delta <= 1; ++rank_delta) {
            if (file_delta == 0 && rank_delta == 0) {
                continue;
            }
            attacks |= square_bb(offset_square(square, file_delta, rank_delta));
        }
    }
    return attacks;
}

[[nodiscard]] consteval Bitboard make_knight_attacks(Square square) {
    constexpr std::array<Offset, 8> knight_offsets{{
        {.file_delta = 1, .rank_delta = 2},
        {.file_delta = 2, .rank_delta = 1},
        {.file_delta = 2, .rank_delta = -1},
        {.file_delta = 1, .rank_delta = -2},
        {.file_delta = -1, .rank_delta = -2},
        {.file_delta = -2, .rank_delta = -1},
        {.file_delta = -2, .rank_delta = 1},
        {.file_delta = -1, .rank_delta = 2},
    }};

    Bitboard attacks = empty_bitboard;
    for (const auto offset : knight_offsets) {
        attacks |= square_bb(offset_square(square, offset.file_delta, offset.rank_delta));
    }
    return attacks;
}

[[nodiscard]] consteval Bitboard make_pawn_attacks(Color color, Square square) {
    const auto rank_delta = color == Color::White ? 1 : -1;
    return square_bb(offset_square(square, -1, rank_delta)) |
           square_bb(offset_square(square, 1, rank_delta));
}

[[nodiscard]] consteval Bitboard make_ray(Square square, Direction direction) {
    Bitboard result = empty_bitboard;
    auto current = step(square, direction);
    while (current.is_valid()) {
        result |= square_bb(current);
        current = step(current, direction);
    }
    return result;
}

[[nodiscard]] consteval Bitboard make_between(Square from_square, Square to_square) {
    if (from_square == to_square || !are_aligned(from_square, to_square)) {
        return empty_bitboard;
    }

    const auto direction = direction_between(from_square, to_square);
    Bitboard result = empty_bitboard;
    auto current = step(from_square, direction);
    while (current.is_valid() && current != to_square) {
        result |= square_bb(current);
        current = step(current, direction);
    }
    return current == to_square ? result : empty_bitboard;
}

[[nodiscard]] consteval Bitboard make_line(Square from_square, Square to_square) {
    if (!from_square.is_valid() || !to_square.is_valid()) {
        return empty_bitboard;
    }
    if (from_square == to_square) {
        return square_bb(from_square);
    }

    const auto from_file = from_square.file().index();
    const auto from_rank = from_square.rank().index();
    const auto to_file = to_square.file().index();
    const auto to_rank = to_square.rank().index();
    const auto file_delta = to_file - from_file;
    const auto rank_delta = to_rank - from_rank;
    if (file_delta != 0 && rank_delta != 0 && abs_value(file_delta) != abs_value(rank_delta)) {
        return empty_bitboard;
    }

    const auto file_step = sign(file_delta);
    const auto rank_step = sign(rank_delta);
    auto file = from_file;
    auto rank = from_rank;
    while (is_on_board(file - file_step, rank - rank_step)) {
        file -= file_step;
        rank -= rank_step;
    }

    Bitboard result = empty_bitboard;
    while (is_on_board(file, rank)) {
        result |= square_bb(Square::from_file_rank(File::from_index(file), Rank::from_index(rank)));
        file += file_step;
        rank += rank_step;
    }
    return result;
}

template <std::size_t DirectionCount>
[[nodiscard]] consteval Bitboard
make_relevant_occupancy(Square square, const std::array<Direction, DirectionCount>& directions) {
    Bitboard mask = empty_bitboard;
    for (const auto direction : directions) {
        auto current = step(square, direction);
        while (current.is_valid()) {
            const auto next = step(current, direction);
            if (!next.is_valid()) {
                break;
            }
            mask |= square_bb(current);
            current = next;
        }
    }
    return mask;
}

template <typename Maker>
[[nodiscard]] consteval std::array<Bitboard, Square::count> make_square_table(Maker maker) {
    std::array<Bitboard, Square::count> table{};
    for (int index = 0; index < Square::count; ++index) {
        table[static_cast<std::size_t>(index)] = maker(Square::from_index(index));
    }
    return table;
}

[[nodiscard]] consteval auto make_pawn_table(Color color) {
    return make_square_table([color](Square square) { return make_pawn_attacks(color, square); });
}

[[nodiscard]] consteval auto make_ray_table() {
    std::array<std::array<Bitboard, Square::count>, direction_count> table{};
    for (int direction = 0; direction < direction_count; ++direction) {
        for (int square = 0; square < Square::count; ++square) {
            table[static_cast<std::size_t>(direction)][static_cast<std::size_t>(square)] =
                make_ray(Square::from_index(square), static_cast<Direction>(direction));
        }
    }
    return table;
}

[[nodiscard]] consteval auto make_between_table() {
    std::array<std::array<Bitboard, Square::count>, Square::count> table{};
    for (int from = 0; from < Square::count; ++from) {
        for (int to = 0; to < Square::count; ++to) {
            table[static_cast<std::size_t>(from)][static_cast<std::size_t>(to)] =
                make_between(Square::from_index(from), Square::from_index(to));
        }
    }
    return table;
}

[[nodiscard]] consteval auto make_line_table() {
    std::array<std::array<Bitboard, Square::count>, Square::count> table{};
    for (int from = 0; from < Square::count; ++from) {
        for (int to = 0; to < Square::count; ++to) {
            table[static_cast<std::size_t>(from)][static_cast<std::size_t>(to)] =
                make_line(Square::from_index(from), Square::from_index(to));
        }
    }
    return table;
}

inline constexpr std::array<Direction, 4> rook_directions{
    Direction::North,
    Direction::South,
    Direction::East,
    Direction::West,
};

inline constexpr std::array<Direction, 4> bishop_directions{
    Direction::NorthEast,
    Direction::NorthWest,
    Direction::SouthEast,
    Direction::SouthWest,
};

[[nodiscard]] consteval auto make_rook_relevant_occupancy_table() {
    return make_square_table(
        [](Square square) { return make_relevant_occupancy(square, rook_directions); });
}

[[nodiscard]] consteval auto make_bishop_relevant_occupancy_table() {
    return make_square_table(
        [](Square square) { return make_relevant_occupancy(square, bishop_directions); });
}

[[nodiscard]] consteval std::size_t
table_size_for_masks(const std::array<Bitboard, Square::count>& masks) {
    std::size_t size = 0;
    for (const auto mask : masks) {
        size += std::size_t{1} << popcount(mask);
    }
    return size;
}

} // namespace bitboard_detail

inline constexpr auto king_attack_table =
    bitboard_detail::make_square_table(bitboard_detail::make_king_attacks);
inline constexpr auto knight_attack_table =
    bitboard_detail::make_square_table(bitboard_detail::make_knight_attacks);
inline constexpr auto white_pawn_attack_table = bitboard_detail::make_pawn_table(Color::White);
inline constexpr auto black_pawn_attack_table = bitboard_detail::make_pawn_table(Color::Black);
inline constexpr auto ray_table = bitboard_detail::make_ray_table();
inline constexpr auto between_table = bitboard_detail::make_between_table();
inline constexpr auto line_table = bitboard_detail::make_line_table();
inline constexpr auto rook_relevant_occupancy_table =
    bitboard_detail::make_rook_relevant_occupancy_table();
inline constexpr auto bishop_relevant_occupancy_table =
    bitboard_detail::make_bishop_relevant_occupancy_table();
inline constexpr std::size_t rook_magic_table_size =
    bitboard_detail::table_size_for_masks(rook_relevant_occupancy_table);
inline constexpr std::size_t bishop_magic_table_size =
    bitboard_detail::table_size_for_masks(bishop_relevant_occupancy_table);
inline constexpr std::size_t magic_attack_table_size =
    rook_magic_table_size + bishop_magic_table_size;
inline constexpr std::size_t expected_rook_magic_table_size = 102400;
inline constexpr std::size_t expected_bishop_magic_table_size = 5248;
inline constexpr std::size_t expected_magic_attack_table_size =
    expected_rook_magic_table_size + expected_bishop_magic_table_size;

[[nodiscard]] constexpr Bitboard king_attacks(Square square) noexcept {
    if (!square.is_valid()) {
        return empty_bitboard;
    }
    return king_attack_table[static_cast<std::size_t>(square.index())];
}

[[nodiscard]] constexpr Bitboard knight_attacks(Square square) noexcept {
    if (!square.is_valid()) {
        return empty_bitboard;
    }
    return knight_attack_table[static_cast<std::size_t>(square.index())];
}

[[nodiscard]] constexpr Bitboard pawn_attacks(Color color, Square square) noexcept {
    if (!square.is_valid()) {
        return empty_bitboard;
    }
    if (color == Color::White) {
        return white_pawn_attack_table[static_cast<std::size_t>(square.index())];
    }
    if (color == Color::Black) {
        return black_pawn_attack_table[static_cast<std::size_t>(square.index())];
    }
    return empty_bitboard;
}

[[nodiscard]] constexpr Bitboard ray(Square square, Direction direction) noexcept {
    if (!square.is_valid()) {
        return empty_bitboard;
    }
    return ray_table[static_cast<std::size_t>(bitboard_detail::direction_index(direction))]
                    [static_cast<std::size_t>(square.index())];
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] constexpr Bitboard between(Square from_square, Square to_square) noexcept {
    if (!from_square.is_valid() || !to_square.is_valid()) {
        return empty_bitboard;
    }
    return between_table[static_cast<std::size_t>(from_square.index())]
                        [static_cast<std::size_t>(to_square.index())];
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] constexpr Bitboard line(Square from_square, Square to_square) noexcept {
    if (!from_square.is_valid() || !to_square.is_valid()) {
        return empty_bitboard;
    }
    return line_table[static_cast<std::size_t>(from_square.index())]
                     [static_cast<std::size_t>(to_square.index())];
}

[[nodiscard]] constexpr Bitboard rook_relevant_occupancy(Square square) noexcept {
    if (!square.is_valid()) {
        return empty_bitboard;
    }
    return rook_relevant_occupancy_table[static_cast<std::size_t>(square.index())];
}

[[nodiscard]] constexpr Bitboard bishop_relevant_occupancy(Square square) noexcept {
    if (!square.is_valid()) {
        return empty_bitboard;
    }
    return bishop_relevant_occupancy_table[static_cast<std::size_t>(square.index())];
}

void init_bitboards() noexcept;

[[nodiscard]] const MagicEntry& rook_magic(Square square) noexcept;
[[nodiscard]] const MagicEntry& bishop_magic(Square square) noexcept;
[[nodiscard]] Bitboard rook_attacks(Square square, Bitboard occupancy) noexcept;
[[nodiscard]] Bitboard bishop_attacks(Square square, Bitboard occupancy) noexcept;
[[nodiscard]] Bitboard queen_attacks(Square square, Bitboard occupancy) noexcept;

static_assert(rook_magic_table_size == expected_rook_magic_table_size);
static_assert(bishop_magic_table_size == expected_bishop_magic_table_size);
static_assert(magic_attack_table_size == expected_magic_attack_table_size);

} // namespace eng

#endif // ENG_BITBOARD_HPP
