#ifndef ENG_TYPES_HPP
#define ENG_TYPES_HPP

#include <cstdint>

namespace eng {

enum class Color : std::uint8_t { White = 0, Black = 1, None = 2 };

[[nodiscard]] constexpr bool is_valid_color(Color color) noexcept {
    return color == Color::White || color == Color::Black;
}

[[nodiscard]] constexpr Color opposite(Color color) noexcept {
    if (color == Color::White) {
        return Color::Black;
    }
    if (color == Color::Black) {
        return Color::White;
    }
    return Color::None;
}

enum class PieceType : std::uint8_t {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
    None = 6
};

[[nodiscard]] constexpr bool is_valid_piece_type(PieceType type) noexcept {
    return type >= PieceType::Pawn && type <= PieceType::King;
}

[[nodiscard]] constexpr bool is_valid_promotion_piece(PieceType type) noexcept {
    return type == PieceType::Knight || type == PieceType::Bishop || type == PieceType::Rook ||
           type == PieceType::Queen;
}

class File {
  public:
    static constexpr int count = 8;

    constexpr File() noexcept = default;

    [[nodiscard]] static constexpr File from_index(int index) noexcept {
        if (index < 0 || index >= count) {
            return File{};
        }
        return File{static_cast<std::uint8_t>(index)};
    }

    [[nodiscard]] constexpr bool is_valid() const noexcept { return index_ < count; }
    [[nodiscard]] constexpr int index() const noexcept { return index_; }

    friend constexpr bool operator==(File lhs, File rhs) noexcept = default;

  private:
    constexpr explicit File(std::uint8_t index) noexcept : index_{index} {}

    std::uint8_t index_{count};
};

class Rank {
  public:
    static constexpr int count = 8;

    constexpr Rank() noexcept = default;

    [[nodiscard]] static constexpr Rank from_index(int index) noexcept {
        if (index < 0 || index >= count) {
            return Rank{};
        }
        return Rank{static_cast<std::uint8_t>(index)};
    }

    [[nodiscard]] constexpr bool is_valid() const noexcept { return index_ < count; }
    [[nodiscard]] constexpr int index() const noexcept { return index_; }

    friend constexpr bool operator==(Rank lhs, Rank rhs) noexcept = default;

  private:
    constexpr explicit Rank(std::uint8_t index) noexcept : index_{index} {}

    std::uint8_t index_{count};
};

class Square {
  public:
    static constexpr int count = 64;

    constexpr Square() noexcept = default;

    [[nodiscard]] static constexpr Square from_index(int index) noexcept {
        if (index < 0 || index >= count) {
            return Square{};
        }
        return Square{static_cast<std::uint8_t>(index)};
    }

    [[nodiscard]] static constexpr Square from_file_rank(File file, Rank rank) noexcept {
        if (!file.is_valid() || !rank.is_valid()) {
            return Square{};
        }
        return from_index((rank.index() * File::count) + file.index());
    }

    [[nodiscard]] constexpr bool is_valid() const noexcept { return index_ < count; }
    [[nodiscard]] constexpr int index() const noexcept { return index_; }

    [[nodiscard]] constexpr File file() const noexcept {
        if (!is_valid()) {
            return File{};
        }
        return File::from_index(index_ % File::count);
    }

    [[nodiscard]] constexpr Rank rank() const noexcept {
        if (!is_valid()) {
            return Rank{};
        }
        return Rank::from_index(index_ / File::count);
    }

    friend constexpr bool operator==(Square lhs, Square rhs) noexcept = default;

  private:
    constexpr explicit Square(std::uint8_t index) noexcept : index_{index} {}

    std::uint8_t index_{count};
};

using Bitboard = std::uint64_t;

[[nodiscard]] constexpr Bitboard square_bb(Square square) noexcept {
    if (!square.is_valid()) {
        return 0;
    }
    return Bitboard{1} << square.index();
}

class Piece {
  public:
    static constexpr int type_count = 6;
    static constexpr int count = 12;

    constexpr Piece() noexcept = default;

    [[nodiscard]] static constexpr Piece make(Color color, PieceType type) noexcept {
        if (!is_valid_color(color) || !is_valid_piece_type(type)) {
            return Piece{};
        }

        const auto color_offset = static_cast<int>(color) * type_count;
        return Piece{static_cast<std::uint8_t>(color_offset + static_cast<int>(type))};
    }

    [[nodiscard]] constexpr bool is_valid() const noexcept { return value_ < count; }
    [[nodiscard]] constexpr bool is_none() const noexcept { return !is_valid(); }
    [[nodiscard]] constexpr int index() const noexcept { return value_; }

    [[nodiscard]] constexpr Color color() const noexcept {
        if (!is_valid()) {
            return Color::None;
        }
        return value_ < type_count ? Color::White : Color::Black;
    }

    [[nodiscard]] constexpr PieceType type() const noexcept {
        if (!is_valid()) {
            return PieceType::None;
        }
        return static_cast<PieceType>(value_ % type_count);
    }

    friend constexpr bool operator==(Piece lhs, Piece rhs) noexcept = default;

  private:
    constexpr explicit Piece(std::uint8_t value) noexcept : value_{value} {}

    std::uint8_t value_{count};
};

class Score {
  public:
    constexpr Score() noexcept = default;

    [[nodiscard]] static constexpr Score centipawns(int value) noexcept { return Score{value}; }
    [[nodiscard]] constexpr int value() const noexcept { return value_; }

    friend constexpr bool operator==(Score lhs, Score rhs) noexcept = default;

  private:
    constexpr explicit Score(int value) noexcept : value_{value} {}

    int value_{0};
};

class Depth {
  public:
    constexpr Depth() noexcept = default;

    [[nodiscard]] static constexpr Depth plies(int value) noexcept { return Depth{value}; }
    [[nodiscard]] constexpr int value() const noexcept { return value_; }

    friend constexpr bool operator==(Depth lhs, Depth rhs) noexcept = default;

  private:
    constexpr explicit Depth(int value) noexcept : value_{value} {}

    int value_{0};
};

class Ply {
  public:
    constexpr Ply() noexcept = default;

    [[nodiscard]] static constexpr Ply from_root(int value) noexcept { return Ply{value}; }
    [[nodiscard]] constexpr int value() const noexcept { return value_; }

    friend constexpr bool operator==(Ply lhs, Ply rhs) noexcept = default;

  private:
    constexpr explicit Ply(int value) noexcept : value_{value} {}

    int value_{0};
};

} // namespace eng

#endif // ENG_TYPES_HPP
