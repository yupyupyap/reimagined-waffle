#ifndef ENG_MOVEGEN_HPP
#define ENG_MOVEGEN_HPP

#include "eng/bitboard.hpp"
#include "eng/move.hpp"
#include "eng/position.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace eng {

enum class MoveGenStage : std::uint8_t { All, Captures, Quiets, Evasions };

inline constexpr std::size_t max_pseudo_legal_moves = 256;

class MoveListBuilder;

class MoveList {
  public:
    using Storage = std::array<Move, max_pseudo_legal_moves>;
    using const_iterator = Storage::const_iterator;

    [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0U; }

    [[nodiscard]] constexpr const_iterator begin() const noexcept { return moves_.begin(); }
    [[nodiscard]] constexpr const_iterator end() const noexcept { return moves_.begin() + size_; }

    [[nodiscard]] constexpr Move operator[](std::size_t index) const noexcept {
        return index < size_ ? moves_[index] : Move::none();
    }

  private:
    friend class MoveListBuilder;

    [[nodiscard]] constexpr bool push(Move move) noexcept {
        if (!move.is_valid() || size_ >= moves_.size()) {
            return false;
        }
        moves_[size_] = move;
        ++size_;
        return true;
    }

    Storage moves_{};
    std::size_t size_{0};
};

[[nodiscard]] MoveList generate_moves(const Position& position, MoveGenStage stage);
[[nodiscard]] MoveList generate_moves(const Position& position);
[[nodiscard]] MoveList generate_captures(const Position& position);
[[nodiscard]] MoveList generate_quiets(const Position& position);
[[nodiscard]] MoveList generate_evasions(const Position& position);
[[nodiscard]] MoveList generate_legal_moves(const Position& position);
[[nodiscard]] bool has_legal_moves(const Position& position);
[[nodiscard]] bool is_checkmate(const Position& position);
[[nodiscard]] bool is_stalemate(const Position& position);

[[nodiscard]] Bitboard attackers_to(const Position& position, Square target,
                                    Color attacker_color) noexcept;
[[nodiscard]] Bitboard attackers_to(const Position& position, Square target, Color attacker_color,
                                    Bitboard occupancy) noexcept;
[[nodiscard]] bool is_square_attacked(const Position& position, Square target,
                                      Color attacker_color) noexcept;
[[nodiscard]] bool is_square_attacked(const Position& position, Square target, Color attacker_color,
                                      Bitboard occupancy) noexcept;
[[nodiscard]] bool in_check(const Position& position, Color color) noexcept;
[[nodiscard]] bool side_to_move_in_check(const Position& position) noexcept;

} // namespace eng

#endif // ENG_MOVEGEN_HPP
