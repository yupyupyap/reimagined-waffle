#ifndef GUI_GAME_STATE_HPP
#define GUI_GAME_STATE_HPP

#include "eng/move.hpp"
#include "eng/position.hpp"
#include "eng/types.hpp"
#include "pgn.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

class GuiGameState final {
  public:
    enum class StatusKind : std::uint8_t {
        Playing,
        Check,
        Checkmate,
        Stalemate,
        DrawFiftyMove,
        DrawRepetition,
        DrawInsufficientMaterial
    };

    struct GameStatus {
        StatusKind kind{StatusKind::Playing};
        eng::Color side_to_move{eng::Color::White};
        eng::Color winner{eng::Color::None};
        bool terminal{false};
    };

    enum class MoveAttemptKind : std::uint8_t { Invalid, Applied, PromotionRequired, GameOver };

    struct MoveAttempt {
        MoveAttemptKind kind{MoveAttemptKind::Invalid};
        eng::Move move{eng::Move::none()};
        std::array<eng::PieceType, 4> promotion_pieces{};
        std::size_t promotion_count{0};
    };

    GuiGameState();

    void new_game();
    void set_position(const eng::Position& position,
                      std::span<const eng::ZobristKey> previous_keys = {});

    [[nodiscard]] const eng::Position& position() const noexcept { return position_; }
    [[nodiscard]] const std::vector<eng::ZobristKey>& previous_keys() const noexcept {
        return previous_keys_;
    }
    [[nodiscard]] const std::vector<eng::Move>& moves() const noexcept { return moves_; }
    [[nodiscard]] const eng::Position& initial_position() const noexcept {
        return initial_position_;
    }
    [[nodiscard]] std::size_t current_ply() const noexcept { return current_ply_; }
    [[nodiscard]] std::size_t move_count() const noexcept { return moves_.size(); }
    [[nodiscard]] bool is_at_latest_ply() const noexcept { return current_ply_ == moves_.size(); }
    [[nodiscard]] const GameStatus& status() const noexcept { return status_; }
    [[nodiscard]] std::optional<eng::Move> last_move() const noexcept { return last_move_; }
    [[nodiscard]] std::optional<eng::Square> selected_square() const noexcept {
        return selected_square_;
    }
    [[nodiscard]] const std::vector<eng::Square>& legal_destinations() const noexcept {
        return legal_destinations_;
    }
    [[nodiscard]] std::optional<eng::Square> check_square() const noexcept;

    [[nodiscard]] bool select_square(eng::Square square);
    void clear_selection();
    [[nodiscard]] bool go_to_ply(std::size_t ply);
    [[nodiscard]] bool load_pgn_game(const gui::pgn::Game& game);
    [[nodiscard]] gui::pgn::Game to_pgn_game() const;
    [[nodiscard]] std::vector<eng::Square> legal_destinations_for(eng::Square square) const;
    [[nodiscard]] bool destination_is_legal(eng::Square square) const;
    [[nodiscard]] MoveAttempt try_move(eng::Square from_square, eng::Square to_square,
                                       std::optional<eng::PieceType> promotion_piece = {});

  private:
    [[nodiscard]] GameStatus classify_status() const;
    [[nodiscard]] static GameStatus
    classify_status_for(const eng::Position& position,
                        std::span<const eng::ZobristKey> previous_keys);
    [[nodiscard]] static std::string result_for_status(const GameStatus& status);
    [[nodiscard]] std::vector<eng::Move> legal_moves_between(eng::Square from_square,
                                                             eng::Square to_square) const;
    [[nodiscard]] static MoveAttempt promotion_required(std::span<const eng::Move> moves);
    [[nodiscard]] MoveAttempt apply_move(eng::Move move);

    eng::Position initial_position_{eng::Position::startpos()};
    eng::Position position_{eng::Position::startpos()};
    std::vector<eng::Move> moves_;
    std::vector<eng::ZobristKey> previous_keys_;
    std::vector<gui::pgn::Header> pgn_headers_;
    std::string pgn_result_{"*"};
    std::size_t current_ply_{0};
    std::optional<eng::Move> last_move_;
    std::optional<eng::Square> selected_square_;
    std::vector<eng::Square> legal_destinations_;
    GameStatus status_;
};

#endif // GUI_GAME_STATE_HPP
