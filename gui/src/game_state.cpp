#include "game_state.hpp"

#include "eng/movegen.hpp"
#include "eng/rules.hpp"

#include <algorithm>

namespace {

constexpr std::array<eng::PieceType, 4> kPromotionPreference{
    eng::PieceType::Queen,
    eng::PieceType::Rook,
    eng::PieceType::Bishop,
    eng::PieceType::Knight,
};

[[nodiscard]] bool contains_square(std::span<const eng::Square> squares,
                                   eng::Square square) noexcept {
    return std::ranges::find(squares, square) != squares.end();
}

[[nodiscard]] std::optional<eng::Square> king_square(const eng::Position& position,
                                                     eng::Color color) noexcept {
    if (!eng::is_valid_color(color)) {
        return std::nullopt;
    }

    const auto kings = position.pieces(color, eng::PieceType::King);
    for (int index = 0; index < eng::Square::count; ++index) {
        const auto square = eng::Square::from_index(index);
        if ((kings & eng::square_bb(square)) != 0U) {
            return square;
        }
    }
    return std::nullopt;
}

[[nodiscard]] int promotion_preference_index(eng::PieceType type) noexcept {
    const auto found = std::ranges::find(kPromotionPreference, type);
    if (found == kPromotionPreference.end()) {
        return static_cast<int>(kPromotionPreference.size());
    }
    return static_cast<int>(found - kPromotionPreference.begin());
}

} // namespace

GuiGameState::GuiGameState() { new_game(); }

void GuiGameState::new_game() {
    initial_position_ = eng::Position::startpos();
    position_ = initial_position_;
    moves_.clear();
    previous_keys_.clear();
    pgn_headers_ = gui::pgn::default_headers("*");
    pgn_result_ = "*";
    current_ply_ = 0;
    last_move_.reset();
    clear_selection();
    status_ = classify_status();
}

void GuiGameState::set_position(const eng::Position& position,
                                std::span<const eng::ZobristKey> previous_keys) {
    initial_position_ = position;
    position_ = position;
    moves_.clear();
    previous_keys_.assign(previous_keys.begin(), previous_keys.end());
    pgn_headers_ = gui::pgn::default_headers("*");
    pgn_result_ = "*";
    current_ply_ = 0;
    last_move_.reset();
    clear_selection();
    status_ = classify_status();
    pgn_result_ = result_for_status(status_);
}

std::optional<eng::Square> GuiGameState::check_square() const noexcept {
    if (status_.kind != StatusKind::Check && status_.kind != StatusKind::Checkmate) {
        return std::nullopt;
    }
    return king_square(position_, position_.side_to_move());
}

bool GuiGameState::select_square(eng::Square square) {
    clear_selection();
    if (status_.terminal || !is_at_latest_ply() || !square.is_valid()) {
        return false;
    }

    const auto piece = position_.piece_on(square);
    if (piece.is_none() || piece.color() != position_.side_to_move()) {
        return false;
    }

    auto destinations = legal_destinations_for(square);
    if (destinations.empty()) {
        return false;
    }

    selected_square_ = square;
    legal_destinations_ = std::move(destinations);
    return true;
}

void GuiGameState::clear_selection() {
    selected_square_.reset();
    legal_destinations_.clear();
}

std::vector<eng::Square> GuiGameState::legal_destinations_for(eng::Square square) const {
    std::vector<eng::Square> destinations;
    if (status_.terminal || !is_at_latest_ply() || !square.is_valid()) {
        return destinations;
    }

    const auto legal_moves = eng::generate_legal_moves(position_);
    for (const auto move : legal_moves) {
        if (move.from() == square && !contains_square(destinations, move.to())) {
            destinations.push_back(move.to());
        }
    }
    return destinations;
}

bool GuiGameState::destination_is_legal(eng::Square square) const {
    return contains_square(legal_destinations_, square);
}

bool GuiGameState::go_to_ply(std::size_t ply) {
    if (ply > moves_.size()) {
        return false;
    }

    eng::Position replay = initial_position_;
    std::vector<eng::ZobristKey> replay_keys;
    replay_keys.reserve(ply);
    for (std::size_t index = 0; index < ply; ++index) {
        eng::StateInfo state;
        const auto previous_key = replay.zobrist_key();
        if (!replay.make_move(moves_[index], state)) {
            return false;
        }
        replay_keys.push_back(previous_key);
    }

    position_ = replay;
    previous_keys_ = std::move(replay_keys);
    current_ply_ = ply;
    last_move_ =
        current_ply_ > 0U ? std::optional<eng::Move>{moves_[current_ply_ - 1U]} : std::nullopt;
    clear_selection();
    status_ = classify_status();
    return true;
}

bool GuiGameState::load_pgn_game(const gui::pgn::Game& game) {
    eng::Position replay = game.initial_position;
    for (const auto move : game.moves) {
        eng::StateInfo state;
        if (!replay.make_move(move, state)) {
            return false;
        }
    }

    initial_position_ = game.initial_position;
    position_ = initial_position_;
    moves_ = game.moves;
    previous_keys_.clear();
    pgn_headers_ = game.headers.empty() ? gui::pgn::default_headers(game.result) : game.headers;
    pgn_result_ = game.result.empty() ? "*" : game.result;
    current_ply_ = 0;
    last_move_.reset();
    clear_selection();
    status_ = classify_status();
    return go_to_ply(moves_.size());
}

gui::pgn::Game GuiGameState::to_pgn_game() const {
    gui::pgn::Game game;
    game.headers = pgn_headers_.empty() ? gui::pgn::default_headers(pgn_result_) : pgn_headers_;
    game.initial_position = initial_position_;
    game.moves = moves_;

    eng::Position replay = initial_position_;
    std::vector<eng::ZobristKey> replay_keys;
    replay_keys.reserve(moves_.size());
    bool valid_replay = true;
    for (const auto move : moves_) {
        eng::StateInfo state;
        const auto previous_key = replay.zobrist_key();
        if (!replay.make_move(move, state)) {
            valid_replay = false;
            break;
        }
        replay_keys.push_back(previous_key);
    }

    std::string final_result = pgn_result_.empty() ? "*" : pgn_result_;
    if (valid_replay) {
        const auto final_status = classify_status_for(replay, replay_keys);
        const auto rules_result = result_for_status(final_status);
        if (rules_result != "*") {
            final_result = rules_result;
        }
    }
    game.result = final_result;
    return game;
}

GuiGameState::MoveAttempt GuiGameState::try_move(eng::Square from_square, eng::Square to_square,
                                                 std::optional<eng::PieceType> promotion_piece) {
    if (status_.terminal) {
        return MoveAttempt{.kind = MoveAttemptKind::GameOver};
    }
    if (!is_at_latest_ply()) {
        return {};
    }
    if (!from_square.is_valid() || !to_square.is_valid() || from_square == to_square) {
        return {};
    }

    auto candidates = legal_moves_between(from_square, to_square);
    if (candidates.empty()) {
        return {};
    }

    const auto promotion_count =
        std::ranges::count_if(candidates, [](eng::Move move) { return move.is_promotion(); });
    if (promotion_count > 1 && !promotion_piece.has_value()) {
        return promotion_required(candidates);
    }

    if (promotion_piece.has_value()) {
        const auto found =
            std::ranges::find_if(candidates, [promotion_piece](eng::Move move) noexcept {
                return move.is_promotion() && move.promotion_piece() == *promotion_piece;
            });
        if (found == candidates.end()) {
            return {};
        }
        return apply_move(*found);
    }

    return apply_move(candidates.front());
}

GuiGameState::GameStatus GuiGameState::classify_status() const {
    return classify_status_for(position_, previous_keys_);
}

GuiGameState::GameStatus
GuiGameState::classify_status_for(const eng::Position& position,
                                  std::span<const eng::ZobristKey> previous_keys) {
    GameStatus status{.side_to_move = position.side_to_move()};
    if (eng::is_checkmate(position)) {
        status.kind = StatusKind::Checkmate;
        status.winner = eng::opposite(position.side_to_move());
        status.terminal = true;
        return status;
    }
    if (eng::is_stalemate(position)) {
        status.kind = StatusKind::Stalemate;
        status.terminal = true;
        return status;
    }
    if (eng::is_fifty_move_rule_draw(position)) {
        status.kind = StatusKind::DrawFiftyMove;
        status.terminal = true;
        return status;
    }
    if (eng::is_threefold_repetition(position, previous_keys)) {
        status.kind = StatusKind::DrawRepetition;
        status.terminal = true;
        return status;
    }
    if (eng::is_insufficient_material(position)) {
        status.kind = StatusKind::DrawInsufficientMaterial;
        status.terminal = true;
        return status;
    }
    if (eng::side_to_move_in_check(position)) {
        status.kind = StatusKind::Check;
        return status;
    }
    return status;
}

std::string GuiGameState::result_for_status(const GameStatus& status) {
    switch (status.kind) {
    case StatusKind::Checkmate:
        return status.winner == eng::Color::White ? "1-0" : "0-1";
    case StatusKind::Stalemate:
    case StatusKind::DrawFiftyMove:
    case StatusKind::DrawRepetition:
    case StatusKind::DrawInsufficientMaterial:
        return "1/2-1/2";
    case StatusKind::Playing:
    case StatusKind::Check:
        break;
    }
    return "*";
}

std::vector<eng::Move> GuiGameState::legal_moves_between(eng::Square from_square,
                                                         eng::Square to_square) const {
    std::vector<eng::Move> moves;
    const auto legal_moves = eng::generate_legal_moves(position_);
    for (const auto move : legal_moves) {
        if (move.from() == from_square && move.to() == to_square) {
            moves.push_back(move);
        }
    }
    std::ranges::sort(moves, [](eng::Move lhs, eng::Move rhs) noexcept {
        return promotion_preference_index(lhs.promotion_piece()) <
               promotion_preference_index(rhs.promotion_piece());
    });
    return moves;
}

GuiGameState::MoveAttempt GuiGameState::promotion_required(std::span<const eng::Move> moves) {
    MoveAttempt attempt{.kind = MoveAttemptKind::PromotionRequired};
    for (const auto preferred : kPromotionPreference) {
        const auto found = std::ranges::find_if(moves, [preferred](eng::Move move) noexcept {
            return move.is_promotion() && move.promotion_piece() == preferred;
        });
        if (found != moves.end() && attempt.promotion_count < attempt.promotion_pieces.size()) {
            attempt.promotion_pieces[attempt.promotion_count] = preferred;
            ++attempt.promotion_count;
        }
    }
    return attempt;
}

GuiGameState::MoveAttempt GuiGameState::apply_move(eng::Move move) {
    eng::StateInfo state;
    const auto previous_key = position_.zobrist_key();
    if (!position_.make_move(move, state)) {
        return {};
    }

    previous_keys_.push_back(previous_key);
    moves_.push_back(move);
    current_ply_ = moves_.size();
    last_move_ = move;
    clear_selection();
    status_ = classify_status();
    pgn_result_ = result_for_status(status_);
    return MoveAttempt{.kind = MoveAttemptKind::Applied, .move = move};
}
