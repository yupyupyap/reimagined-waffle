#ifndef ENG_EVAL_HPP
#define ENG_EVAL_HPP

#include "eng/position.hpp"
#include "eng/types.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace eng {

using EvaluationPieceValues = std::array<int, Piece::type_count>;
using EvaluationRankBonuses = std::array<int, Rank::count>;

struct EvaluationOptions {
    bool material_and_psqt{true};
    bool mobility{true};
    bool pawn_structure{true};
    bool king_safety{true};
    bool bishop_pair{true};
    bool rook_files{true};
};

struct EvaluationParameters {
    EvaluationPieceValues middle_game_piece_values{};
    EvaluationPieceValues end_game_piece_values{};
    EvaluationPieceValues middle_game_mobility_weights{};
    EvaluationPieceValues end_game_mobility_weights{};
    EvaluationRankBonuses middle_game_passed_pawn_bonus{};
    EvaluationRankBonuses end_game_passed_pawn_bonus{};

    int middle_game_isolated_pawn_penalty{0};
    int end_game_isolated_pawn_penalty{0};
    int middle_game_doubled_pawn_penalty{0};
    int end_game_doubled_pawn_penalty{0};
    int middle_game_bishop_pair_bonus{0};
    int end_game_bishop_pair_bonus{0};
    int middle_game_rook_open_file_bonus{0};
    int end_game_rook_open_file_bonus{0};
    int middle_game_rook_semi_open_file_bonus{0};
    int end_game_rook_semi_open_file_bonus{0};
    int king_shield_rank_one_bonus{0};
    int king_shield_rank_two_bonus{0};
    int king_semi_open_file_penalty{0};
    int king_open_file_penalty{0};
    int king_zone_attack_penalty{0};
};

struct EvaluationParameterDescriptor {
    std::string_view name;
    int default_value;
    int min_value;
    int max_value;
};

[[nodiscard]] EvaluationParameters default_evaluation_parameters() noexcept;
[[nodiscard]] std::span<const EvaluationParameterDescriptor>
evaluation_parameter_descriptors() noexcept;
[[nodiscard]] int evaluation_parameter_value(const EvaluationParameters& parameters,
                                             std::string_view name) noexcept;
[[nodiscard]] bool set_evaluation_parameter(EvaluationParameters& parameters, std::string_view name,
                                            int value) noexcept;
[[nodiscard]] std::uint64_t
evaluation_parameters_checksum(const EvaluationParameters& parameters) noexcept;

[[nodiscard]] Score evaluate(const Position& position) noexcept;
[[nodiscard]] Score evaluate(const Position& position, EvaluationOptions options) noexcept;
[[nodiscard]] Score evaluate(const Position& position, EvaluationOptions options,
                             const EvaluationParameters& parameters) noexcept;

} // namespace eng

#endif // ENG_EVAL_HPP
