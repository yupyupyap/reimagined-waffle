#include <eng/eval.hpp>

#include <eng/bitboard.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <ranges>

namespace eng {
namespace {

enum class EvalPhase : std::uint8_t { MiddleGame, EndGame };

using PhaseWeights = std::array<int, Piece::type_count>;
using PieceSquareTable = std::array<std::array<int, Square::count>, Piece::type_count>;
using FileMasks = std::array<Bitboard, File::count>;

constexpr int max_game_phase = 24;
constexpr int mirror_vertical_mask = 56;
constexpr int scaled_board_center = 7;
constexpr int file_centrality_base = 6;
constexpr int center_score_base = 14;
constexpr int doubled_coordinate_scale = 2;
constexpr int pawn_middle_game_rank_bonus = 4;
constexpr int pawn_end_game_rank_bonus = 8;
constexpr int knight_middle_game_center_weight = 3;
constexpr int knight_end_game_center_weight = 2;
constexpr int bishop_center_weight = 2;
constexpr int rook_middle_game_rank_bonus = 2;
constexpr int rook_end_game_rank_bonus = 3;
constexpr int rook_edge_file_penalty = -4;
constexpr int queen_end_game_rank_bonus = 1;
constexpr int king_middle_game_center_weight = -3;
constexpr int king_middle_game_rank_penalty = -4;
constexpr int king_end_game_center_weight = 4;
constexpr int king_end_game_rank_bonus = 2;

inline constexpr PhaseWeights phase_weights{0, 1, 1, 2, 4, 0};

constexpr int king_near_file_span = 1;
constexpr std::uint64_t checksum_offset_basis = 14695981039346656037ULL;
constexpr std::uint64_t checksum_prime = 1099511628211ULL;
constexpr int checksum_int_bits = 32;
constexpr int bits_per_byte = 8;
constexpr std::size_t relative_rank_1 = 1;
constexpr std::size_t relative_rank_2 = 2;
constexpr std::size_t relative_rank_3 = 3;
constexpr std::size_t relative_rank_4 = 4;
constexpr std::size_t relative_rank_5 = 5;
constexpr std::size_t relative_rank_6 = 6;

struct EvalTerms {
    int middle_game{0};
    int end_game{0};
    int phase{0};
};

[[nodiscard]] constexpr int abs_value(int value) noexcept { return value < 0 ? -value : value; }

[[nodiscard]] constexpr std::size_t color_index(Color color) noexcept {
    return static_cast<std::size_t>(color);
}

[[nodiscard]] constexpr std::size_t piece_type_index(PieceType type) noexcept {
    return static_cast<std::size_t>(type);
}

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,modernize-use-designated-initializers)
inline constexpr EvaluationParameters baseline_parameters{
    .middle_game_piece_values = {100, 320, 330, 500, 900, 0},
    .end_game_piece_values = {120, 300, 320, 510, 900, 0},
    .middle_game_mobility_weights = {0, 4, 4, 2, 1, 0},
    .end_game_mobility_weights = {0, 4, 4, 3, 2, 0},
    .middle_game_passed_pawn_bonus = {0, 6, 12, 22, 38, 64, 100, 0},
    .end_game_passed_pawn_bonus = {0, 12, 24, 42, 72, 112, 170, 0},
    .middle_game_isolated_pawn_penalty = 12,
    .end_game_isolated_pawn_penalty = 10,
    .middle_game_doubled_pawn_penalty = 10,
    .end_game_doubled_pawn_penalty = 14,
    .middle_game_bishop_pair_bonus = 30,
    .end_game_bishop_pair_bonus = 40,
    .middle_game_rook_open_file_bonus = 22,
    .end_game_rook_open_file_bonus = 12,
    .middle_game_rook_semi_open_file_bonus = 12,
    .end_game_rook_semi_open_file_bonus = 8,
    .king_shield_rank_one_bonus = 7,
    .king_shield_rank_two_bonus = 3,
    .king_semi_open_file_penalty = 8,
    .king_open_file_penalty = 14,
    .king_zone_attack_penalty = 5,
};

inline constexpr auto parameter_descriptors = std::array{
    EvaluationParameterDescriptor{"material.mg.pawn", 100, 50, 200},
    EvaluationParameterDescriptor{"material.mg.knight", 320, 100, 500},
    EvaluationParameterDescriptor{"material.mg.bishop", 330, 100, 500},
    EvaluationParameterDescriptor{"material.mg.rook", 500, 250, 800},
    EvaluationParameterDescriptor{"material.mg.queen", 900, 500, 1400},
    EvaluationParameterDescriptor{"material.eg.pawn", 120, 50, 220},
    EvaluationParameterDescriptor{"material.eg.knight", 300, 100, 500},
    EvaluationParameterDescriptor{"material.eg.bishop", 320, 100, 500},
    EvaluationParameterDescriptor{"material.eg.rook", 510, 250, 800},
    EvaluationParameterDescriptor{"material.eg.queen", 900, 500, 1400},
    EvaluationParameterDescriptor{"mobility.mg.knight", 4, 0, 20},
    EvaluationParameterDescriptor{"mobility.mg.bishop", 4, 0, 20},
    EvaluationParameterDescriptor{"mobility.mg.rook", 2, 0, 20},
    EvaluationParameterDescriptor{"mobility.mg.queen", 1, 0, 20},
    EvaluationParameterDescriptor{"mobility.eg.knight", 4, 0, 20},
    EvaluationParameterDescriptor{"mobility.eg.bishop", 4, 0, 20},
    EvaluationParameterDescriptor{"mobility.eg.rook", 3, 0, 20},
    EvaluationParameterDescriptor{"mobility.eg.queen", 2, 0, 20},
    EvaluationParameterDescriptor{"passed.mg.rank1", 6, 0, 80},
    EvaluationParameterDescriptor{"passed.mg.rank2", 12, 0, 100},
    EvaluationParameterDescriptor{"passed.mg.rank3", 22, 0, 140},
    EvaluationParameterDescriptor{"passed.mg.rank4", 38, 0, 180},
    EvaluationParameterDescriptor{"passed.mg.rank5", 64, 0, 240},
    EvaluationParameterDescriptor{"passed.mg.rank6", 100, 0, 320},
    EvaluationParameterDescriptor{"passed.eg.rank1", 12, 0, 100},
    EvaluationParameterDescriptor{"passed.eg.rank2", 24, 0, 130},
    EvaluationParameterDescriptor{"passed.eg.rank3", 42, 0, 180},
    EvaluationParameterDescriptor{"passed.eg.rank4", 72, 0, 240},
    EvaluationParameterDescriptor{"passed.eg.rank5", 112, 0, 320},
    EvaluationParameterDescriptor{"passed.eg.rank6", 170, 0, 420},
    EvaluationParameterDescriptor{"pawn.mg.isolated_penalty", 12, 0, 80},
    EvaluationParameterDescriptor{"pawn.eg.isolated_penalty", 10, 0, 80},
    EvaluationParameterDescriptor{"pawn.mg.doubled_penalty", 10, 0, 80},
    EvaluationParameterDescriptor{"pawn.eg.doubled_penalty", 14, 0, 80},
    EvaluationParameterDescriptor{"bishop_pair.mg.bonus", 30, 0, 120},
    EvaluationParameterDescriptor{"bishop_pair.eg.bonus", 40, 0, 140},
    EvaluationParameterDescriptor{"rook.mg.open_file_bonus", 22, 0, 120},
    EvaluationParameterDescriptor{"rook.eg.open_file_bonus", 12, 0, 100},
    EvaluationParameterDescriptor{"rook.mg.semi_open_file_bonus", 12, 0, 80},
    EvaluationParameterDescriptor{"rook.eg.semi_open_file_bonus", 8, 0, 80},
    EvaluationParameterDescriptor{"king.shield_rank1_bonus", 7, 0, 60},
    EvaluationParameterDescriptor{"king.shield_rank2_bonus", 3, 0, 40},
    EvaluationParameterDescriptor{"king.semi_open_file_penalty", 8, 0, 80},
    EvaluationParameterDescriptor{"king.open_file_penalty", 14, 0, 120},
    EvaluationParameterDescriptor{"king.zone_attack_penalty", 5, 0, 60},
};
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,modernize-use-designated-initializers)

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] int* parameter_pointer(EvaluationParameters& parameters,
                                     std::string_view name) noexcept {
    const auto pawn = piece_type_index(PieceType::Pawn);
    const auto knight = piece_type_index(PieceType::Knight);
    const auto bishop = piece_type_index(PieceType::Bishop);
    const auto rook = piece_type_index(PieceType::Rook);
    const auto queen = piece_type_index(PieceType::Queen);

    if (name == "material.mg.pawn") {
        return &parameters.middle_game_piece_values[pawn];
    }
    if (name == "material.mg.knight") {
        return &parameters.middle_game_piece_values[knight];
    }
    if (name == "material.mg.bishop") {
        return &parameters.middle_game_piece_values[bishop];
    }
    if (name == "material.mg.rook") {
        return &parameters.middle_game_piece_values[rook];
    }
    if (name == "material.mg.queen") {
        return &parameters.middle_game_piece_values[queen];
    }
    if (name == "material.eg.pawn") {
        return &parameters.end_game_piece_values[pawn];
    }
    if (name == "material.eg.knight") {
        return &parameters.end_game_piece_values[knight];
    }
    if (name == "material.eg.bishop") {
        return &parameters.end_game_piece_values[bishop];
    }
    if (name == "material.eg.rook") {
        return &parameters.end_game_piece_values[rook];
    }
    if (name == "material.eg.queen") {
        return &parameters.end_game_piece_values[queen];
    }
    if (name == "mobility.mg.knight") {
        return &parameters.middle_game_mobility_weights[knight];
    }
    if (name == "mobility.mg.bishop") {
        return &parameters.middle_game_mobility_weights[bishop];
    }
    if (name == "mobility.mg.rook") {
        return &parameters.middle_game_mobility_weights[rook];
    }
    if (name == "mobility.mg.queen") {
        return &parameters.middle_game_mobility_weights[queen];
    }
    if (name == "mobility.eg.knight") {
        return &parameters.end_game_mobility_weights[knight];
    }
    if (name == "mobility.eg.bishop") {
        return &parameters.end_game_mobility_weights[bishop];
    }
    if (name == "mobility.eg.rook") {
        return &parameters.end_game_mobility_weights[rook];
    }
    if (name == "mobility.eg.queen") {
        return &parameters.end_game_mobility_weights[queen];
    }
    if (name == "passed.mg.rank1") {
        return &parameters.middle_game_passed_pawn_bonus[relative_rank_1];
    }
    if (name == "passed.mg.rank2") {
        return &parameters.middle_game_passed_pawn_bonus[relative_rank_2];
    }
    if (name == "passed.mg.rank3") {
        return &parameters.middle_game_passed_pawn_bonus[relative_rank_3];
    }
    if (name == "passed.mg.rank4") {
        return &parameters.middle_game_passed_pawn_bonus[relative_rank_4];
    }
    if (name == "passed.mg.rank5") {
        return &parameters.middle_game_passed_pawn_bonus[relative_rank_5];
    }
    if (name == "passed.mg.rank6") {
        return &parameters.middle_game_passed_pawn_bonus[relative_rank_6];
    }
    if (name == "passed.eg.rank1") {
        return &parameters.end_game_passed_pawn_bonus[relative_rank_1];
    }
    if (name == "passed.eg.rank2") {
        return &parameters.end_game_passed_pawn_bonus[relative_rank_2];
    }
    if (name == "passed.eg.rank3") {
        return &parameters.end_game_passed_pawn_bonus[relative_rank_3];
    }
    if (name == "passed.eg.rank4") {
        return &parameters.end_game_passed_pawn_bonus[relative_rank_4];
    }
    if (name == "passed.eg.rank5") {
        return &parameters.end_game_passed_pawn_bonus[relative_rank_5];
    }
    if (name == "passed.eg.rank6") {
        return &parameters.end_game_passed_pawn_bonus[relative_rank_6];
    }
    if (name == "pawn.mg.isolated_penalty") {
        return &parameters.middle_game_isolated_pawn_penalty;
    }
    if (name == "pawn.eg.isolated_penalty") {
        return &parameters.end_game_isolated_pawn_penalty;
    }
    if (name == "pawn.mg.doubled_penalty") {
        return &parameters.middle_game_doubled_pawn_penalty;
    }
    if (name == "pawn.eg.doubled_penalty") {
        return &parameters.end_game_doubled_pawn_penalty;
    }
    if (name == "bishop_pair.mg.bonus") {
        return &parameters.middle_game_bishop_pair_bonus;
    }
    if (name == "bishop_pair.eg.bonus") {
        return &parameters.end_game_bishop_pair_bonus;
    }
    if (name == "rook.mg.open_file_bonus") {
        return &parameters.middle_game_rook_open_file_bonus;
    }
    if (name == "rook.eg.open_file_bonus") {
        return &parameters.end_game_rook_open_file_bonus;
    }
    if (name == "rook.mg.semi_open_file_bonus") {
        return &parameters.middle_game_rook_semi_open_file_bonus;
    }
    if (name == "rook.eg.semi_open_file_bonus") {
        return &parameters.end_game_rook_semi_open_file_bonus;
    }
    if (name == "king.shield_rank1_bonus") {
        return &parameters.king_shield_rank_one_bonus;
    }
    if (name == "king.shield_rank2_bonus") {
        return &parameters.king_shield_rank_two_bonus;
    }
    if (name == "king.semi_open_file_penalty") {
        return &parameters.king_semi_open_file_penalty;
    }
    if (name == "king.open_file_penalty") {
        return &parameters.king_open_file_penalty;
    }
    if (name == "king.zone_attack_penalty") {
        return &parameters.king_zone_attack_penalty;
    }
    return nullptr;
}

[[nodiscard]] const int* parameter_pointer(const EvaluationParameters& parameters,
                                           std::string_view name) noexcept {
    return parameter_pointer(const_cast<EvaluationParameters&>(parameters), name);
}

void mix_checksum_byte(std::uint64_t& hash, std::uint8_t value) noexcept {
    hash ^= value;
    hash *= checksum_prime;
}

void mix_checksum_int(std::uint64_t& hash, int value) noexcept {
    const auto unsigned_value = static_cast<std::uint32_t>(value);
    for (int shift = 0; shift < checksum_int_bits; shift += bits_per_byte) {
        mix_checksum_byte(hash, static_cast<std::uint8_t>(unsigned_value >> shift));
    }
}

void mix_checksum_values(std::uint64_t& hash, std::span<const int> values) noexcept {
    for (const auto value : values) {
        mix_checksum_int(hash, value);
    }
}

[[nodiscard]] constexpr Square square(int file_index, int rank_index) noexcept {
    return Square::from_file_rank(File::from_index(file_index), Rank::from_index(rank_index));
}

[[nodiscard]] constexpr Square mirror_for_black(Square square) noexcept {
    return Square::from_index(square.index() ^ mirror_vertical_mask);
}

[[nodiscard]] constexpr int file_centrality(Square square) noexcept {
    const auto file = square.file().index();
    return file_centrality_base -
           (doubled_coordinate_scale *
            abs_value((doubled_coordinate_scale * file) - scaled_board_center));
}

[[nodiscard]] constexpr int center_score(Square square) noexcept {
    const auto file = square.file().index();
    const auto rank = square.rank().index();
    const auto distance = abs_value((doubled_coordinate_scale * file) - scaled_board_center) +
                          abs_value((doubled_coordinate_scale * rank) - scaled_board_center);
    return center_score_base - (doubled_coordinate_scale * distance);
}

[[nodiscard]] consteval FileMasks make_file_masks() {
    FileMasks masks{};
    for (int file = 0; file < File::count; ++file) {
        for (int rank = 0; rank < Rank::count; ++rank) {
            masks[static_cast<std::size_t>(file)] |= square_bb(square(file, rank));
        }
    }
    return masks;
}

inline constexpr auto file_masks = make_file_masks();

[[nodiscard]] constexpr Bitboard file_mask(int file_index) noexcept {
    if (file_index < 0 || file_index >= File::count) {
        return empty_bitboard;
    }
    return file_masks[static_cast<std::size_t>(file_index)];
}

[[nodiscard]] constexpr Bitboard adjacent_file_mask(int file_index) noexcept {
    Bitboard mask = empty_bitboard;
    if (file_index > 0) {
        mask |= file_mask(file_index - 1);
    }
    if (file_index + 1 < File::count) {
        mask |= file_mask(file_index + 1);
    }
    return mask;
}

[[nodiscard]] constexpr int relative_rank(Color color, Square square_value) noexcept {
    const auto rank = square_value.rank().index();
    return color == Color::White ? rank : (Rank::count - 1 - rank);
}

[[nodiscard]] constexpr int forward_delta(Color color) noexcept {
    return color == Color::White ? 1 : -1;
}

[[nodiscard]] constexpr Bitboard passed_pawn_span(Color color, Square square_value) noexcept {
    const auto file = square_value.file().index();
    auto rank = square_value.rank().index() + forward_delta(color);
    Bitboard span = empty_bitboard;
    while (rank >= 0 && rank < Rank::count) {
        for (int current_file = file - 1; current_file <= file + 1; ++current_file) {
            span |= square_bb(square(current_file, rank));
        }
        rank += forward_delta(color);
    }
    return span;
}

[[nodiscard]] constexpr int psqt_value(PieceType type, Square square, EvalPhase phase) noexcept {
    const auto rank = square.rank().index();
    const auto center = center_score(square);
    const auto file_center = file_centrality(square);

    switch (type) {
    case PieceType::Pawn:
        return phase == EvalPhase::MiddleGame
                   ? (pawn_middle_game_rank_bonus * rank) + file_center
                   : (pawn_end_game_rank_bonus * rank) + (file_center / doubled_coordinate_scale);
    case PieceType::Knight:
        return center * (phase == EvalPhase::MiddleGame ? knight_middle_game_center_weight
                                                        : knight_end_game_center_weight);
    case PieceType::Bishop:
        return (bishop_center_weight * center) + (phase == EvalPhase::MiddleGame ? rank : 0);
    case PieceType::Rook:
        return (phase == EvalPhase::MiddleGame ? rook_middle_game_rank_bonus * rank
                                               : rook_end_game_rank_bonus * rank) +
               (square.file().index() == 0 || square.file().index() == File::count - 1
                    ? rook_edge_file_penalty
                    : 0);
    case PieceType::Queen:
        return center + (phase == EvalPhase::MiddleGame ? 0 : queen_end_game_rank_bonus * rank);
    case PieceType::King:
        return phase == EvalPhase::MiddleGame
                   ? (king_middle_game_center_weight * center) +
                         (king_middle_game_rank_penalty * rank)
                   : (king_end_game_center_weight * center) + (king_end_game_rank_bonus * rank);
    case PieceType::None:
        break;
    }
    return 0;
}

template <EvalPhase phase> [[nodiscard]] consteval PieceSquareTable make_piece_square_table() {
    PieceSquareTable table{};
    for (int piece_type = 0; piece_type < Piece::type_count; ++piece_type) {
        for (int square_index = 0; square_index < Square::count; ++square_index) {
            table[static_cast<std::size_t>(piece_type)][static_cast<std::size_t>(square_index)] =
                psqt_value(static_cast<PieceType>(piece_type), Square::from_index(square_index),
                           phase);
        }
    }
    return table;
}

inline constexpr auto middle_game_psqt = make_piece_square_table<EvalPhase::MiddleGame>();
inline constexpr auto end_game_psqt = make_piece_square_table<EvalPhase::EndGame>();

[[nodiscard]] constexpr Square square_from_piece_perspective(Color color, Square square) noexcept {
    return color == Color::Black ? mirror_for_black(square) : square;
}

void add_material_and_psqt(EvalTerms& terms, PieceType type, Color color, Square square,
                           const EvaluationParameters& parameters) noexcept {
    const auto type_index = piece_type_index(type);
    const auto table_square = square_from_piece_perspective(color, square);
    const auto square_index = static_cast<std::size_t>(table_square.index());

    terms.middle_game += parameters.middle_game_piece_values[type_index] +
                         middle_game_psqt[type_index][square_index];
    terms.end_game +=
        parameters.end_game_piece_values[type_index] + end_game_psqt[type_index][square_index];
}

void add_mobility(EvalTerms& terms, PieceType type, int attack_count,
                  const EvaluationParameters& parameters) noexcept {
    const auto type_index = piece_type_index(type);
    terms.middle_game += parameters.middle_game_mobility_weights[type_index] * attack_count;
    terms.end_game += parameters.end_game_mobility_weights[type_index] * attack_count;
}

[[nodiscard]] Bitboard piece_attacks(PieceType type, Color color, Square from,
                                     Bitboard occupancy) noexcept {
    switch (type) {
    case PieceType::Pawn:
        return pawn_attacks(color, from);
    case PieceType::Knight:
        return knight_attacks(from);
    case PieceType::Bishop:
        return bishop_attacks(from, occupancy);
    case PieceType::Rook:
        return rook_attacks(from, occupancy);
    case PieceType::Queen:
        return queen_attacks(from, occupancy);
    case PieceType::King:
        return king_attacks(from);
    case PieceType::None:
        break;
    }
    return empty_bitboard;
}

[[nodiscard]] Bitboard attacked_squares(const Position& position, Color color) noexcept {
    const auto occupancy = position.occupancy();
    Bitboard attacks = empty_bitboard;
    for (int piece_type_index_value = 0; piece_type_index_value < Piece::type_count;
         ++piece_type_index_value) {
        const auto type = static_cast<PieceType>(piece_type_index_value);
        auto pieces = position.pieces(color, type);
        while (pieces != empty_bitboard) {
            attacks |= piece_attacks(type, color, pop_lsb(pieces), occupancy);
        }
    }
    return attacks;
}

void add_mobility_terms(const Position& position, std::array<EvalTerms, 2>& terms,
                        const EvaluationParameters& parameters) noexcept {
    const auto occupancy = position.occupancy();
    for (const auto color : {Color::White, Color::Black}) {
        const auto own_occupancy = position.occupancy(color);
        for (const auto type :
             {PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen}) {
            auto pieces = position.pieces(color, type);
            while (pieces != empty_bitboard) {
                const auto from = pop_lsb(pieces);
                const auto attacks = piece_attacks(type, color, from, occupancy) & ~own_occupancy;
                add_mobility(terms[color_index(color)], type, popcount(attacks), parameters);
            }
        }
    }
}

void add_pawn_structure_terms(const Position& position, std::array<EvalTerms, 2>& terms,
                              const EvaluationParameters& parameters) noexcept {
    for (const auto color : {Color::White, Color::Black}) {
        const auto enemy = opposite(color);
        const auto friendly_pawns = position.pieces(color, PieceType::Pawn);
        const auto enemy_pawns = position.pieces(enemy, PieceType::Pawn);
        auto pawns = friendly_pawns;
        while (pawns != empty_bitboard) {
            const auto pawn_square = pop_lsb(pawns);
            const auto file = pawn_square.file().index();
            const auto rank = static_cast<std::size_t>(relative_rank(color, pawn_square));

            if ((friendly_pawns & adjacent_file_mask(file)) == empty_bitboard) {
                terms[color_index(color)].middle_game -=
                    parameters.middle_game_isolated_pawn_penalty;
                terms[color_index(color)].end_game -= parameters.end_game_isolated_pawn_penalty;
            }

            if (popcount(friendly_pawns & file_mask(file)) > 1) {
                terms[color_index(color)].middle_game -=
                    parameters.middle_game_doubled_pawn_penalty;
                terms[color_index(color)].end_game -= parameters.end_game_doubled_pawn_penalty;
            }

            if ((enemy_pawns & passed_pawn_span(color, pawn_square)) == empty_bitboard) {
                terms[color_index(color)].middle_game +=
                    parameters.middle_game_passed_pawn_bonus[rank];
                terms[color_index(color)].end_game += parameters.end_game_passed_pawn_bonus[rank];
            }
        }
    }
}

void add_bishop_pair_terms(const Position& position, std::array<EvalTerms, 2>& terms,
                           const EvaluationParameters& parameters) noexcept {
    for (const auto color : {Color::White, Color::Black}) {
        if (popcount(position.pieces(color, PieceType::Bishop)) >= 2) {
            terms[color_index(color)].middle_game += parameters.middle_game_bishop_pair_bonus;
            terms[color_index(color)].end_game += parameters.end_game_bishop_pair_bonus;
        }
    }
}

void add_rook_file_terms(const Position& position, std::array<EvalTerms, 2>& terms,
                         const EvaluationParameters& parameters) noexcept {
    for (const auto color : {Color::White, Color::Black}) {
        const auto friendly_pawns = position.pieces(color, PieceType::Pawn);
        const auto enemy_pawns = position.pieces(opposite(color), PieceType::Pawn);
        auto rooks = position.pieces(color, PieceType::Rook);
        while (rooks != empty_bitboard) {
            const auto rook_square = pop_lsb(rooks);
            const auto file = rook_square.file().index();
            const auto file_bitboard = file_mask(file);
            if ((friendly_pawns & file_bitboard) != empty_bitboard) {
                continue;
            }

            if ((enemy_pawns & file_bitboard) == empty_bitboard) {
                terms[color_index(color)].middle_game +=
                    parameters.middle_game_rook_open_file_bonus;
                terms[color_index(color)].end_game += parameters.end_game_rook_open_file_bonus;
            } else {
                terms[color_index(color)].middle_game +=
                    parameters.middle_game_rook_semi_open_file_bonus;
                terms[color_index(color)].end_game += parameters.end_game_rook_semi_open_file_bonus;
            }
        }
    }
}

[[nodiscard]] Square king_square(const Position& position, Color color) noexcept {
    return lsb(position.pieces(color, PieceType::King));
}

void add_king_shield(EvalTerms& terms, const Position& position, Color color, Square king,
                     const EvaluationParameters& parameters) noexcept {
    const auto friendly_pawns = position.pieces(color, PieceType::Pawn);
    const auto direction = forward_delta(color);
    const auto king_file = king.file().index();
    const auto king_rank = king.rank().index();

    for (int file = king_file - king_near_file_span; file <= king_file + king_near_file_span;
         ++file) {
        const auto near_square = square(file, king_rank + direction);
        if (has_square(friendly_pawns, near_square)) {
            terms.middle_game += parameters.king_shield_rank_one_bonus;
        }

        const auto far_square = square(file, king_rank + (2 * direction));
        if (has_square(friendly_pawns, far_square)) {
            terms.middle_game += parameters.king_shield_rank_two_bonus;
        }
    }
}

void add_king_file_safety(EvalTerms& terms, const Position& position, Color color, Square king,
                          const EvaluationParameters& parameters) noexcept {
    const auto friendly_pawns = position.pieces(color, PieceType::Pawn);
    const auto enemy_pawns = position.pieces(opposite(color), PieceType::Pawn);
    const auto king_file = king.file().index();

    for (int file = king_file - king_near_file_span; file <= king_file + king_near_file_span;
         ++file) {
        const auto file_bitboard = file_mask(file);
        if (file_bitboard == empty_bitboard || (friendly_pawns & file_bitboard) != empty_bitboard) {
            continue;
        }

        terms.middle_game -= (enemy_pawns & file_bitboard) == empty_bitboard
                                 ? parameters.king_open_file_penalty
                                 : parameters.king_semi_open_file_penalty;
    }
}

void add_king_attack_pressure(EvalTerms& terms, const Position& position, Color color, Square king,
                              const EvaluationParameters& parameters) noexcept {
    const auto enemy_attacks = attacked_squares(position, opposite(color));
    const auto king_zone = king_attacks(king) | square_bb(king);
    terms.middle_game -= parameters.king_zone_attack_penalty * popcount(enemy_attacks & king_zone);
}

void add_king_safety_terms(const Position& position, std::array<EvalTerms, 2>& terms,
                           const EvaluationParameters& parameters) noexcept {
    for (const auto color : {Color::White, Color::Black}) {
        const auto king = king_square(position, color);
        if (!king.is_valid()) {
            continue;
        }

        auto& color_terms = terms[color_index(color)];
        add_king_shield(color_terms, position, color, king, parameters);
        add_king_file_safety(color_terms, position, color, king, parameters);
        add_king_attack_pressure(color_terms, position, color, king, parameters);
    }
}

} // namespace

EvaluationParameters default_evaluation_parameters() noexcept { return baseline_parameters; }

std::span<const EvaluationParameterDescriptor> evaluation_parameter_descriptors() noexcept {
    return parameter_descriptors;
}

int evaluation_parameter_value(const EvaluationParameters& parameters,
                               std::string_view name) noexcept {
    const auto* value = parameter_pointer(parameters, name);
    return value == nullptr ? 0 : *value;
}

bool set_evaluation_parameter(EvaluationParameters& parameters, std::string_view name,
                              int value) noexcept {
    const auto descriptor = std::ranges::find_if(
        parameter_descriptors, [name](const EvaluationParameterDescriptor& current) noexcept {
            return current.name == name;
        });
    if (descriptor == parameter_descriptors.end() || value < descriptor->min_value ||
        value > descriptor->max_value) {
        return false;
    }

    auto* parameter = parameter_pointer(parameters, name);
    if (parameter == nullptr) {
        return false;
    }
    *parameter = value;
    return true;
}

std::uint64_t evaluation_parameters_checksum(const EvaluationParameters& parameters) noexcept {
    auto hash = checksum_offset_basis;
    mix_checksum_values(hash, parameters.middle_game_piece_values);
    mix_checksum_values(hash, parameters.end_game_piece_values);
    mix_checksum_values(hash, parameters.middle_game_mobility_weights);
    mix_checksum_values(hash, parameters.end_game_mobility_weights);
    mix_checksum_values(hash, parameters.middle_game_passed_pawn_bonus);
    mix_checksum_values(hash, parameters.end_game_passed_pawn_bonus);
    mix_checksum_int(hash, parameters.middle_game_isolated_pawn_penalty);
    mix_checksum_int(hash, parameters.end_game_isolated_pawn_penalty);
    mix_checksum_int(hash, parameters.middle_game_doubled_pawn_penalty);
    mix_checksum_int(hash, parameters.end_game_doubled_pawn_penalty);
    mix_checksum_int(hash, parameters.middle_game_bishop_pair_bonus);
    mix_checksum_int(hash, parameters.end_game_bishop_pair_bonus);
    mix_checksum_int(hash, parameters.middle_game_rook_open_file_bonus);
    mix_checksum_int(hash, parameters.end_game_rook_open_file_bonus);
    mix_checksum_int(hash, parameters.middle_game_rook_semi_open_file_bonus);
    mix_checksum_int(hash, parameters.end_game_rook_semi_open_file_bonus);
    mix_checksum_int(hash, parameters.king_shield_rank_one_bonus);
    mix_checksum_int(hash, parameters.king_shield_rank_two_bonus);
    mix_checksum_int(hash, parameters.king_semi_open_file_penalty);
    mix_checksum_int(hash, parameters.king_open_file_penalty);
    mix_checksum_int(hash, parameters.king_zone_attack_penalty);
    return hash;
}

Score evaluate(const Position& position) noexcept {
    return evaluate(position, EvaluationOptions{});
}

Score evaluate(const Position& position, EvaluationOptions options) noexcept {
    return evaluate(position, options, default_evaluation_parameters());
}

Score evaluate(const Position& position, EvaluationOptions options,
               const EvaluationParameters& parameters) noexcept {
    std::array<EvalTerms, 2> terms{};

    for (const auto color : {Color::White, Color::Black}) {
        for (int piece_type_index_value = 0; piece_type_index_value < Piece::type_count;
             ++piece_type_index_value) {
            const auto type = static_cast<PieceType>(piece_type_index_value);
            auto pieces = position.pieces(color, type);
            while (pieces != empty_bitboard) {
                const auto square_value = pop_lsb(pieces);
                terms[color_index(color)].phase += phase_weights[piece_type_index(type)];
                if (options.material_and_psqt) {
                    add_material_and_psqt(terms[color_index(color)], type, color, square_value,
                                          parameters);
                }
            }
        }
    }

    if (options.mobility) {
        add_mobility_terms(position, terms, parameters);
    }
    if (options.pawn_structure) {
        add_pawn_structure_terms(position, terms, parameters);
    }
    if (options.king_safety) {
        add_king_safety_terms(position, terms, parameters);
    }
    if (options.bishop_pair) {
        add_bishop_pair_terms(position, terms, parameters);
    }
    if (options.rook_files) {
        add_rook_file_terms(position, terms, parameters);
    }

    const auto middle_game_score =
        terms[color_index(Color::White)].middle_game - terms[color_index(Color::Black)].middle_game;
    const auto end_game_score =
        terms[color_index(Color::White)].end_game - terms[color_index(Color::Black)].end_game;
    const auto phase =
        std::min(terms[color_index(Color::White)].phase + terms[color_index(Color::Black)].phase,
                 max_game_phase);
    auto score = ((middle_game_score * phase) + (end_game_score * (max_game_phase - phase))) /
                 max_game_phase;

    if (position.side_to_move() == Color::Black) {
        score = -score;
    }
    return Score::centipawns(score);
}

} // namespace eng
