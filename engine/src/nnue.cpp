#include <eng/nnue.hpp>

#include <eng/bitboard.hpp>
#include <eng/eval.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <utility>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace eng {
namespace {

constexpr std::uint32_t squares_per_piece = 64;
constexpr std::uint32_t black_piece_offset = 6;
constexpr int first_rank_index = 0;
constexpr int eighth_rank_index = 7;
constexpr int queen_side_rook_file_index = 0;
constexpr int queen_side_rook_to_file_index = 3;
constexpr int king_side_rook_to_file_index = 5;
constexpr int king_side_king_to_file_index = 6;
constexpr int king_side_rook_file_index = 7;
constexpr int black_perspective_square_xor = 56;
constexpr std::int64_t nnue_output_scale =
    static_cast<std::int64_t>(nnue_network_qa) * static_cast<std::int64_t>(nnue_network_qb);

#if defined(__ARM_NEON) && defined(__aarch64__)
constexpr bool neon_backend_compiled = true;
#else
constexpr bool neon_backend_compiled = false;
#endif

struct RookMove {
    Square from;
    Square to;
};

[[nodiscard]] constexpr std::uint32_t piece_type_index(PieceType type) noexcept {
    return static_cast<std::uint32_t>(type);
}

[[nodiscard]] int divide_round_nearest(std::int64_t value, std::int64_t divisor) noexcept {
    if (value >= 0) {
        return static_cast<int>((value + (divisor / 2)) / divisor);
    }
    return -static_cast<int>(((-value) + (divisor / 2)) / divisor);
}

[[nodiscard]] std::int64_t activated_value(std::int32_t accumulator_value,
                                           NnueNetworkActivation activation) noexcept {
    const auto clipped = std::clamp<std::int64_t>(accumulator_value, 0, nnue_network_qa);
    if (activation == NnueNetworkActivation::SCReLU) {
        return ((clipped * clipped) + (nnue_network_qa / 2)) / nnue_network_qa;
    }
    return clipped;
}

void apply_feature_delta_scalar(std::span<std::int32_t> values,
                                std::span<const std::int16_t> weights, int sign) noexcept {
    const auto signed_delta = static_cast<std::int32_t>(sign);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] += signed_delta * static_cast<std::int32_t>(weights[index]);
    }
}

[[nodiscard]] std::int64_t dot_activated_weights_scalar(std::span<const std::int32_t> values,
                                                        std::span<const std::int16_t> weights,
                                                        NnueNetworkActivation activation) noexcept {
    std::int64_t sum = 0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        sum +=
            activated_value(values[index], activation) * static_cast<std::int64_t>(weights[index]);
    }
    return sum;
}

#if defined(__ARM_NEON) && defined(__aarch64__)
[[nodiscard]] int32x4_t screlu_divide_by_qa_neon(int32x4_t numerator) noexcept {
    // Exact floor(n / 255) for n in the SCReLU numerator range [0, 65152].
    return vshrq_n_s32(vmulq_n_s32(numerator, 32897), 23);
}

[[nodiscard]] int32x4_t activated_values_neon(int32x4_t values,
                                              NnueNetworkActivation activation) noexcept {
    const auto zero = vdupq_n_s32(0);
    const auto quantization_a = vdupq_n_s32(static_cast<std::int32_t>(nnue_network_qa));
    auto clipped = vminq_s32(vmaxq_s32(values, zero), quantization_a);
    if (activation == NnueNetworkActivation::SCReLU) {
        clipped = vaddq_s32(vmulq_s32(clipped, clipped),
                            vdupq_n_s32(static_cast<std::int32_t>(nnue_network_qa / 2)));
        return screlu_divide_by_qa_neon(clipped);
    }
    return clipped;
}

void apply_feature_delta_neon(std::span<std::int32_t> values, std::span<const std::int16_t> weights,
                              int sign) noexcept {
    constexpr std::size_t lanes = 4;
    const auto positive_delta = sign >= 0;
    std::size_t index = 0;
    for (; index + lanes <= values.size(); index += lanes) {
        const auto current = vld1q_s32(values.data() + index);
        const auto delta = vmovl_s16(vld1_s16(weights.data() + index));
        const auto updated = positive_delta ? vaddq_s32(current, delta) : vsubq_s32(current, delta);
        vst1q_s32(values.data() + index, updated);
    }
    for (; index < values.size(); ++index) {
        values[index] += (positive_delta ? 1 : -1) * static_cast<std::int32_t>(weights[index]);
    }
}

[[nodiscard]] std::int64_t dot_activated_weights_neon(std::span<const std::int32_t> values,
                                                      std::span<const std::int16_t> weights,
                                                      NnueNetworkActivation activation) noexcept {
    constexpr std::size_t lanes = 4;
    auto sum = vdupq_n_s64(0);
    std::size_t index = 0;
    for (; index + lanes <= values.size(); index += lanes) {
        const auto activated = activated_values_neon(vld1q_s32(values.data() + index), activation);
        const auto weight = vmovl_s16(vld1_s16(weights.data() + index));
        const auto product = vmulq_s32(activated, weight);
        sum = vaddq_s64(sum, vmovl_s32(vget_low_s32(product)));
        sum = vaddq_s64(sum, vmovl_s32(vget_high_s32(product)));
    }

    auto total = vgetq_lane_s64(sum, 0) + vgetq_lane_s64(sum, 1);
    for (; index < values.size(); ++index) {
        total +=
            activated_value(values[index], activation) * static_cast<std::int64_t>(weights[index]);
    }
    return total;
}
#endif

[[nodiscard]] RookMove castling_rook_move(Color color, Square king_to) noexcept {
    const auto rank = color == Color::White ? Rank::from_index(first_rank_index)
                                            : Rank::from_index(eighth_rank_index);
    const auto king_side = king_to.file().index() == king_side_king_to_file_index;
    return RookMove{
        .from = Square::from_file_rank(
            File::from_index(king_side ? king_side_rook_file_index : queen_side_rook_file_index),
            rank),
        .to = Square::from_file_rank(File::from_index(king_side ? king_side_rook_to_file_index
                                                                : queen_side_rook_to_file_index),
                                     rank),
    };
}

} // namespace

bool nnue_neon_available() noexcept { return neon_backend_compiled; }

NnueInferenceBackend resolve_nnue_inference_backend(NnueInferenceBackend backend) noexcept {
    if (backend == NnueInferenceBackend::Auto) {
        return nnue_neon_available() ? NnueInferenceBackend::Neon : NnueInferenceBackend::Scalar;
    }
    if (backend == NnueInferenceBackend::Neon && !nnue_neon_available()) {
        return NnueInferenceBackend::Scalar;
    }
    if (backend == NnueInferenceBackend::Scalar || backend == NnueInferenceBackend::Neon) {
        return backend;
    }
    return NnueInferenceBackend::Scalar;
}

std::string_view nnue_inference_backend_name(NnueInferenceBackend backend) noexcept {
    switch (backend) {
    case NnueInferenceBackend::Auto:
        return "auto";
    case NnueInferenceBackend::Scalar:
        return "scalar";
    case NnueInferenceBackend::Neon:
        return "neon";
    }
    return "unknown";
}

NnueAccumulator::NnueAccumulator(std::uint32_t hidden_size) { reset(hidden_size); }

void NnueAccumulator::reset(std::uint32_t hidden_size) {
    white_values_.assign(hidden_size, 0);
    black_values_.assign(hidden_size, 0);
}

bool NnueAccumulator::copy_from(const NnueAccumulator& other) noexcept {
    if (hidden_size() != other.hidden_size()) {
        return false;
    }
    std::ranges::copy(other.white_values_, white_values_.begin());
    std::ranges::copy(other.black_values_, black_values_.begin());
    return true;
}

std::uint32_t NnueAccumulator::hidden_size() const noexcept {
    return static_cast<std::uint32_t>(white_values_.size());
}

std::span<const std::int32_t> NnueAccumulator::white_values() const noexcept {
    return white_values_;
}

std::span<const std::int32_t> NnueAccumulator::black_values() const noexcept {
    return black_values_;
}

std::expected<NnueNetwork, NnueNetworkError> NnueNetwork::load(const std::filesystem::path& path) {
    auto file = read_nnue_network_file(path);
    if (!file.has_value()) {
        return std::unexpected(file.error());
    }

    NnueNetwork network;
    network.info_ = std::move(file->info);
    network.tensors_ = std::move(file->tensors);
    return network;
}

int NnueNetwork::evaluate_white_relative(const NnueAccumulator& accumulator,
                                         NnueInferenceBackend backend) const noexcept {
    if (!is_loaded() || accumulator.hidden_size() != hidden_size()) {
        return 0;
    }

    const auto hidden = static_cast<std::size_t>(hidden_size());
    std::int64_t raw = tensors_.output_bias;
    const auto white_values = std::span<const std::int32_t>(accumulator.white_values_);
    const auto black_values = std::span<const std::int32_t>(accumulator.black_values_);
    const auto white_output_weights =
        std::span<const std::int16_t>(tensors_.output_weights.data(), hidden);
    const auto black_output_weights =
        std::span<const std::int16_t>(tensors_.output_weights.data() + hidden, hidden);

    const auto resolved_backend = resolve_nnue_inference_backend(backend);
#if defined(__ARM_NEON) && defined(__aarch64__)
    if (resolved_backend == NnueInferenceBackend::Neon) {
        raw += dot_activated_weights_neon(white_values, white_output_weights, activation());
        raw += dot_activated_weights_neon(black_values, black_output_weights, activation());
        return divide_round_nearest(raw, nnue_output_scale);
    }
#else
    (void)resolved_backend;
#endif

    raw += dot_activated_weights_scalar(white_values, white_output_weights, activation());
    raw += dot_activated_weights_scalar(black_values, black_output_weights, activation());
    return divide_round_nearest(raw, nnue_output_scale);
}

int NnueNetwork::evaluate_relative(const NnueAccumulator& accumulator, Color side_to_move,
                                   NnueInferenceBackend backend) const noexcept {
    if (!is_loaded() || accumulator.hidden_size() != hidden_size()) {
        return 0;
    }
    if (!info_.side_to_move_ordering) {
        auto score = evaluate_white_relative(accumulator, backend);
        return side_to_move == Color::Black ? -score : score;
    }

    const auto hidden = static_cast<std::size_t>(hidden_size());
    std::int64_t raw = tensors_.output_bias;
    const auto white_values = std::span<const std::int32_t>(accumulator.white_values_);
    const auto black_values = std::span<const std::int32_t>(accumulator.black_values_);
    const auto first_values = side_to_move == Color::Black ? black_values : white_values;
    const auto second_values = side_to_move == Color::Black ? white_values : black_values;
    const auto first_output_weights =
        std::span<const std::int16_t>(tensors_.output_weights.data(), hidden);
    const auto second_output_weights =
        std::span<const std::int16_t>(tensors_.output_weights.data() + hidden, hidden);

    const auto resolved_backend = resolve_nnue_inference_backend(backend);
#if defined(__ARM_NEON) && defined(__aarch64__)
    if (resolved_backend == NnueInferenceBackend::Neon) {
        raw += dot_activated_weights_neon(first_values, first_output_weights, activation());
        raw += dot_activated_weights_neon(second_values, second_output_weights, activation());
        return divide_round_nearest(raw, nnue_output_scale);
    }
#else
    (void)resolved_backend;
#endif

    raw += dot_activated_weights_scalar(first_values, first_output_weights, activation());
    raw += dot_activated_weights_scalar(second_values, second_output_weights, activation());
    return divide_round_nearest(raw, nnue_output_scale);
}

bool NnueNetwork::add_feature_delta(NnueAccumulator& accumulator, Piece piece, Square square,
                                    int sign, NnueInferenceBackend backend) const noexcept {
    if (!piece.is_valid() || !square.is_valid() || accumulator.hidden_size() != hidden_size()) {
        return false;
    }

    const auto hidden = static_cast<std::size_t>(hidden_size());
    const auto white_feature = nnue_feature_index(Color::White, piece, square);
    const auto black_feature = nnue_feature_index(Color::Black, piece, square);
    if (white_feature == nnue_invalid_feature_index ||
        black_feature == nnue_invalid_feature_index) {
        return false;
    }

    const auto white_offset = static_cast<std::size_t>(white_feature) * hidden;
    const auto black_offset = static_cast<std::size_t>(black_feature) * hidden;
    const auto white_values = std::span<std::int32_t>(accumulator.white_values_);
    const auto black_values = std::span<std::int32_t>(accumulator.black_values_);
    const auto white_weights =
        std::span<const std::int16_t>(tensors_.white_feature_weights.data() + white_offset, hidden);
    const auto black_weights =
        std::span<const std::int16_t>(tensors_.black_feature_weights.data() + black_offset, hidden);

    const auto resolved_backend = resolve_nnue_inference_backend(backend);
#if defined(__ARM_NEON) && defined(__aarch64__)
    if (resolved_backend == NnueInferenceBackend::Neon) {
        apply_feature_delta_neon(white_values, white_weights, sign);
        apply_feature_delta_neon(black_values, black_weights, sign);
        return true;
    }
#else
    (void)resolved_backend;
#endif

    apply_feature_delta_scalar(white_values, white_weights, sign);
    apply_feature_delta_scalar(black_values, black_weights, sign);
    return true;
}

bool NnueNetwork::move_feature_delta(NnueAccumulator& accumulator, Piece piece, Square from,
                                     Square to_square,
                                     NnueInferenceBackend backend) const noexcept {
    return add_feature_delta(accumulator, piece, from, -1, backend) &&
           add_feature_delta(accumulator, piece, to_square, 1, backend);
}

std::uint32_t nnue_feature_index(Color perspective, Piece piece, Square square) noexcept {
    if (!is_valid_color(perspective) || !piece.is_valid() || !square.is_valid()) {
        return nnue_invalid_feature_index;
    }

    const auto piece_offset = piece.color() == perspective ? 0U : black_piece_offset;
    const auto perspective_square = perspective == Color::Black
                                        ? square.index() ^ black_perspective_square_xor
                                        : square.index();
    return ((piece_offset + piece_type_index(piece.type())) * squares_per_piece) +
           static_cast<std::uint32_t>(perspective_square);
}

NnueAccumulator refresh_nnue_accumulator(const Position& position, const NnueNetwork& network,
                                         NnueInferenceBackend backend) {
    NnueAccumulator accumulator(network.hidden_size());
    if (network.info_.has_feature_bias) {
        std::ranges::copy(network.tensors_.white_feature_bias, accumulator.white_values_.begin());
        std::ranges::copy(network.tensors_.black_feature_bias, accumulator.black_values_.begin());
    }
    for (const auto color : std::array{Color::White, Color::Black}) {
        for (int piece_type = 0; piece_type < Piece::type_count; ++piece_type) {
            const auto type = static_cast<PieceType>(piece_type);
            const auto piece = Piece::make(color, type);
            auto pieces = position.pieces(color, type);
            while (pieces != empty_bitboard) {
                const auto square = pop_lsb(pieces);
                static_cast<void>(
                    network.add_feature_delta(accumulator, piece, square, 1, backend));
            }
        }
    }
    return accumulator;
}

bool update_nnue_accumulator_after_make(NnueAccumulator& accumulator, const NnueNetwork& network,
                                        Move move, const StateInfo& state,
                                        NnueInferenceBackend backend) noexcept {
    if (!network.is_loaded() || accumulator.hidden_size() != network.hidden_size()) {
        return false;
    }
    if (move.is_null() || !state.moved_piece.is_valid()) {
        return true;
    }

    const auto moved_piece = state.moved_piece;
    const auto moving_color = moved_piece.color();
    const auto from = move.from();
    const auto to_square = move.to();

    if (move.is_castling()) {
        const auto rook_move = castling_rook_move(moving_color, to_square);
        const auto rook = Piece::make(moving_color, PieceType::Rook);
        return network.move_feature_delta(accumulator, moved_piece, from, to_square, backend) &&
               network.move_feature_delta(accumulator, rook, rook_move.from, rook_move.to, backend);
    }

    if (move.is_promotion()) {
        if (!network.add_feature_delta(accumulator, moved_piece, from, -1, backend)) {
            return false;
        }
        if (state.captured_piece.is_valid() &&
            !network.add_feature_delta(accumulator, state.captured_piece, state.captured_square, -1,
                                       backend)) {
            return false;
        }
        return network.add_feature_delta(
            accumulator, Piece::make(moving_color, move.promotion_piece()), to_square, 1, backend);
    }

    if (state.captured_piece.is_valid() &&
        !network.add_feature_delta(accumulator, state.captured_piece, state.captured_square, -1,
                                   backend)) {
        return false;
    }

    return network.move_feature_delta(accumulator, moved_piece, from, to_square, backend);
}

Score evaluate_nnue(const Position& position, const NnueNetwork& network,
                    const NnueAccumulator& accumulator, NnueInferenceBackend backend) noexcept {
    auto score = network.evaluate_relative(accumulator, position.side_to_move(), backend);
    return Score::centipawns(score);
}

Score evaluate_nnue(const Position& position, const NnueNetwork& network,
                    NnueInferenceBackend backend) {
    const auto accumulator = refresh_nnue_accumulator(position, network, backend);
    return evaluate_nnue(position, network, accumulator, backend);
}

Score NnueEvaluator::evaluate(const Position& position) const {
    if (network_ == nullptr || !network_->is_loaded()) {
        return eng::evaluate(position);
    }
    return evaluate_nnue(position, *network_, backend_);
}

} // namespace eng
