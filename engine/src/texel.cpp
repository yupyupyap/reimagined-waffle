#include <eng/texel.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace eng {
namespace {

constexpr std::uint64_t checksum_offset_basis = 14695981039346656037ULL;
constexpr std::uint64_t checksum_prime = 1099511628211ULL;
constexpr int bits_per_byte = 8;
constexpr int checksum_int_bits = 32;
constexpr int checksum_u64_bits = 64;
constexpr int double_checksum_scale = 1'000'000;
constexpr double loss_epsilon = 1.0e-12;

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

void mix_checksum_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
    for (int shift = 0; shift < checksum_u64_bits; shift += bits_per_byte) {
        mix_checksum_byte(hash, static_cast<std::uint8_t>(value >> shift));
    }
}

[[nodiscard]] int normalized_scale(int scale) noexcept { return std::max(1, scale); }

[[nodiscard]] double sigmoid_score(int score, int scale) noexcept {
    return 1.0 / (1.0 + std::exp(-static_cast<double>(score) / normalized_scale(scale)));
}

[[nodiscard]] int white_relative_eval(const Position& position,
                                      const EvaluationParameters& parameters) noexcept {
    const auto score = evaluate(position, EvaluationOptions{}, parameters).value();
    return position.side_to_move() == Color::Black ? -score : score;
}

[[nodiscard]] int clamped_scale_step(TexelTuneConfig config) noexcept {
    return std::max(1, config.scale_step);
}

[[nodiscard]] bool improved(double candidate, double current) noexcept {
    return candidate + loss_epsilon < current;
}

} // namespace

double texel_loss(std::span<const TexelSample> samples, const EvaluationParameters& parameters,
                  int scale) noexcept {
    if (samples.empty()) {
        return 0.0;
    }

    double weighted_error = 0.0;
    double total_weight = 0.0;
    for (const auto& sample : samples) {
        const auto prediction =
            sigmoid_score(white_relative_eval(sample.position, parameters), scale);
        const auto error = prediction - sample.result;
        weighted_error += sample.weight * error * error;
        total_weight += sample.weight;
    }
    return total_weight > 0.0 ? weighted_error / total_weight : 0.0;
}

int choose_texel_scale(std::span<const TexelSample> samples, const EvaluationParameters& parameters,
                       TexelTuneConfig config) noexcept {
    const auto min_scale = normalized_scale(std::min(config.scale_min, config.scale_max));
    const auto max_scale = normalized_scale(std::max(config.scale_min, config.scale_max));
    const auto step = clamped_scale_step(config);

    auto best_scale = min_scale;
    auto best_loss = texel_loss(samples, parameters, best_scale);
    for (int scale = min_scale + step; scale <= max_scale; scale += step) {
        const auto candidate_loss = texel_loss(samples, parameters, scale);
        if (improved(candidate_loss, best_loss)) {
            best_scale = scale;
            best_loss = candidate_loss;
        }
    }
    return best_scale;
}

std::uint64_t texel_samples_checksum(std::span<const TexelSample> samples) noexcept {
    auto hash = checksum_offset_basis;
    for (const auto& sample : samples) {
        const auto fen = sample.position.to_fen();
        for (const auto symbol : fen) {
            mix_checksum_byte(hash, static_cast<std::uint8_t>(symbol));
        }
        mix_checksum_int(hash,
                         static_cast<int>(std::lround(sample.result * double_checksum_scale)));
        mix_checksum_int(hash,
                         static_cast<int>(std::lround(sample.weight * double_checksum_scale)));
        mix_checksum_u64(hash, sample.position.zobrist_key());
    }
    return hash;
}

TexelTuneResult tune_texel(std::span<const TexelSample> samples, TexelTuneConfig config,
                           EvaluationParameters initial_parameters) noexcept {
    TexelTuneResult result{
        .initial_parameters = initial_parameters,
        .final_parameters = initial_parameters,
        .scale = choose_texel_scale(samples, initial_parameters, config),
        .samples = samples.size(),
        .dataset_checksum = texel_samples_checksum(samples),
        .initial_parameter_checksum = evaluation_parameters_checksum(initial_parameters),
        .final_parameter_checksum = evaluation_parameters_checksum(initial_parameters),
    };
    result.initial_loss = texel_loss(samples, initial_parameters, result.scale);
    result.final_loss = result.initial_loss;

    const auto iterations = std::max(0, config.iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto step = std::max(config.min_step, config.initial_step >> iteration);
        if (step <= 0) {
            break;
        }

        for (const auto& descriptor : evaluation_parameter_descriptors()) {
            const auto original_value =
                evaluation_parameter_value(result.final_parameters, descriptor.name);
            auto best_value = original_value;
            auto best_loss = result.final_loss;

            for (const auto delta : {step, -step}) {
                const auto candidate_value =
                    std::clamp(original_value + delta, descriptor.min_value, descriptor.max_value);
                if (candidate_value == original_value) {
                    continue;
                }

                auto candidate_parameters = result.final_parameters;
                if (!set_evaluation_parameter(candidate_parameters, descriptor.name,
                                              candidate_value)) {
                    continue;
                }

                const auto candidate_loss = texel_loss(samples, candidate_parameters, result.scale);
                if (improved(candidate_loss, best_loss)) {
                    best_value = candidate_value;
                    best_loss = candidate_loss;
                }
            }

            if (best_value != original_value) {
                static_cast<void>(
                    set_evaluation_parameter(result.final_parameters, descriptor.name, best_value));
                result.final_loss = best_loss;
            }
        }
    }

    result.final_parameter_checksum = evaluation_parameters_checksum(result.final_parameters);
    return result;
}

} // namespace eng
