#ifndef ENG_TEXEL_HPP
#define ENG_TEXEL_HPP

#include "eng/eval.hpp"
#include "eng/position.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace eng {

inline constexpr double texel_default_result = 0.5;
inline constexpr double texel_default_weight = 1.0;
inline constexpr int texel_default_scale_min = 200;
inline constexpr int texel_default_scale_max = 800;
inline constexpr int texel_default_scale_step = 100;
inline constexpr int texel_default_iterations = 2;
inline constexpr int texel_default_initial_step = 8;
inline constexpr int texel_default_min_step = 1;
inline constexpr int texel_default_scale = 400;

struct TexelSample {
    Position position;
    double result{texel_default_result};
    double weight{texel_default_weight};
};

struct TexelTuneConfig {
    int scale_min{texel_default_scale_min};
    int scale_max{texel_default_scale_max};
    int scale_step{texel_default_scale_step};
    int iterations{texel_default_iterations};
    int initial_step{texel_default_initial_step};
    int min_step{texel_default_min_step};
};

struct TexelTuneResult {
    EvaluationParameters initial_parameters{};
    EvaluationParameters final_parameters{};
    double initial_loss{0.0};
    double final_loss{0.0};
    int scale{texel_default_scale};
    std::size_t samples{0};
    std::uint64_t dataset_checksum{0};
    std::uint64_t initial_parameter_checksum{0};
    std::uint64_t final_parameter_checksum{0};
};

[[nodiscard]] double texel_loss(std::span<const TexelSample> samples,
                                const EvaluationParameters& parameters, int scale) noexcept;
[[nodiscard]] int choose_texel_scale(std::span<const TexelSample> samples,
                                     const EvaluationParameters& parameters,
                                     TexelTuneConfig config) noexcept;
[[nodiscard]] std::uint64_t texel_samples_checksum(std::span<const TexelSample> samples) noexcept;
[[nodiscard]] TexelTuneResult
tune_texel(std::span<const TexelSample> samples, TexelTuneConfig config,
           EvaluationParameters initial_parameters = default_evaluation_parameters()) noexcept;

} // namespace eng

#endif // ENG_TEXEL_HPP
