#include <eng/position.hpp>
#include <eng/texel.hpp>

#include <doctest/doctest.h>

#include <string_view>
#include <vector>

namespace {

constexpr double draw_result = 0.5;
constexpr double favored_result = 0.7;
constexpr double disfavored_result = 0.3;
constexpr double unit_weight = 1.0;
constexpr int fixed_scale = 400;
constexpr int scale_step = 100;
constexpr int tuning_iterations = 3;
constexpr int initial_step = 64;
constexpr int min_step = 8;

[[nodiscard]] eng::Position parse(std::string_view fen) {
    const auto position = eng::Position::from_fen(fen);
    REQUIRE(position.has_value());
    return *position;
}

[[nodiscard]] std::vector<eng::TexelSample> make_samples() {
    return {
        eng::TexelSample{
            .position = parse("4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
            .result = draw_result,
            .weight = unit_weight,
        },
        eng::TexelSample{
            .position = parse("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1"),
            .result = favored_result,
            .weight = unit_weight,
        },
        eng::TexelSample{
            .position = parse("4kq2/8/8/8/8/8/8/4K3 w - - 0 1"),
            .result = disfavored_result,
            .weight = unit_weight,
        },
    };
}

} // namespace

TEST_CASE("texel sample checksum is deterministic") {
    const auto samples = make_samples();

    CHECK(eng::texel_samples_checksum(samples) == eng::texel_samples_checksum(samples));
}

TEST_CASE("texel sanity tuning reduces loss deterministically") {
    const auto samples = make_samples();
    const eng::TexelTuneConfig config{
        .scale_min = fixed_scale,
        .scale_max = fixed_scale,
        .scale_step = scale_step,
        .iterations = tuning_iterations,
        .initial_step = initial_step,
        .min_step = min_step,
    };

    const auto first = eng::tune_texel(samples, config);
    const auto second = eng::tune_texel(samples, config);

    CHECK(first.scale == fixed_scale);
    CHECK(first.initial_loss > first.final_loss);
    CHECK(first.dataset_checksum == second.dataset_checksum);
    CHECK(first.initial_parameter_checksum == second.initial_parameter_checksum);
    CHECK(first.final_parameter_checksum == second.final_parameter_checksum);
    CHECK(first.final_loss == doctest::Approx(second.final_loss));
}
