#ifndef ENG_NNUE_HPP
#define ENG_NNUE_HPP

#include "eng/nnue_file.hpp"
#include "eng/position.hpp"
#include "eng/types.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

namespace eng {

inline constexpr std::uint32_t nnue_invalid_feature_index = UINT32_MAX;

enum class NnueInferenceBackend : std::uint8_t {
    Auto = 0,
    Scalar = 1,
    Neon = 2,
};

class NnueNetwork;

class NnueAccumulator {
  public:
    NnueAccumulator() = default;
    explicit NnueAccumulator(std::uint32_t hidden_size);

    void reset(std::uint32_t hidden_size);
    [[nodiscard]] bool copy_from(const NnueAccumulator& other) noexcept;

    [[nodiscard]] std::uint32_t hidden_size() const noexcept;
    [[nodiscard]] bool empty() const noexcept { return white_values_.empty(); }
    [[nodiscard]] std::span<const std::int32_t> white_values() const noexcept;
    [[nodiscard]] std::span<const std::int32_t> black_values() const noexcept;

    friend bool operator==(const NnueAccumulator& lhs, const NnueAccumulator& rhs) = default;

  private:
    friend class NnueNetwork;
    friend NnueAccumulator refresh_nnue_accumulator(const Position& position,
                                                    const NnueNetwork& network,
                                                    NnueInferenceBackend backend);
    friend bool update_nnue_accumulator_after_make(NnueAccumulator& accumulator,
                                                   const NnueNetwork& network, Move move,
                                                   const StateInfo& state,
                                                   NnueInferenceBackend backend) noexcept;

    std::vector<std::int32_t> white_values_;
    std::vector<std::int32_t> black_values_;
};

class NnueNetwork {
  public:
    NnueNetwork() = default;

    [[nodiscard]] static std::expected<NnueNetwork, NnueNetworkError>
    load(const std::filesystem::path& path);

    [[nodiscard]] const NnueNetworkInfo& info() const noexcept { return info_; }
    [[nodiscard]] NnueNetworkArchitecture architecture() const noexcept {
        return info_.architecture;
    }
    [[nodiscard]] NnueNetworkActivation activation() const noexcept { return info_.activation; }
    [[nodiscard]] std::uint32_t input_features() const noexcept { return info_.input_features; }
    [[nodiscard]] std::uint32_t hidden_size() const noexcept { return info_.hidden_size; }
    [[nodiscard]] bool is_loaded() const noexcept { return info_.hidden_size != 0U; }

    [[nodiscard]] int evaluate_white_relative(
        const NnueAccumulator& accumulator,
        NnueInferenceBackend backend = NnueInferenceBackend::Auto) const noexcept;
    [[nodiscard]] int
    evaluate_relative(const NnueAccumulator& accumulator, Color side_to_move,
                      NnueInferenceBackend backend = NnueInferenceBackend::Auto) const noexcept;

  private:
    friend NnueAccumulator refresh_nnue_accumulator(const Position& position,
                                                    const NnueNetwork& network,
                                                    NnueInferenceBackend backend);
    friend bool update_nnue_accumulator_after_make(NnueAccumulator& accumulator,
                                                   const NnueNetwork& network, Move move,
                                                   const StateInfo& state,
                                                   NnueInferenceBackend backend) noexcept;

    [[nodiscard]] bool
    add_feature_delta(NnueAccumulator& accumulator, Piece piece, Square square, int sign,
                      NnueInferenceBackend backend = NnueInferenceBackend::Auto) const noexcept;
    [[nodiscard]] bool
    move_feature_delta(NnueAccumulator& accumulator, Piece piece, Square from, Square to_square,
                       NnueInferenceBackend backend = NnueInferenceBackend::Auto) const noexcept;

    NnueNetworkInfo info_{};
    NnueNetworkTensors tensors_{};
};

[[nodiscard]] bool nnue_neon_available() noexcept;
[[nodiscard]] NnueInferenceBackend
resolve_nnue_inference_backend(NnueInferenceBackend backend) noexcept;
[[nodiscard]] std::string_view nnue_inference_backend_name(NnueInferenceBackend backend) noexcept;
[[nodiscard]] std::uint32_t nnue_feature_index(Color perspective, Piece piece,
                                               Square square) noexcept;
[[nodiscard]] NnueAccumulator
refresh_nnue_accumulator(const Position& position, const NnueNetwork& network,
                         NnueInferenceBackend backend = NnueInferenceBackend::Auto);
[[nodiscard]] bool update_nnue_accumulator_after_make(
    NnueAccumulator& accumulator, const NnueNetwork& network, Move move, const StateInfo& state,
    NnueInferenceBackend backend = NnueInferenceBackend::Auto) noexcept;
[[nodiscard]] Score
evaluate_nnue(const Position& position, const NnueNetwork& network,
              const NnueAccumulator& accumulator,
              NnueInferenceBackend backend = NnueInferenceBackend::Auto) noexcept;
[[nodiscard]] Score evaluate_nnue(const Position& position, const NnueNetwork& network,
                                  NnueInferenceBackend backend = NnueInferenceBackend::Auto);

class NnueEvaluator {
  public:
    NnueEvaluator() = default;
    explicit NnueEvaluator(const NnueNetwork* network,
                           NnueInferenceBackend backend = NnueInferenceBackend::Auto) noexcept
        : network_{network}, backend_{backend} {}
    explicit NnueEvaluator(const NnueNetwork& network,
                           NnueInferenceBackend backend = NnueInferenceBackend::Auto) noexcept
        : network_{&network}, backend_{backend} {}

    [[nodiscard]] bool has_network() const noexcept { return network_ != nullptr; }
    [[nodiscard]] Score evaluate(const Position& position) const;

  private:
    const NnueNetwork* network_{nullptr};
    NnueInferenceBackend backend_{NnueInferenceBackend::Auto};
};

} // namespace eng

#endif // ENG_NNUE_HPP
