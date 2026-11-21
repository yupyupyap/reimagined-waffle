#ifndef ENG_NNUE_FILE_HPP
#define ENG_NNUE_FILE_HPP

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace eng {

inline constexpr std::string_view nnue_network_format = "localchessengine.nnue.network.v2";
inline constexpr std::string_view nnue_legacy_network_format = "localchessengine.nnue.network.v1";
inline constexpr std::uint32_t nnue_network_version = 2;
inline constexpr std::uint32_t nnue_legacy_network_version = 1;
inline constexpr std::uint32_t nnue_network_qa = 255;
inline constexpr std::uint32_t nnue_network_qb = 64;
inline constexpr std::uint32_t nnue_network_input_features = 768;
inline constexpr std::uint32_t nnue_network_metadata_max_bytes = 1024U * 1024U;

enum class NnueNetworkArchitecture : std::uint8_t {
    Small = 1,
    V1 = 2,
};

enum class NnueNetworkActivation : std::uint8_t {
    CReLU = 1,
    SCReLU = 2,
};

enum class NnueNetworkErrorCode : std::uint8_t {
    Io = 0,
    InvalidMagic,
    UnsupportedVersion,
    InvalidHeader,
    UnsupportedArchitecture,
    InvalidTensorSize,
    MetadataTooLarge,
    Truncated,
    ChecksumMismatch,
};

struct NnueNetworkError {
    NnueNetworkErrorCode code{NnueNetworkErrorCode::Io};
    std::string message;

    friend bool operator==(const NnueNetworkError& lhs, const NnueNetworkError& rhs) = default;
};

struct NnueNetworkInfo {
    std::uint32_t version{0};
    NnueNetworkArchitecture architecture{NnueNetworkArchitecture::Small};
    NnueNetworkActivation activation{NnueNetworkActivation::CReLU};
    std::uint32_t input_features{0};
    std::uint32_t hidden_size{0};
    std::uint32_t qa{0};
    std::uint32_t qb{0};
    std::uint32_t metadata_bytes{0};
    std::uint64_t tensor_bytes{0};
    std::uint64_t checksum{0};
    std::uint64_t file_bytes{0};
    bool has_feature_bias{false};
    bool side_to_move_ordering{false};
    std::string metadata_json;

    friend bool operator==(const NnueNetworkInfo& lhs, const NnueNetworkInfo& rhs) = default;
};

struct NnueNetworkTensors {
    std::vector<std::int16_t> white_feature_weights;
    std::vector<std::int16_t> black_feature_weights;
    std::vector<std::int32_t> white_feature_bias;
    std::vector<std::int32_t> black_feature_bias;
    std::vector<std::int16_t> output_weights;
    std::int32_t output_bias{0};
};

struct NnueNetworkFile {
    NnueNetworkInfo info;
    NnueNetworkTensors tensors;
};

[[nodiscard]] std::uint64_t expected_nnue_tensor_bytes(std::uint32_t input_features,
                                                       std::uint32_t hidden_size) noexcept;
[[nodiscard]] std::expected<NnueNetworkInfo, NnueNetworkError>
read_nnue_network_info(const std::filesystem::path& path);
[[nodiscard]] std::expected<NnueNetworkFile, NnueNetworkError>
read_nnue_network_file(const std::filesystem::path& path);

} // namespace eng

#endif // ENG_NNUE_FILE_HPP
