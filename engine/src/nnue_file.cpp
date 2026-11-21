#include <eng/nnue_file.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace eng {
namespace {

constexpr std::array<char, 16> network_magic{'L', 'C', 'E', 'N', 'N', 'U', 'E', 'W',
                                             'E', 'I', 'G', 'H', 'T', 'S', '1', '\0'};
constexpr std::uint32_t header_size = 68;
constexpr std::uint32_t small_hidden_size = 512;
constexpr std::uint32_t v1_hidden_size = 1024;
constexpr std::uint32_t int16_bytes = 2;
constexpr std::uint32_t int32_bytes = 4;
constexpr std::uint64_t checksum_offset_basis = 14695981039346656037ULL;
constexpr std::uint64_t checksum_prime = 1099511628211ULL;
constexpr int bits_per_byte = 8;
constexpr int int16_bits = 16;
constexpr int u32_bits = 32;
constexpr int u64_bits = 64;
constexpr std::size_t version_offset = 16;
constexpr std::size_t header_size_offset = 20;
constexpr std::size_t checksum_offset = 24;
constexpr std::size_t checksum_tail_offset = 32;
constexpr std::size_t arch_offset = 32;
constexpr std::size_t input_features_offset = 36;
constexpr std::size_t hidden_size_offset = 40;
constexpr std::size_t activation_offset = 44;
constexpr std::size_t qa_offset = 48;
constexpr std::size_t qb_offset = 52;
constexpr std::size_t metadata_len_offset = 56;
constexpr std::size_t tensor_bytes_offset = 60;
constexpr std::size_t io_chunk_bytes = std::size_t{16} * 1024U;

[[nodiscard]] NnueNetworkError make_error(NnueNetworkErrorCode code, std::string message) {
    return NnueNetworkError{.code = code, .message = std::move(message)};
}

[[nodiscard]] bool supported_network_version(std::uint32_t version) noexcept {
    return version == nnue_legacy_network_version || version == nnue_network_version;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] std::uint64_t expected_tensor_bytes_for_version(std::uint32_t version,
                                                              std::uint32_t input_features,
                                                              std::uint32_t hidden_size) noexcept {
    const auto feature_bytes = 2ULL * input_features * hidden_size * int16_bytes;
    const auto hidden_bias_bytes =
        version >= nnue_network_version ? 2ULL * hidden_size * int32_bytes : 0ULL;
    const auto output_weight_bytes = 2ULL * hidden_size * int16_bytes;
    return feature_bytes + hidden_bias_bytes + output_weight_bytes + int32_bytes;
}

[[nodiscard]] std::uint32_t read_u32_le(std::span<const char> bytes, std::size_t offset) noexcept {
    std::uint32_t value = 0;
    for (int shift = 0; shift < u32_bits; shift += bits_per_byte) {
        const auto byte = static_cast<unsigned char>(bytes[offset + (shift / bits_per_byte)]);
        value |= static_cast<std::uint32_t>(byte) << shift;
    }
    return value;
}

[[nodiscard]] std::uint64_t read_u64_le(std::span<const char> bytes, std::size_t offset) noexcept {
    std::uint64_t value = 0;
    for (int shift = 0; shift < u64_bits; shift += bits_per_byte) {
        const auto byte = static_cast<unsigned char>(bytes[offset + (shift / bits_per_byte)]);
        value |= static_cast<std::uint64_t>(byte) << shift;
    }
    return value;
}

[[nodiscard]] std::int16_t read_i16_le(std::span<const char> bytes, std::size_t offset) noexcept {
    std::uint16_t value = 0;
    for (int shift = 0; shift < int16_bits; shift += bits_per_byte) {
        const auto byte = static_cast<unsigned char>(bytes[offset + (shift / bits_per_byte)]);
        value |= static_cast<std::uint16_t>(byte) << shift;
    }
    return std::bit_cast<std::int16_t>(value);
}

[[nodiscard]] std::int32_t read_i32_le(std::span<const char> bytes, std::size_t offset) noexcept {
    const auto value = read_u32_le(bytes, offset);
    return std::bit_cast<std::int32_t>(value);
}

void mix_checksum_byte(std::uint64_t& hash, std::uint8_t value) noexcept {
    hash ^= value;
    hash *= checksum_prime;
}

void mix_checksum_bytes(std::uint64_t& hash, std::span<const char> bytes) noexcept {
    for (const auto byte : bytes) {
        mix_checksum_byte(hash, static_cast<std::uint8_t>(byte));
    }
}

[[nodiscard]] bool is_supported_layout(NnueNetworkArchitecture architecture,
                                       std::uint32_t hidden_size,
                                       NnueNetworkActivation activation) noexcept {
    if (architecture == NnueNetworkArchitecture::Small) {
        return hidden_size == small_hidden_size && activation == NnueNetworkActivation::CReLU;
    }
    if (architecture == NnueNetworkArchitecture::V1) {
        return hidden_size == v1_hidden_size && activation == NnueNetworkActivation::SCReLU;
    }
    return false;
}

[[nodiscard]] std::expected<std::string, NnueNetworkError>
read_metadata(std::istream& input, std::uint32_t size, std::uint64_t& checksum) {
    std::string metadata(size, '\0');
    if (size == 0) {
        return metadata;
    }
    if (!input.read(metadata.data(), static_cast<std::streamsize>(metadata.size()))) {
        return std::unexpected(
            make_error(NnueNetworkErrorCode::Truncated, "truncated NNUE metadata"));
    }
    mix_checksum_bytes(checksum, metadata);
    return metadata;
}

[[nodiscard]] std::expected<void, NnueNetworkError>
consume_tensor_payload(std::istream& input, std::uint64_t size, std::uint64_t& checksum) {
    std::vector<char> buffer(io_chunk_bytes);
    std::uint64_t remaining = size;
    while (remaining > 0) {
        const auto wanted = static_cast<std::streamsize>(
            std::min<std::uint64_t>(remaining, static_cast<std::uint64_t>(buffer.size())));
        if (!input.read(buffer.data(), wanted)) {
            return std::unexpected(
                make_error(NnueNetworkErrorCode::Truncated, "truncated NNUE tensor payload"));
        }
        mix_checksum_bytes(checksum, std::span<const char>(buffer.data(), wanted));
        remaining -= static_cast<std::uint64_t>(wanted);
    }
    return {};
}

} // namespace

std::uint64_t expected_nnue_tensor_bytes(std::uint32_t input_features,
                                         std::uint32_t hidden_size) noexcept {
    return expected_tensor_bytes_for_version(nnue_network_version, input_features, hidden_size);
}

std::expected<NnueNetworkInfo, NnueNetworkError>
read_nnue_network_info(const std::filesystem::path& path) {
    std::error_code file_size_error;
    const auto file_bytes = std::filesystem::file_size(path, file_size_error);
    if (file_size_error) {
        return std::unexpected(
            make_error(NnueNetworkErrorCode::Io, "could not stat NNUE network file"));
    }
    if (file_bytes < header_size) {
        return std::unexpected(
            make_error(NnueNetworkErrorCode::Truncated, "NNUE network header is truncated"));
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return std::unexpected(
            make_error(NnueNetworkErrorCode::Io, "could not open NNUE network file"));
    }

    std::array<char, header_size> header{};
    if (!input.read(header.data(), static_cast<std::streamsize>(header.size()))) {
        return std::unexpected(
            make_error(NnueNetworkErrorCode::Truncated, "NNUE network header is truncated"));
    }

    if (!std::equal(network_magic.begin(), network_magic.end(), header.begin())) {
        return std::unexpected(
            make_error(NnueNetworkErrorCode::InvalidMagic, "NNUE network magic mismatch"));
    }

    const auto version = read_u32_le(header, version_offset);
    if (!supported_network_version(version)) {
        return std::unexpected(make_error(NnueNetworkErrorCode::UnsupportedVersion,
                                          "unsupported NNUE network version"));
    }

    const auto observed_header_size = read_u32_le(header, header_size_offset);
    if (observed_header_size != header_size) {
        return std::unexpected(
            make_error(NnueNetworkErrorCode::InvalidHeader, "NNUE network header size mismatch"));
    }

    const auto checksum = read_u64_le(header, checksum_offset);
    const auto arch_id = read_u32_le(header, arch_offset);
    const auto input_features = read_u32_le(header, input_features_offset);
    const auto hidden_size = read_u32_le(header, hidden_size_offset);
    const auto activation_id = read_u32_le(header, activation_offset);
    const auto quantization_a = read_u32_le(header, qa_offset);
    const auto quantization_b = read_u32_le(header, qb_offset);
    const auto metadata_bytes = read_u32_le(header, metadata_len_offset);
    const auto tensor_bytes = read_u64_le(header, tensor_bytes_offset);

    if (input_features != nnue_network_input_features || quantization_a != nnue_network_qa ||
        quantization_b != nnue_network_qb) {
        return std::unexpected(
            make_error(NnueNetworkErrorCode::InvalidHeader, "NNUE network constants mismatch"));
    }
    if (arch_id > std::numeric_limits<std::uint8_t>::max() ||
        activation_id > std::numeric_limits<std::uint8_t>::max()) {
        return std::unexpected(make_error(NnueNetworkErrorCode::UnsupportedArchitecture,
                                          "unsupported NNUE network architecture"));
    }
    const auto architecture = static_cast<NnueNetworkArchitecture>(arch_id);
    const auto activation = static_cast<NnueNetworkActivation>(activation_id);
    if (!is_supported_layout(architecture, hidden_size, activation)) {
        return std::unexpected(make_error(NnueNetworkErrorCode::UnsupportedArchitecture,
                                          "unsupported NNUE network architecture"));
    }
    if (metadata_bytes > nnue_network_metadata_max_bytes) {
        return std::unexpected(make_error(NnueNetworkErrorCode::MetadataTooLarge,
                                          "NNUE network metadata is too large"));
    }
    const auto expected_tensor_bytes =
        expected_tensor_bytes_for_version(version, input_features, hidden_size);
    if (tensor_bytes != expected_tensor_bytes) {
        return std::unexpected(make_error(NnueNetworkErrorCode::InvalidTensorSize,
                                          "NNUE network tensor byte count mismatch"));
    }

    const auto expected_file_bytes =
        static_cast<std::uint64_t>(header_size) + metadata_bytes + tensor_bytes;
    if (file_bytes != expected_file_bytes) {
        return std::unexpected(make_error(NnueNetworkErrorCode::Truncated,
                                          "NNUE network file size does not match header"));
    }

    auto observed_checksum = checksum_offset_basis;
    mix_checksum_bytes(observed_checksum,
                       std::span<const char>(header.data() + checksum_tail_offset,
                                             header.size() - checksum_tail_offset));
    const auto metadata = read_metadata(input, metadata_bytes, observed_checksum);
    if (!metadata.has_value()) {
        return std::unexpected(metadata.error());
    }
    const auto consumed = consume_tensor_payload(input, tensor_bytes, observed_checksum);
    if (!consumed.has_value()) {
        return std::unexpected(consumed.error());
    }
    if (observed_checksum != checksum) {
        return std::unexpected(
            make_error(NnueNetworkErrorCode::ChecksumMismatch, "NNUE network checksum mismatch"));
    }

    return NnueNetworkInfo{
        .version = version,
        .architecture = architecture,
        .activation = activation,
        .input_features = input_features,
        .hidden_size = hidden_size,
        .qa = quantization_a,
        .qb = quantization_b,
        .metadata_bytes = metadata_bytes,
        .tensor_bytes = tensor_bytes,
        .checksum = checksum,
        .file_bytes = file_bytes,
        .has_feature_bias = version >= nnue_network_version,
        .side_to_move_ordering = version >= nnue_network_version,
        .metadata_json = *metadata,
    };
}

std::expected<NnueNetworkFile, NnueNetworkError>
read_nnue_network_file(const std::filesystem::path& path) {
    const auto info = read_nnue_network_info(path);
    if (!info.has_value()) {
        return std::unexpected(info.error());
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return std::unexpected(
            make_error(NnueNetworkErrorCode::Io, "could not open NNUE network file"));
    }

    const auto tensor_offset =
        static_cast<std::uint64_t>(header_size) + static_cast<std::uint64_t>(info->metadata_bytes);
    input.seekg(static_cast<std::streamoff>(tensor_offset), std::ios::beg);
    if (!input) {
        return std::unexpected(
            make_error(NnueNetworkErrorCode::Io, "could not seek NNUE tensor payload"));
    }

    std::vector<char> payload(static_cast<std::size_t>(info->tensor_bytes));
    if (!input.read(payload.data(), static_cast<std::streamsize>(payload.size()))) {
        return std::unexpected(
            make_error(NnueNetworkErrorCode::Truncated, "truncated NNUE tensor payload"));
    }

    const auto hidden_size = static_cast<std::size_t>(info->hidden_size);
    const auto feature_weight_count = static_cast<std::size_t>(info->input_features) *
                                      static_cast<std::size_t>(info->hidden_size);
    const auto output_weight_count = 2ULL * hidden_size;
    NnueNetworkTensors tensors{
        .white_feature_weights = std::vector<std::int16_t>(feature_weight_count),
        .black_feature_weights = std::vector<std::int16_t>(feature_weight_count),
        .white_feature_bias = std::vector<std::int32_t>(hidden_size, 0),
        .black_feature_bias = std::vector<std::int32_t>(hidden_size, 0),
        .output_weights = std::vector<std::int16_t>(static_cast<std::size_t>(output_weight_count)),
        .output_bias = 0,
    };

    std::size_t offset = 0;
    for (auto& value : tensors.white_feature_weights) {
        value = read_i16_le(payload, offset);
        offset += int16_bytes;
    }
    for (auto& value : tensors.black_feature_weights) {
        value = read_i16_le(payload, offset);
        offset += int16_bytes;
    }
    if (info->has_feature_bias) {
        for (auto& value : tensors.white_feature_bias) {
            value = read_i32_le(payload, offset);
            offset += int32_bytes;
        }
        for (auto& value : tensors.black_feature_bias) {
            value = read_i32_le(payload, offset);
            offset += int32_bytes;
        }
    }
    for (auto& value : tensors.output_weights) {
        value = read_i16_le(payload, offset);
        offset += int16_bytes;
    }
    tensors.output_bias = read_i32_le(payload, offset);
    offset += int32_bytes;

    if (offset != payload.size()) {
        return std::unexpected(make_error(NnueNetworkErrorCode::InvalidTensorSize,
                                          "NNUE tensor payload parsing mismatch"));
    }

    return NnueNetworkFile{.info = *info, .tensors = std::move(tensors)};
}

} // namespace eng
