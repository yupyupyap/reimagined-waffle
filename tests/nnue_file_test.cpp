#include <eng/nnue_file.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::array<char, 16> magic{'L', 'C', 'E', 'N', 'N', 'U', 'E', 'W',
                                     'E', 'I', 'G', 'H', 'T', 'S', '1', '\0'};
constexpr std::uint32_t header_size = 68;
constexpr std::uint32_t small_hidden_size = 512;
constexpr std::uint32_t unsupported_architecture_id = 99;
constexpr std::uint8_t corruption_mask = 0x7F;
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
constexpr std::uint64_t checksum_offset_basis = 14695981039346656037ULL;
constexpr std::uint64_t checksum_prime = 1099511628211ULL;
constexpr int bits_per_byte = 8;
constexpr int u32_bits = 32;
constexpr int u64_bits = 64;

[[nodiscard]] constexpr std::uint64_t legacy_tensor_bytes(std::uint32_t input_features,
                                                          std::uint32_t hidden_size) noexcept {
    return (2ULL * input_features * hidden_size * 2ULL) + (2ULL * hidden_size * 2ULL) + 4ULL;
}

struct NetworkFixture {
    std::uint32_t version{eng::nnue_network_version};
    std::uint32_t arch_id{static_cast<std::uint8_t>(eng::NnueNetworkArchitecture::Small)};
    std::uint32_t input_features{eng::nnue_network_input_features};
    std::uint32_t hidden_size{small_hidden_size};
    std::uint32_t activation_id{static_cast<std::uint8_t>(eng::NnueNetworkActivation::CReLU)};
    std::uint32_t qa{eng::nnue_network_qa};
    std::uint32_t qb{eng::nnue_network_qb};
    std::string metadata{R"({"schema":"localchessengine.nnue.network.v2"})"};
    std::uint64_t tensor_bytes{
        eng::expected_nnue_tensor_bytes(eng::nnue_network_input_features, small_hidden_size)};
};

void mix_checksum_byte(std::uint64_t& hash, std::uint8_t value) noexcept {
    hash ^= value;
    hash *= checksum_prime;
}

void mix_checksum_bytes(std::uint64_t& hash, std::span<const char> bytes) noexcept {
    for (const auto byte : bytes) {
        mix_checksum_byte(hash, static_cast<std::uint8_t>(byte));
    }
}

void write_u32(std::vector<char>& bytes, std::size_t offset, std::uint32_t value) noexcept {
    for (int shift = 0; shift < u32_bits; shift += bits_per_byte) {
        bytes[offset + (shift / bits_per_byte)] = static_cast<char>(value >> shift);
    }
}

void write_u64(std::vector<char>& bytes, std::size_t offset, std::uint64_t value) noexcept {
    for (int shift = 0; shift < u64_bits; shift += bits_per_byte) {
        bytes[offset + (shift / bits_per_byte)] = static_cast<char>(value >> shift);
    }
}

[[nodiscard]] std::filesystem::path temp_path(std::string_view suffix) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("localchessengine-nnue-file-" + std::to_string(now) + std::string{suffix});
}

void write_network(const std::filesystem::path& path, const NetworkFixture& fixture) {
    std::vector<char> header(header_size, '\0');
    std::ranges::copy(magic, header.begin());
    write_u32(header, version_offset, fixture.version);
    write_u32(header, header_size_offset, header_size);
    write_u32(header, arch_offset, fixture.arch_id);
    write_u32(header, input_features_offset, fixture.input_features);
    write_u32(header, hidden_size_offset, fixture.hidden_size);
    write_u32(header, activation_offset, fixture.activation_id);
    write_u32(header, qa_offset, fixture.qa);
    write_u32(header, qb_offset, fixture.qb);
    write_u32(header, metadata_len_offset, static_cast<std::uint32_t>(fixture.metadata.size()));
    write_u64(header, tensor_bytes_offset, fixture.tensor_bytes);

    std::vector<char> payload(static_cast<std::size_t>(fixture.tensor_bytes), '\0');
    auto checksum = checksum_offset_basis;
    mix_checksum_bytes(checksum, std::span<const char>(header.data() + checksum_tail_offset,
                                                       header.size() - checksum_tail_offset));
    mix_checksum_bytes(checksum, fixture.metadata);
    mix_checksum_bytes(checksum, payload);
    write_u64(header, checksum_offset, checksum);

    std::ofstream output(path, std::ios::binary);
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(fixture.metadata.data(), static_cast<std::streamsize>(fixture.metadata.size()));
    output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

} // namespace

TEST_CASE("NNUE network reader accepts valid v2 header and metadata") {
    const auto path = temp_path("-valid.nnue");
    write_network(path, NetworkFixture{});

    const auto info = eng::read_nnue_network_info(path);
    REQUIRE(info.has_value());
    CHECK(info->version == eng::nnue_network_version);
    CHECK(info->architecture == eng::NnueNetworkArchitecture::Small);
    CHECK(info->activation == eng::NnueNetworkActivation::CReLU);
    CHECK(info->input_features == eng::nnue_network_input_features);
    CHECK(info->hidden_size == small_hidden_size);
    CHECK(info->qa == eng::nnue_network_qa);
    CHECK(info->qb == eng::nnue_network_qb);
    CHECK(info->tensor_bytes ==
          eng::expected_nnue_tensor_bytes(eng::nnue_network_input_features, small_hidden_size));
    CHECK(info->has_feature_bias);
    CHECK(info->side_to_move_ordering);
    CHECK(info->metadata_json.find("localchessengine.nnue.network.v2") != std::string::npos);

    std::filesystem::remove(path);
}

TEST_CASE("NNUE network reader keeps legacy v1 files loadable") {
    const auto path = temp_path("-legacy.nnue");
    write_network(path, NetworkFixture{
                            .version = eng::nnue_legacy_network_version,
                            .metadata = R"({"schema":"localchessengine.nnue.network.v1"})",
                            .tensor_bytes = legacy_tensor_bytes(eng::nnue_network_input_features,
                                                                small_hidden_size),
                        });

    const auto info = eng::read_nnue_network_info(path);
    REQUIRE(info.has_value());
    CHECK(info->version == eng::nnue_legacy_network_version);
    CHECK(!info->has_feature_bias);
    CHECK(!info->side_to_move_ordering);
    CHECK(info->tensor_bytes ==
          legacy_tensor_bytes(eng::nnue_network_input_features, small_hidden_size));

    std::filesystem::remove(path);
}

TEST_CASE("NNUE network reader rejects invalid files without crashing") {
    const auto valid_path = temp_path("-valid.nnue");
    const auto bad_magic_path = temp_path("-bad-magic.nnue");
    const auto truncated_path = temp_path("-truncated.nnue");
    const auto bad_checksum_path = temp_path("-bad-checksum.nnue");
    const auto bad_tensor_path = temp_path("-bad-tensor.nnue");
    const auto bad_arch_path = temp_path("-bad-arch.nnue");
    write_network(valid_path, NetworkFixture{});

    {
        std::ifstream input(valid_path, std::ios::binary);
        std::ofstream output(bad_magic_path, std::ios::binary);
        output << input.rdbuf();
        output.seekp(0);
        output.put('X');
    }
    {
        std::ofstream output(truncated_path, std::ios::binary);
        output.write("LCEN", 4);
    }
    {
        std::ifstream input(valid_path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                                std::istreambuf_iterator<char>());
        bytes.back() = static_cast<char>(bytes.back() ^ corruption_mask);
        std::ofstream output(bad_checksum_path, std::ios::binary);
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    write_network(bad_tensor_path, NetworkFixture{.tensor_bytes = 4});
    write_network(bad_arch_path, NetworkFixture{.arch_id = unsupported_architecture_id});

    const auto bad_magic = eng::read_nnue_network_info(bad_magic_path);
    REQUIRE(!bad_magic.has_value());
    CHECK(bad_magic.error().code == eng::NnueNetworkErrorCode::InvalidMagic);

    const auto truncated = eng::read_nnue_network_info(truncated_path);
    REQUIRE(!truncated.has_value());
    CHECK(truncated.error().code == eng::NnueNetworkErrorCode::Truncated);

    const auto bad_checksum = eng::read_nnue_network_info(bad_checksum_path);
    REQUIRE(!bad_checksum.has_value());
    CHECK(bad_checksum.error().code == eng::NnueNetworkErrorCode::ChecksumMismatch);

    const auto bad_tensor = eng::read_nnue_network_info(bad_tensor_path);
    REQUIRE(!bad_tensor.has_value());
    CHECK(bad_tensor.error().code == eng::NnueNetworkErrorCode::InvalidTensorSize);

    const auto bad_arch = eng::read_nnue_network_info(bad_arch_path);
    REQUIRE(!bad_arch.has_value());
    CHECK(bad_arch.error().code == eng::NnueNetworkErrorCode::UnsupportedArchitecture);

    std::filesystem::remove(valid_path);
    std::filesystem::remove(bad_magic_path);
    std::filesystem::remove(truncated_path);
    std::filesystem::remove(bad_checksum_path);
    std::filesystem::remove(bad_tensor_path);
    std::filesystem::remove(bad_arch_path);
}
