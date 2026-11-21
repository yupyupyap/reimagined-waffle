#include <eng/eval.hpp>
#include <eng/nnue.hpp>

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

// NOLINTBEGIN(bugprone-easily-swappable-parameters,
// bugprone-implicit-widening-of-multiplication-result,
// cppcoreguidelines-avoid-magic-numbers,readability-function-cognitive-complexity,
// readability-identifier-length)

namespace {

constexpr std::array<char, 16> magic{'L', 'C', 'E', 'N', 'N', 'U', 'E', 'W',
                                     'E', 'I', 'G', 'H', 'T', 'S', '1', '\0'};
constexpr std::uint32_t header_size = 68;
constexpr std::uint32_t small_hidden_size = 512;
constexpr std::uint32_t v1_hidden_size = 1024;
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
constexpr std::int64_t output_scale =
    static_cast<std::int64_t>(eng::nnue_network_qa) * eng::nnue_network_qb;

struct NetworkFixture {
    std::uint32_t version{eng::nnue_network_version};
    std::uint32_t arch_id{static_cast<std::uint8_t>(eng::NnueNetworkArchitecture::Small)};
    std::uint32_t hidden_size{small_hidden_size};
    std::uint32_t activation_id{static_cast<std::uint8_t>(eng::NnueNetworkActivation::CReLU)};
    std::string metadata{R"({"schema":"localchessengine.nnue.network.v2"})"};
    std::vector<std::int16_t> white_feature_weights =
        std::vector<std::int16_t>(eng::nnue_network_input_features * small_hidden_size);
    std::vector<std::int16_t> black_feature_weights =
        std::vector<std::int16_t>(eng::nnue_network_input_features * small_hidden_size);
    std::vector<std::int32_t> white_feature_bias = std::vector<std::int32_t>(small_hidden_size);
    std::vector<std::int32_t> black_feature_bias = std::vector<std::int32_t>(small_hidden_size);
    std::vector<std::int16_t> output_weights = std::vector<std::int16_t>(2 * small_hidden_size);
    std::int32_t output_bias{0};
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

void append_i16(std::vector<char>& bytes, std::int16_t value) {
    const auto raw = static_cast<std::uint16_t>(value);
    bytes.push_back(static_cast<char>(raw & 0xFFU));
    bytes.push_back(static_cast<char>((raw >> bits_per_byte) & 0xFFU));
}

void append_i32(std::vector<char>& bytes, std::int32_t value) {
    const auto raw = static_cast<std::uint32_t>(value);
    for (int shift = 0; shift < u32_bits; shift += bits_per_byte) {
        bytes.push_back(static_cast<char>((raw >> shift) & 0xFFU));
    }
}

[[nodiscard]] std::filesystem::path temp_path(std::string_view suffix) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("localchessengine-nnue-" + std::to_string(now) + std::string{suffix});
}

[[nodiscard]] eng::Square square(int file, int rank) noexcept {
    return eng::Square::from_file_rank(eng::File::from_index(file), eng::Rank::from_index(rank));
}

[[nodiscard]] NetworkFixture make_fixture(std::uint32_t hidden_size,
                                          eng::NnueNetworkArchitecture architecture,
                                          eng::NnueNetworkActivation activation) {
    return NetworkFixture{
        .arch_id = static_cast<std::uint8_t>(architecture),
        .hidden_size = hidden_size,
        .activation_id = static_cast<std::uint8_t>(activation),
        .metadata = R"({"schema":"localchessengine.nnue.network.v2"})",
        .white_feature_weights =
            std::vector<std::int16_t>(eng::nnue_network_input_features * hidden_size),
        .black_feature_weights =
            std::vector<std::int16_t>(eng::nnue_network_input_features * hidden_size),
        .white_feature_bias = std::vector<std::int32_t>(hidden_size),
        .black_feature_bias = std::vector<std::int32_t>(hidden_size),
        .output_weights = std::vector<std::int16_t>(2 * hidden_size),
        .output_bias = 0,
    };
}

void fill_deterministic_feature_weights(NetworkFixture& fixture) {
    for (std::uint32_t feature = 0; feature < eng::nnue_network_input_features; ++feature) {
        for (std::uint32_t hidden = 0; hidden < fixture.hidden_size; ++hidden) {
            const auto offset = (feature * fixture.hidden_size) + hidden;
            fixture.white_feature_weights[offset] =
                static_cast<std::int16_t>(((feature * 3U) + (hidden * 5U)) % 31U) - 15;
            fixture.black_feature_weights[offset] =
                static_cast<std::int16_t>(((feature * 7U) + (hidden * 2U)) % 29U) - 14;
        }
    }
}

void fill_deterministic_output_weights(NetworkFixture& fixture) {
    for (std::size_t index = 0; index < fixture.output_weights.size(); ++index) {
        fixture.output_weights[index] = static_cast<std::int16_t>(((index * 17U) + 9U) % 97U) - 48;
    }
    fixture.output_bias = static_cast<std::int32_t>(3 * output_scale);
}

void write_network(const std::filesystem::path& path, const NetworkFixture& fixture) {
    std::vector<char> tensor_payload;
    tensor_payload.reserve(static_cast<std::size_t>(
        eng::expected_nnue_tensor_bytes(eng::nnue_network_input_features, fixture.hidden_size)));
    for (const auto value : fixture.white_feature_weights) {
        append_i16(tensor_payload, value);
    }
    for (const auto value : fixture.black_feature_weights) {
        append_i16(tensor_payload, value);
    }
    if (fixture.version >= eng::nnue_network_version) {
        for (const auto value : fixture.white_feature_bias) {
            append_i32(tensor_payload, value);
        }
        for (const auto value : fixture.black_feature_bias) {
            append_i32(tensor_payload, value);
        }
    }
    for (const auto value : fixture.output_weights) {
        append_i16(tensor_payload, value);
    }
    append_i32(tensor_payload, fixture.output_bias);

    std::vector<char> header(header_size, '\0');
    std::ranges::copy(magic, header.begin());
    write_u32(header, version_offset, fixture.version);
    write_u32(header, header_size_offset, header_size);
    write_u32(header, arch_offset, fixture.arch_id);
    write_u32(header, input_features_offset, eng::nnue_network_input_features);
    write_u32(header, hidden_size_offset, fixture.hidden_size);
    write_u32(header, activation_offset, fixture.activation_id);
    write_u32(header, qa_offset, eng::nnue_network_qa);
    write_u32(header, qb_offset, eng::nnue_network_qb);
    write_u32(header, metadata_len_offset, static_cast<std::uint32_t>(fixture.metadata.size()));
    write_u64(header, tensor_bytes_offset, tensor_payload.size());

    auto checksum = checksum_offset_basis;
    mix_checksum_bytes(checksum, std::span<const char>(header.data() + checksum_tail_offset,
                                                       header.size() - checksum_tail_offset));
    mix_checksum_bytes(checksum, fixture.metadata);
    mix_checksum_bytes(checksum, tensor_payload);
    write_u64(header, checksum_offset, checksum);

    std::ofstream output(path, std::ios::binary);
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(fixture.metadata.data(), static_cast<std::streamsize>(fixture.metadata.size()));
    output.write(tensor_payload.data(), static_cast<std::streamsize>(tensor_payload.size()));
}

[[nodiscard]] eng::Position require_position(std::string_view fen) {
    auto position = eng::Position::from_fen(fen);
    REQUIRE(position.has_value());
    return *position;
}

[[nodiscard]] int round_cp(std::int64_t raw) noexcept {
    if (raw >= 0) {
        return static_cast<int>((raw + (output_scale / 2)) / output_scale);
    }
    return -static_cast<int>(((-raw) + (output_scale / 2)) / output_scale);
}

void set_feature_weight(std::vector<std::int16_t>& weights, std::uint32_t hidden_size,
                        std::uint32_t feature, std::uint32_t hidden, std::int16_t value) noexcept {
    weights[(feature * hidden_size) + hidden] = value;
}

[[nodiscard]] eng::NnueNetwork load_fixture(const NetworkFixture& fixture,
                                            std::string_view suffix) {
    const auto path = temp_path(suffix);
    write_network(path, fixture);
    auto network = eng::NnueNetwork::load(path);
    std::filesystem::remove(path);
    REQUIRE(network.has_value());
    return *network;
}

void check_incremental_matches_refresh(eng::Position& position, eng::NnueAccumulator& accumulator,
                                       const eng::NnueNetwork& network, eng::Move move) {
    eng::StateInfo state;
    const auto before = accumulator;
    REQUIRE(position.make_move(move, state));
    REQUIRE(eng::update_nnue_accumulator_after_make(accumulator, network, move, state));
    CHECK(accumulator == eng::refresh_nnue_accumulator(position, network));
    position.unmake_move(move, state);
    accumulator = before;
    CHECK(accumulator == eng::refresh_nnue_accumulator(position, network));
}

} // namespace

TEST_CASE("NNUE feature indices match trainer binpack perspective contract") {
    const auto black_rook = eng::Piece::make(eng::Color::Black, eng::PieceType::Rook);
    const auto white_rook = eng::Piece::make(eng::Color::White, eng::PieceType::Rook);

    CHECK(eng::nnue_feature_index(eng::Color::White, black_rook, square(0, 7)) ==
          ((6U + 3U) * 64U) + 56U);
    CHECK(eng::nnue_feature_index(eng::Color::Black, black_rook, square(0, 7)) == (3U * 64U));
    CHECK(eng::nnue_feature_index(eng::Color::Black, white_rook, square(0, 0)) ==
          ((6U + 3U) * 64U) + 56U);
    CHECK(eng::nnue_feature_index(eng::Color::White, eng::Piece{}, square(0, 0)) ==
          eng::nnue_invalid_feature_index);
}

TEST_CASE("NNUE backend selection reports scalar fallback and NEON availability") {
    CHECK(eng::resolve_nnue_inference_backend(eng::NnueInferenceBackend::Scalar) ==
          eng::NnueInferenceBackend::Scalar);
    CHECK(eng::resolve_nnue_inference_backend(eng::NnueInferenceBackend::Auto) ==
          (eng::nnue_neon_available() ? eng::NnueInferenceBackend::Neon
                                      : eng::NnueInferenceBackend::Scalar));
    CHECK(eng::resolve_nnue_inference_backend(eng::NnueInferenceBackend::Neon) ==
          (eng::nnue_neon_available() ? eng::NnueInferenceBackend::Neon
                                      : eng::NnueInferenceBackend::Scalar));
    CHECK(eng::nnue_inference_backend_name(eng::NnueInferenceBackend::Auto) == "auto");
    CHECK(eng::nnue_inference_backend_name(eng::NnueInferenceBackend::Scalar) == "scalar");
    CHECK(eng::nnue_inference_backend_name(eng::NnueInferenceBackend::Neon) == "neon");
}

TEST_CASE("NNUE loader consumes tensor payload and rejects corrupted networks") {
    auto fixture = make_fixture(small_hidden_size, eng::NnueNetworkArchitecture::Small,
                                eng::NnueNetworkActivation::CReLU);
    fill_deterministic_feature_weights(fixture);
    fixture.output_bias = 123;

    const auto valid_path = temp_path("-valid.nnue");
    const auto corrupt_path = temp_path("-corrupt.nnue");
    const auto truncated_path = temp_path("-truncated.nnue");
    write_network(valid_path, fixture);

    auto loaded = eng::NnueNetwork::load(valid_path);
    REQUIRE(loaded.has_value());
    CHECK(loaded->hidden_size() == small_hidden_size);
    CHECK(loaded->activation() == eng::NnueNetworkActivation::CReLU);
    CHECK(loaded->info().has_feature_bias);
    CHECK(loaded->info().side_to_move_ordering);

    {
        std::ifstream input(valid_path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                                std::istreambuf_iterator<char>());
        bytes.back() = static_cast<char>(bytes.back() ^ 0x7F);
        std::ofstream output(corrupt_path, std::ios::binary);
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    {
        std::ofstream output(truncated_path, std::ios::binary);
        output.write("LCEN", 4);
    }

    const auto corrupt = eng::NnueNetwork::load(corrupt_path);
    REQUIRE(!corrupt.has_value());
    CHECK(corrupt.error().code == eng::NnueNetworkErrorCode::ChecksumMismatch);

    const auto truncated = eng::NnueNetwork::load(truncated_path);
    REQUIRE(!truncated.has_value());
    CHECK(truncated.error().code == eng::NnueNetworkErrorCode::Truncated);

    std::filesystem::remove(valid_path);
    std::filesystem::remove(corrupt_path);
    std::filesystem::remove(truncated_path);
}

TEST_CASE("NNUE scalar evaluation matches integer reference for CReLU and SCReLU") {
    const auto rook = eng::Piece::make(eng::Color::White, eng::PieceType::Rook);
    const auto rook_square = square(0, 0);
    auto position = require_position("4k3/8/8/8/8/8/4K3/R7 w - - 0 1");
    auto black_to_move = require_position("4k3/8/8/8/8/8/4K3/R7 b - - 0 1");

    auto small = make_fixture(small_hidden_size, eng::NnueNetworkArchitecture::Small,
                              eng::NnueNetworkActivation::CReLU);
    set_feature_weight(small.white_feature_weights, small.hidden_size,
                       eng::nnue_feature_index(eng::Color::White, rook, rook_square), 0, 200);
    set_feature_weight(small.black_feature_weights, small.hidden_size,
                       eng::nnue_feature_index(eng::Color::Black, rook, rook_square), 0, 50);
    small.white_feature_bias[0] = 10;
    small.black_feature_bias[0] = 20;
    small.output_weights[0] = 1000;
    small.output_weights[small.hidden_size] = -500;
    small.output_bias = static_cast<std::int32_t>(7 * output_scale);
    const auto small_network = load_fixture(small, "-small-eval.nnue");
    const auto small_accumulator = eng::refresh_nnue_accumulator(position, small_network);
    const auto white_acc = 210;
    const auto black_acc = 70;
    const auto small_expected =
        round_cp((white_acc * 1000) - (black_acc * 500) + (7 * output_scale));
    const auto small_black_expected =
        round_cp((black_acc * 1000) - (white_acc * 500) + (7 * output_scale));
    CHECK(small_network.evaluate_white_relative(small_accumulator) == small_expected);
    CHECK(eng::evaluate_nnue(position, small_network, small_accumulator).value() == small_expected);
    CHECK(eng::evaluate_nnue(black_to_move, small_network).value() == small_black_expected);

    auto v1 = make_fixture(v1_hidden_size, eng::NnueNetworkArchitecture::V1,
                           eng::NnueNetworkActivation::SCReLU);
    set_feature_weight(v1.white_feature_weights, v1.hidden_size,
                       eng::nnue_feature_index(eng::Color::White, rook, rook_square), 0, 200);
    set_feature_weight(v1.black_feature_weights, v1.hidden_size,
                       eng::nnue_feature_index(eng::Color::Black, rook, rook_square), 0, 50);
    v1.output_weights[0] = 1000;
    v1.output_weights[v1.hidden_size] = -500;
    v1.output_bias = static_cast<std::int32_t>(7 * output_scale);
    const auto v1_network = load_fixture(v1, "-v1-eval.nnue");
    const auto v1_accumulator = eng::refresh_nnue_accumulator(position, v1_network);
    const auto white_screlu = ((200 * 200) + (eng::nnue_network_qa / 2)) / eng::nnue_network_qa;
    const auto black_screlu = ((50 * 50) + (eng::nnue_network_qa / 2)) / eng::nnue_network_qa;
    const auto v1_expected =
        round_cp((white_screlu * 1000) - (black_screlu * 500) + (7 * output_scale));
    CHECK(v1_network.evaluate_white_relative(v1_accumulator) == v1_expected);
}

TEST_CASE("NNUE NEON evaluation matches scalar for CReLU and SCReLU") {
    if (!eng::nnue_neon_available()) {
        MESSAGE("NEON backend is not available on this platform; scalar fallback remains covered");
        return;
    }

    auto small = make_fixture(small_hidden_size, eng::NnueNetworkArchitecture::Small,
                              eng::NnueNetworkActivation::CReLU);
    fill_deterministic_feature_weights(small);
    fill_deterministic_output_weights(small);
    const auto small_network = load_fixture(small, "-small-neon-eval.nnue");

    auto v1 = make_fixture(v1_hidden_size, eng::NnueNetworkArchitecture::V1,
                           eng::NnueNetworkActivation::SCReLU);
    fill_deterministic_feature_weights(v1);
    fill_deterministic_output_weights(v1);
    const auto v1_network = load_fixture(v1, "-v1-neon-eval.nnue");

    for (const auto fen : std::array{
             std::string_view{eng::Position::startpos_fen},
             std::string_view{"4k3/8/8/8/8/8/4K3/R7 w - - 0 1"},
             std::string_view{"4k3/8/8/8/8/8/4K3/R7 b - - 0 1"},
             std::string_view{"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/"
                              "R3K2R w KQkq - 0 1"},
         }) {
        const auto position = require_position(fen);

        const auto small_scalar = eng::refresh_nnue_accumulator(position, small_network,
                                                                eng::NnueInferenceBackend::Scalar);
        const auto small_neon =
            eng::refresh_nnue_accumulator(position, small_network, eng::NnueInferenceBackend::Neon);
        CHECK(small_scalar == small_neon);
        CHECK(small_network.evaluate_white_relative(small_scalar,
                                                    eng::NnueInferenceBackend::Scalar) ==
              small_network.evaluate_white_relative(small_neon, eng::NnueInferenceBackend::Neon));
        CHECK(eng::evaluate_nnue(position, small_network, small_scalar,
                                 eng::NnueInferenceBackend::Scalar) ==
              eng::evaluate_nnue(position, small_network, small_neon,
                                 eng::NnueInferenceBackend::Neon));

        const auto v1_scalar =
            eng::refresh_nnue_accumulator(position, v1_network, eng::NnueInferenceBackend::Scalar);
        const auto v1_neon =
            eng::refresh_nnue_accumulator(position, v1_network, eng::NnueInferenceBackend::Neon);
        CHECK(v1_scalar == v1_neon);
        CHECK(v1_network.evaluate_white_relative(v1_scalar, eng::NnueInferenceBackend::Scalar) ==
              v1_network.evaluate_white_relative(v1_neon, eng::NnueInferenceBackend::Neon));
        CHECK(eng::evaluate_nnue(position, v1_network, v1_scalar,
                                 eng::NnueInferenceBackend::Scalar) ==
              eng::evaluate_nnue(position, v1_network, v1_neon, eng::NnueInferenceBackend::Neon));
    }
}

TEST_CASE("NNUE incremental accumulator updates match full refresh across legal moves") {
    auto fixture = make_fixture(small_hidden_size, eng::NnueNetworkArchitecture::Small,
                                eng::NnueNetworkActivation::CReLU);
    fill_deterministic_feature_weights(fixture);
    const auto network = load_fixture(fixture, "-incremental.nnue");

    auto position = eng::Position::startpos();
    auto accumulator = eng::refresh_nnue_accumulator(position, network);
    std::vector<eng::Move> moves{
        eng::Move::normal(square(4, 1), square(4, 3)),
        eng::Move::normal(square(4, 6), square(4, 4)),
        eng::Move::normal(square(6, 0), square(5, 2)),
        eng::Move::normal(square(1, 7), square(2, 5)),
        eng::Move::normal(square(5, 0), square(1, 4)),
        eng::Move::normal(square(0, 6), square(0, 5)),
        eng::Move::castling(square(4, 0), square(6, 0)),
    };
    std::vector<eng::StateInfo> states(moves.size());
    std::vector<eng::NnueAccumulator> snapshots;
    snapshots.reserve(moves.size());

    for (std::size_t index = 0; index < moves.size(); ++index) {
        snapshots.push_back(accumulator);
        REQUIRE(position.make_move(moves[index], states[index]));
        REQUIRE(eng::update_nnue_accumulator_after_make(accumulator, network, moves[index],
                                                        states[index]));
        CHECK(accumulator == eng::refresh_nnue_accumulator(position, network));
    }

    for (std::size_t index = moves.size(); index > 0; --index) {
        const auto move_index = index - 1;
        position.unmake_move(moves[move_index], states[move_index]);
        accumulator = snapshots[move_index];
        CHECK(accumulator == eng::refresh_nnue_accumulator(position, network));
    }
}

TEST_CASE("NNUE NEON incremental updates match scalar across legal moves") {
    if (!eng::nnue_neon_available()) {
        MESSAGE("NEON backend is not available on this platform; scalar fallback remains covered");
        return;
    }

    auto fixture = make_fixture(small_hidden_size, eng::NnueNetworkArchitecture::Small,
                                eng::NnueNetworkActivation::CReLU);
    fill_deterministic_feature_weights(fixture);
    fill_deterministic_output_weights(fixture);
    const auto network = load_fixture(fixture, "-neon-incremental.nnue");

    auto position = eng::Position::startpos();
    auto scalar_accumulator =
        eng::refresh_nnue_accumulator(position, network, eng::NnueInferenceBackend::Scalar);
    auto neon_accumulator =
        eng::refresh_nnue_accumulator(position, network, eng::NnueInferenceBackend::Neon);
    CHECK(scalar_accumulator == neon_accumulator);

    std::vector<eng::Move> moves{
        eng::Move::normal(square(4, 1), square(4, 3)),
        eng::Move::normal(square(4, 6), square(4, 4)),
        eng::Move::normal(square(6, 0), square(5, 2)),
        eng::Move::normal(square(1, 7), square(2, 5)),
        eng::Move::normal(square(5, 0), square(1, 4)),
        eng::Move::normal(square(0, 6), square(0, 5)),
        eng::Move::castling(square(4, 0), square(6, 0)),
    };

    for (const auto move : moves) {
        eng::StateInfo state;
        REQUIRE(position.make_move(move, state));
        REQUIRE(eng::update_nnue_accumulator_after_make(scalar_accumulator, network, move, state,
                                                        eng::NnueInferenceBackend::Scalar));
        REQUIRE(eng::update_nnue_accumulator_after_make(neon_accumulator, network, move, state,
                                                        eng::NnueInferenceBackend::Neon));

        const auto scalar_refresh =
            eng::refresh_nnue_accumulator(position, network, eng::NnueInferenceBackend::Scalar);
        const auto neon_refresh =
            eng::refresh_nnue_accumulator(position, network, eng::NnueInferenceBackend::Neon);
        CHECK(scalar_accumulator == scalar_refresh);
        CHECK(neon_accumulator == neon_refresh);
        CHECK(scalar_accumulator == neon_accumulator);
        CHECK(network.evaluate_white_relative(scalar_accumulator,
                                              eng::NnueInferenceBackend::Scalar) ==
              network.evaluate_white_relative(neon_accumulator, eng::NnueInferenceBackend::Neon));
    }
}

TEST_CASE("NNUE incremental accumulator handles captures en passant promotions and null moves") {
    auto fixture = make_fixture(small_hidden_size, eng::NnueNetworkArchitecture::Small,
                                eng::NnueNetworkActivation::CReLU);
    fill_deterministic_feature_weights(fixture);
    const auto network = load_fixture(fixture, "-directed.nnue");

    {
        auto position = require_position("4k3/8/8/8/4p3/3P4/8/4K3 w - - 0 1");
        auto accumulator = eng::refresh_nnue_accumulator(position, network);
        check_incremental_matches_refresh(position, accumulator, network,
                                          eng::Move::normal(square(3, 2), square(4, 3)));
    }
    {
        auto position = require_position("7k/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        auto accumulator = eng::refresh_nnue_accumulator(position, network);
        check_incremental_matches_refresh(position, accumulator, network,
                                          eng::Move::en_passant(square(4, 4), square(3, 5)));
    }
    {
        auto position = require_position("4k3/6P1/8/8/8/8/8/4K3 w - - 0 1");
        auto accumulator = eng::refresh_nnue_accumulator(position, network);
        check_incremental_matches_refresh(
            position, accumulator, network,
            eng::Move::promotion(square(6, 6), square(6, 7), eng::PieceType::Queen));
    }
    {
        auto position = require_position("4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1");
        auto accumulator = eng::refresh_nnue_accumulator(position, network);
        check_incremental_matches_refresh(
            position, accumulator, network,
            eng::Move::promotion(square(6, 6), square(7, 7), eng::PieceType::Queen));
    }
    {
        auto position = require_position("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        auto accumulator = eng::refresh_nnue_accumulator(position, network);
        check_incremental_matches_refresh(position, accumulator, network,
                                          eng::Move::castling(square(4, 0), square(6, 0)));
    }
    {
        auto position = eng::Position::startpos();
        auto accumulator = eng::refresh_nnue_accumulator(position, network);
        const auto before = accumulator;
        eng::StateInfo state;
        REQUIRE(position.make_null_move(state));
        REQUIRE(eng::update_nnue_accumulator_after_make(accumulator, network, eng::Move::null(),
                                                        state));
        CHECK(accumulator == before);
        CHECK(accumulator == eng::refresh_nnue_accumulator(position, network));
        position.unmake_null_move(state);
    }
}

TEST_CASE("NNUE evaluator falls back to HCE when no network is loaded") {
    const auto position = eng::Position::startpos();
    const eng::NnueEvaluator evaluator;
    CHECK(!evaluator.has_network());
    CHECK(evaluator.evaluate(position) == eng::evaluate(position));
}

// NOLINTEND(bugprone-easily-swappable-parameters,
// bugprone-implicit-widening-of-multiplication-result,
// cppcoreguidelines-avoid-magic-numbers,readability-function-cognitive-complexity,
// readability-identifier-length)
