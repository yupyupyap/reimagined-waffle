#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr std::array<char, 16> dataset_magic{'L', 'C', 'E', 'N', 'N', 'U', 'E', 'D',
                                             'A', 'T', 'A', 'S', 'E', 'T', '1', '\0'};
constexpr std::string_view dataset_format = "localchessengine.nnue.dataset.v1";
constexpr std::uint32_t dataset_version = 1;
constexpr std::uint64_t checksum_offset_basis = 14695981039346656037ULL;
constexpr std::uint64_t checksum_prime = 1099511628211ULL;
constexpr int bits_per_byte = 8;
constexpr int checksum_u16_bits = 16;
constexpr int checksum_u32_bits = 32;
constexpr int checksum_u64_bits = 64;
constexpr std::uint16_t move_none = 0;
constexpr std::uint16_t move_null = std::uint16_t{1} | (std::uint16_t{1} << 6U);
constexpr int squares_per_piece = 64;
constexpr int black_piece_offset = 6;
constexpr std::uint16_t wdl_permille_total = 1000;

class BinpackError : public std::runtime_error {
  public:
    explicit BinpackError(const std::string& message) : std::runtime_error(message) {}
};

struct Header {
    std::uint64_t record_count{0};
    std::uint64_t checksum{0};
};

struct Record {
    std::string fen;
    std::uint16_t best_move{0};
    std::int32_t eval_cp{0};
    std::uint16_t white_win{0};
    std::uint16_t draw{0};
    std::uint16_t black_win{0};
    std::uint16_t ply{0};
    std::uint16_t depth{0};
    std::uint64_t nodes{0};
    std::uint64_t zobrist_key{0};
    std::uint32_t game_index{0};
};

void mix_checksum_byte(std::uint64_t& hash, std::uint8_t value) noexcept {
    hash ^= value;
    hash *= checksum_prime;
}

void mix_checksum_u16(std::uint64_t& hash, std::uint16_t value) noexcept {
    for (int shift = 0; shift < checksum_u16_bits; shift += bits_per_byte) {
        mix_checksum_byte(hash, static_cast<std::uint8_t>(value >> shift));
    }
}

void mix_checksum_u32(std::uint64_t& hash, std::uint32_t value) noexcept {
    for (int shift = 0; shift < checksum_u32_bits; shift += bits_per_byte) {
        mix_checksum_byte(hash, static_cast<std::uint8_t>(value >> shift));
    }
}

void mix_checksum_i32(std::uint64_t& hash, std::int32_t value) noexcept {
    mix_checksum_u32(hash, static_cast<std::uint32_t>(value));
}

void mix_checksum_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
    for (int shift = 0; shift < checksum_u64_bits; shift += bits_per_byte) {
        mix_checksum_byte(hash, static_cast<std::uint8_t>(value >> shift));
    }
}

void mix_checksum_string(std::uint64_t& hash, std::string_view value) noexcept {
    mix_checksum_u64(hash, value.size());
    for (const auto symbol : value) {
        mix_checksum_byte(hash, static_cast<std::uint8_t>(symbol));
    }
}

void mix_record_checksum(std::uint64_t& hash, const Record& record) noexcept {
    mix_checksum_string(hash, record.fen);
    mix_checksum_u16(hash, record.best_move);
    mix_checksum_i32(hash, record.eval_cp);
    mix_checksum_u16(hash, record.white_win);
    mix_checksum_u16(hash, record.draw);
    mix_checksum_u16(hash, record.black_win);
    mix_checksum_u16(hash, record.ply);
    mix_checksum_u16(hash, record.depth);
    mix_checksum_u64(hash, record.nodes);
    mix_checksum_u64(hash, record.zobrist_key);
    mix_checksum_u32(hash, record.game_index);
}

[[nodiscard]] std::uint8_t read_u8(std::istream& input) {
    char value = '\0';
    if (!input.get(value)) {
        throw BinpackError("unexpected end of file");
    }
    return static_cast<std::uint8_t>(value);
}

[[nodiscard]] std::uint16_t read_u16(std::istream& input) {
    std::uint16_t value = 0;
    for (int shift = 0; shift < checksum_u16_bits; shift += bits_per_byte) {
        value = static_cast<std::uint16_t>(value |
                                           (static_cast<std::uint16_t>(read_u8(input)) << shift));
    }
    return value;
}

[[nodiscard]] std::uint32_t read_u32(std::istream& input) {
    std::uint32_t value = 0;
    for (int shift = 0; shift < checksum_u32_bits; shift += bits_per_byte) {
        value |= static_cast<std::uint32_t>(read_u8(input)) << shift;
    }
    return value;
}

[[nodiscard]] std::int32_t read_i32(std::istream& input) {
    return static_cast<std::int32_t>(read_u32(input));
}

[[nodiscard]] std::uint64_t read_u64(std::istream& input) {
    std::uint64_t value = 0;
    for (int shift = 0; shift < checksum_u64_bits; shift += bits_per_byte) {
        value |= static_cast<std::uint64_t>(read_u8(input)) << shift;
    }
    return value;
}

[[nodiscard]] std::vector<std::string_view> split_fields(std::string_view text, char delimiter) {
    std::vector<std::string_view> fields;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto next = text.find(delimiter, start);
        const auto end = next == std::string_view::npos ? text.size() : next;
        fields.push_back(text.substr(start, end - start));
        if (next == std::string_view::npos) {
            break;
        }
        start = next + 1;
    }
    return fields;
}

[[nodiscard]] bool is_piece_symbol(char symbol) noexcept {
    switch (symbol) {
    case 'P':
    case 'N':
    case 'B':
    case 'R':
    case 'Q':
    case 'K':
    case 'p':
    case 'n':
    case 'b':
    case 'r':
    case 'q':
    case 'k':
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool is_decimal_number(std::string_view text, bool positive) noexcept {
    if (text.empty()) {
        return false;
    }
    if (positive && text == "0") {
        return false;
    }
    return std::all_of(text.begin(), text.end(),
                       [](char symbol) { return symbol >= '0' && symbol <= '9'; });
}

[[nodiscard]] bool validate_piece_placement(std::string_view placement) noexcept {
    int rank_count = 1;
    int file_count = 0;
    int white_kings = 0;
    int black_kings = 0;
    for (const auto symbol : placement) {
        if (symbol == '/') {
            if (file_count != 8) {
                return false;
            }
            file_count = 0;
            ++rank_count;
            continue;
        }
        if (symbol >= '1' && symbol <= '8') {
            file_count += symbol - '0';
        } else if (is_piece_symbol(symbol)) {
            ++file_count;
            white_kings += symbol == 'K' ? 1 : 0;
            black_kings += symbol == 'k' ? 1 : 0;
        } else {
            return false;
        }
        if (file_count > 8) {
            return false;
        }
    }
    return rank_count == 8 && file_count == 8 && white_kings == 1 && black_kings == 1;
}

[[nodiscard]] bool validate_castling(std::string_view castling) noexcept {
    if (castling == "-") {
        return true;
    }
    constexpr std::string_view order = "KQkq";
    std::size_t cursor = 0;
    for (const auto symbol : castling) {
        const auto found = order.find(symbol, cursor);
        if (found == std::string_view::npos) {
            return false;
        }
        cursor = found + 1;
    }
    return !castling.empty();
}

[[nodiscard]] bool validate_ep(std::string_view ep) noexcept {
    if (ep == "-") {
        return true;
    }
    return ep.size() == 2 && ep[0] >= 'a' && ep[0] <= 'h' && (ep[1] == '3' || ep[1] == '6');
}

[[nodiscard]] bool validate_fen(std::string_view fen) {
    const auto fields = split_fields(fen, ' ');
    if (fields.size() != 6 || std::any_of(fields.begin(), fields.end(),
                                          [](std::string_view field) { return field.empty(); })) {
        return false;
    }
    return validate_piece_placement(fields[0]) && (fields[1] == "w" || fields[1] == "b") &&
           validate_castling(fields[2]) && validate_ep(fields[3]) &&
           is_decimal_number(fields[4], false) && is_decimal_number(fields[5], true);
}

[[nodiscard]] int piece_type_index(char symbol) {
    switch (symbol) {
    case 'P':
    case 'p':
        return 0;
    case 'N':
    case 'n':
        return 1;
    case 'B':
    case 'b':
        return 2;
    case 'R':
    case 'r':
        return 3;
    case 'Q':
    case 'q':
        return 4;
    case 'K':
    case 'k':
        return 5;
    default:
        throw BinpackError("invalid FEN piece symbol");
    }
}

[[nodiscard]] bool is_white_piece(char symbol) noexcept { return symbol >= 'A' && symbol <= 'Z'; }

void append_features(std::string_view fen, std::vector<std::int64_t>& white_indices,
                     std::vector<std::int64_t>& black_indices) {
    const auto fields = split_fields(fen, ' ');
    const auto placement = fields[0];
    int rank = 7;
    int file = 0;
    for (const auto symbol : placement) {
        if (symbol == '/') {
            --rank;
            file = 0;
            continue;
        }
        if (symbol >= '1' && symbol <= '8') {
            file += symbol - '0';
            continue;
        }

        const auto square = (rank * 8) + file;
        const auto type = piece_type_index(symbol);
        const auto white_piece_index = (is_white_piece(symbol) ? 0 : black_piece_offset) + type;
        const auto black_piece_index = (is_white_piece(symbol) ? black_piece_offset : 0) + type;
        white_indices.push_back((white_piece_index * squares_per_piece) + square);
        black_indices.push_back((black_piece_index * squares_per_piece) + (square ^ 56));
        ++file;
    }
}

[[nodiscard]] std::int64_t side_to_move_index(std::string_view fen) {
    const auto fields = split_fields(fen, ' ');
    if (fields.size() < 2) {
        throw BinpackError("dataset record contains malformed FEN side-to-move");
    }
    if (fields[1] == "w") {
        return 0;
    }
    if (fields[1] == "b") {
        return 1;
    }
    throw BinpackError("dataset record contains invalid FEN side-to-move");
}

[[nodiscard]] Header read_header(std::istream& input) {
    std::array<char, dataset_magic.size()> magic{};
    if (!input.read(magic.data(), static_cast<std::streamsize>(magic.size()))) {
        throw BinpackError("dataset is truncated before magic header");
    }
    if (magic != dataset_magic) {
        throw BinpackError("dataset magic does not match");
    }

    const auto version = read_u32(input);
    if (version != dataset_version) {
        throw BinpackError("unsupported dataset version");
    }
    return Header{read_u64(input), read_u64(input)};
}

[[nodiscard]] Record read_record(std::istream& input) {
    const auto fen_size = read_u16(input);
    std::string fen(fen_size, '\0');
    if (!fen.empty() && !input.read(fen.data(), static_cast<std::streamsize>(fen.size()))) {
        throw BinpackError("truncated FEN payload in dataset record");
    }

    Record record;
    record.fen = std::move(fen);
    record.best_move = read_u16(input);
    record.eval_cp = read_i32(input);
    record.white_win = read_u16(input);
    record.draw = read_u16(input);
    record.black_win = read_u16(input);
    record.ply = read_u16(input);
    record.depth = read_u16(input);
    record.nodes = read_u64(input);
    record.zobrist_key = read_u64(input);
    record.game_index = read_u32(input);

    if (!validate_fen(record.fen)) {
        throw BinpackError("dataset record contains invalid FEN");
    }
    if (record.best_move == move_none || record.best_move == move_null) {
        throw BinpackError("dataset record has invalid best move");
    }
    const auto wdl_sum = static_cast<int>(record.white_win) + static_cast<int>(record.draw) +
                         static_cast<int>(record.black_win);
    if (wdl_sum != wdl_permille_total) {
        throw BinpackError("dataset record WDL permille does not sum to 1000");
    }
    return record;
}

[[nodiscard]] std::ifstream open_input(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw BinpackError("could not open dataset for reading: " + path);
    }
    return input;
}

void ensure_no_trailing_bytes(std::istream& input) {
    char trailing = '\0';
    if (input.get(trailing)) {
        throw BinpackError("dataset contains trailing bytes");
    }
}

void validate_checksum(std::uint64_t observed, std::uint64_t expected) {
    if (observed != expected) {
        throw BinpackError("dataset checksum mismatch");
    }
}

template <typename T> [[nodiscard]] py::array_t<T> make_array(const std::vector<T>& values) {
    py::array_t<T> array(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) {
        std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(T));
    }
    return array;
}

[[nodiscard]] py::dict make_metadata_dict(const std::string& path, const Header& header) {
    py::dict result;
    result["path"] = path;
    result["format"] = std::string{dataset_format};
    result["version"] = dataset_version;
    result["record_count"] = py::int_(header.record_count);
    result["checksum"] = py::int_(header.checksum);
    return result;
}

[[nodiscard]] py::dict metadata(const std::string& path) {
    auto input = open_input(path);
    const auto header = read_header(input);
    return make_metadata_dict(path, header);
}

[[nodiscard]] py::dict validate_dataset(const std::string& path) {
    auto input = open_input(path);
    const auto header = read_header(input);
    std::uint64_t checksum = checksum_offset_basis;
    for (std::uint64_t index = 0; index < header.record_count; ++index) {
        const auto record = read_record(input);
        mix_record_checksum(checksum, record);
    }
    ensure_no_trailing_bytes(input);
    validate_checksum(checksum, header.checksum);

    py::dict result;
    result["path"] = path;
    result["record_count"] = py::int_(header.record_count);
    result["checksum"] = py::int_(checksum);
    return result;
}

class BatchIterator {
  public:
    BatchIterator(std::string path, std::uint64_t batch_size)
        : path_{std::move(path)}, batch_size_{batch_size}, input_{open_input(path_)},
          header_{read_header(input_)} {
        if (batch_size_ == 0) {
            throw BinpackError("batch size must be positive");
        }
    }

    [[nodiscard]] BatchIterator& iter() noexcept { return *this; }

    [[nodiscard]] py::dict next() {
        if (records_read_ >= header_.record_count) {
            finish();
            throw py::stop_iteration();
        }

        const auto remaining = header_.record_count - records_read_;
        const auto count = std::min(batch_size_, remaining);
        std::vector<std::int64_t> white_indices;
        std::vector<std::int64_t> black_indices;
        std::vector<std::int64_t> offsets;
        std::vector<std::int64_t> side_to_move;
        std::vector<float> eval_cp;
        std::vector<float> wdl;
        white_indices.reserve(static_cast<std::size_t>(count) * 32U);
        black_indices.reserve(static_cast<std::size_t>(count) * 32U);
        offsets.reserve(static_cast<std::size_t>(count));
        side_to_move.reserve(static_cast<std::size_t>(count));
        eval_cp.reserve(static_cast<std::size_t>(count));
        wdl.reserve(static_cast<std::size_t>(count));

        for (std::uint64_t index = 0; index < count; ++index) {
            const auto record = read_record(input_);
            mix_record_checksum(checksum_, record);
            offsets.push_back(static_cast<std::int64_t>(white_indices.size()));
            append_features(record.fen, white_indices, black_indices);
            side_to_move.push_back(side_to_move_index(record.fen));
            eval_cp.push_back(static_cast<float>(record.eval_cp));
            wdl.push_back(
                (static_cast<float>(record.white_win) + (0.5F * static_cast<float>(record.draw))) /
                static_cast<float>(wdl_permille_total));
            ++records_read_;
        }

        if (white_indices.size() != black_indices.size()) {
            throw BinpackError("internal feature extraction mismatch");
        }

        py::dict result;
        result["white_indices"] = make_array(white_indices);
        result["black_indices"] = make_array(black_indices);
        result["offsets"] = make_array(offsets);
        result["side_to_move"] = make_array(side_to_move);
        result["eval_cp"] = make_array(eval_cp);
        result["wdl"] = make_array(wdl);
        return result;
    }

  private:
    void finish() {
        if (finished_) {
            return;
        }
        ensure_no_trailing_bytes(input_);
        validate_checksum(checksum_, header_.checksum);
        finished_ = true;
    }

    std::string path_;
    std::uint64_t batch_size_{0};
    std::ifstream input_;
    Header header_;
    std::uint64_t records_read_{0};
    std::uint64_t checksum_{checksum_offset_basis};
    bool finished_{false};
};

} // namespace

PYBIND11_MODULE(_binpack_native, module) {
    py::register_exception<BinpackError>(module, "BinpackError");
    module.doc() = "Native LocalChessEngine NNUE binpack loader.";
    module.def("metadata", &metadata, py::arg("path"));
    module.def("validate", &validate_dataset, py::arg("path"));
    py::class_<BatchIterator>(module, "BatchIterator")
        .def(py::init<std::string, std::uint64_t>(), py::arg("path"), py::arg("batch_size"))
        .def("__iter__", &BatchIterator::iter, py::return_value_policy::reference_internal)
        .def("__next__", &BatchIterator::next);
}
