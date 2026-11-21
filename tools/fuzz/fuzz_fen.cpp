#include <eng/position.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace {

constexpr std::size_t max_input_bytes = 4096;

[[noreturn]] void fuzz_abort() noexcept { std::abort(); }

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (data == nullptr || size == 0U || size > max_input_bytes) {
        return 0;
    }

    const std::string fen{reinterpret_cast<const char*>(data), size};
    const auto position = eng::Position::from_fen(fen);
    if (!position.has_value()) {
        return 0;
    }

    if (!position->has_consistent_occupancies()) {
        fuzz_abort();
    }
    if (position->zobrist_key() != position->recompute_zobrist_key()) {
        fuzz_abort();
    }

    const auto serialized = position->to_fen();
    const auto reparsed = eng::Position::from_fen(serialized);
    if (!reparsed.has_value()) {
        fuzz_abort();
    }
    if (reparsed->to_fen() != serialized) {
        fuzz_abort();
    }
    if (!reparsed->has_consistent_occupancies()) {
        fuzz_abort();
    }

    return 0;
}
