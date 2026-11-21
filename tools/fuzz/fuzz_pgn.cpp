#include "pgn.hpp"

#include <eng/bitboard.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace {

constexpr std::size_t max_input_bytes = 4096;

[[noreturn]] void fuzz_abort() noexcept { std::abort(); }

} // namespace

// NOLINTNEXTLINE(readability-non-const-parameter): libFuzzer requires this signature.
extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
    (void)argc;
    (void)argv;
    eng::init_bitboards();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (data == nullptr || size == 0U || size > max_input_bytes) {
        return 0;
    }

    const std::string pgn{reinterpret_cast<const char*>(data), size};
    const auto game = gui::pgn::parse_game(pgn);
    if (!game.has_value()) {
        return 0;
    }

    const auto exported = gui::pgn::write_game(*game);
    if (!exported.has_value()) {
        fuzz_abort();
    }

    const auto reparsed = gui::pgn::parse_game(*exported);
    if (!reparsed.has_value()) {
        fuzz_abort();
    }
    if (reparsed->moves != game->moves || reparsed->result != game->result) {
        fuzz_abort();
    }

    return 0;
}
