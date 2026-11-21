#include <eng/nnue_file.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace {

constexpr std::size_t max_input_bytes = 4096;
constexpr std::uint64_t max_full_read_bytes = 4ULL * 1024ULL * 1024ULL;

[[nodiscard]] std::filesystem::path temp_network_path() {
    static std::atomic_uint64_t counter{0};
    const auto file_id = counter.fetch_add(1, std::memory_order_relaxed);
    return std::filesystem::temp_directory_path() /
           ("localchessengine-fuzz-nnue-" + std::to_string(file_id) + ".nnue");
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (data == nullptr || size == 0U || size > max_input_bytes) {
        return 0;
    }

    const auto path = temp_network_path();
    {
        std::ofstream output(path, std::ios::binary);
        if (!output) {
            return 0;
        }
        output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
        if (!output) {
            std::error_code remove_error;
            std::filesystem::remove(path, remove_error);
            return 0;
        }
    }

    const auto info = eng::read_nnue_network_info(path);
    if (info.has_value() && info->file_bytes <= max_full_read_bytes) {
        const auto network = eng::read_nnue_network_file(path);
        (void)network;
    }

    std::error_code remove_error;
    std::filesystem::remove(path, remove_error);
    return 0;
}
