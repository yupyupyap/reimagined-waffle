#ifndef ENG_UCI_HPP
#define ENG_UCI_HPP

#include "eng/move.hpp"
#include "eng/nnue.hpp"
#include "eng/position.hpp"
#include "eng/search.hpp"
#include "eng/tt.hpp"

#include <atomic>
#include <iosfwd>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace eng {

inline constexpr int uci_default_hash_mb = 16;
inline constexpr int uci_default_threads = 10;
inline constexpr int uci_default_multi_pv = 1;
inline constexpr int uci_default_move_overhead_ms = 10;

struct UciOptions {
    int hash_mb{uci_default_hash_mb};
    int threads{uci_default_threads};
    int multi_pv{uci_default_multi_pv};
    int move_overhead_ms{uci_default_move_overhead_ms};
    std::string eval_file;
    NnueInferenceBackend nnue_backend{NnueInferenceBackend::Scalar};
    bool use_nnue{false};
    SearchPruningOptions pruning;
    SearchExtensionOptions extensions;

    friend bool operator==(const UciOptions& lhs, const UciOptions& rhs) = default;
};

[[nodiscard]] std::string move_to_uci(Move move);
[[nodiscard]] std::optional<Move> move_from_uci(const Position& position,
                                                std::string_view notation);

class UciSession {
  public:
    UciSession(std::istream& input, std::ostream& output);
    ~UciSession() noexcept;

    UciSession(const UciSession&) = delete;
    UciSession& operator=(const UciSession&) = delete;
    UciSession(UciSession&&) = delete;
    UciSession& operator=(UciSession&&) = delete;

    void run();
    void handle_line(std::string_view line);
    void wait_for_search();

    [[nodiscard]] const UciOptions& options() const noexcept { return options_; }
    [[nodiscard]] const Position& position() const noexcept { return position_; }
    [[nodiscard]] const TranspositionTable& transposition_table() const noexcept {
        return transposition_table_;
    }

  private:
    struct OptionValue {
        std::string_view name;
        std::string_view value;
    };

    void handle_uci();
    void handle_debug(std::string_view line);
    void handle_setoption(std::string_view line);
    void handle_position(std::string_view line);
    void handle_go(std::string_view line);
    void handle_stop();
    bool apply_nnue_option(OptionValue option);
    void finish_pending_search();
    void request_stop_and_join();

    void emit(std::string_view line);
    void emit_bestmove(Move move);

    std::istream* input_;
    std::ostream* output_;
    Position position_{Position::startpos()};
    UciOptions options_;
    TranspositionTable transposition_table_;
    NnueNetwork nnue_network_;
    std::atomic_bool stop_requested_{false};
    std::jthread search_thread_;
    std::mutex output_mutex_;
    std::mutex search_mutex_;
    Move pending_bestmove_{Move::none()};
    bool debug_enabled_{false};
    bool has_pending_search_{false};
    bool quit_requested_{false};
};

int run_uci(std::istream& input, std::ostream& output);

} // namespace eng

#endif // ENG_UCI_HPP
