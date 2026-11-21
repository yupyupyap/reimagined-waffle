#include <eng/uci.hpp>

#include <eng/bitboard.hpp>
#include <eng/movegen.hpp>
#include <eng/search.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <mutex>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace eng {
namespace {

constexpr int hash_min_mb = 1;
constexpr int hash_max_mb = 65536;
constexpr int threads_min = 1;
constexpr int threads_max = 512;
constexpr int multi_pv_min = 1;
constexpr int multi_pv_max = 256;
constexpr int move_overhead_min_ms = 0;
constexpr int move_overhead_max_ms = 5000;
constexpr int fen_field_count = 6;
constexpr int max_uci_depth = max_search_ply - 1;
constexpr int default_clock_moves_to_go = 30;
constexpr int max_clock_moves_to_go = 200;
constexpr int increment_budget_numerator = 3;
constexpr int increment_budget_denominator = 4;
constexpr std::size_t uci_move_text_length = 4;
constexpr std::size_t uci_promotion_move_text_length = 5;
constexpr std::string_view default_eval_file;

inline constexpr std::array go_keywords{
    std::string_view{"searchmoves"}, std::string_view{"ponder"},   std::string_view{"wtime"},
    std::string_view{"btime"},       std::string_view{"winc"},     std::string_view{"binc"},
    std::string_view{"movestogo"},   std::string_view{"depth"},    std::string_view{"nodes"},
    std::string_view{"mate"},        std::string_view{"movetime"}, std::string_view{"infinite"},
};

struct GoParameters {
    std::vector<std::string_view> searchmoves;
    int depth{default_search_depth};
    std::uint64_t nodes{0};
    int movetime_ms{0};
    int wtime_ms{0};
    int btime_ms{0};
    int winc_ms{0};
    int binc_ms{0};
    int movestogo{0};
    bool has_depth{false};
    bool has_movetime{false};
    bool has_wtime{false};
    bool has_btime{false};
    bool has_winc{false};
    bool has_binc{false};
    bool has_movestogo{false};
    bool infinite{false};
    bool ponder{false};
};

struct TimeBudget {
    std::optional<int> soft_ms;
    std::optional<int> hard_ms;
};

struct SearchInfoSink {
    std::ostream* output{nullptr};
    std::mutex* output_mutex{nullptr};
    SearchResult last_emitted;
    std::uint32_t emitted_iterations{0};
};

enum class GoValueField : std::uint8_t {
    None,
    Depth,
    Nodes,
    MoveTime,
    WhiteTime,
    BlackTime,
    WhiteIncrement,
    BlackIncrement,
    MovesToGo
};

struct GoValueOption {
    std::string_view name;
    GoValueField field{GoValueField::None};
};

struct PruningOptionBinding {
    std::string_view name;
    bool SearchPruningOptions::* flag{nullptr};
};

struct ExtensionOptionBinding {
    std::string_view name;
    bool SearchExtensionOptions::* flag{nullptr};
};

struct ParsedOptionValue {
    std::string_view name;
    std::string_view value;
};

inline constexpr std::array go_value_options{
    GoValueOption{.name = std::string_view{"depth"}, .field = GoValueField::Depth},
    GoValueOption{.name = std::string_view{"nodes"}, .field = GoValueField::Nodes},
    GoValueOption{.name = std::string_view{"movetime"}, .field = GoValueField::MoveTime},
    GoValueOption{.name = std::string_view{"wtime"}, .field = GoValueField::WhiteTime},
    GoValueOption{.name = std::string_view{"btime"}, .field = GoValueField::BlackTime},
    GoValueOption{.name = std::string_view{"winc"}, .field = GoValueField::WhiteIncrement},
    GoValueOption{.name = std::string_view{"binc"}, .field = GoValueField::BlackIncrement},
    GoValueOption{.name = std::string_view{"movestogo"}, .field = GoValueField::MovesToGo},
};

inline constexpr std::array pruning_option_bindings{
    PruningOptionBinding{.name = std::string_view{"nullmovepruning"},
                         .flag = &SearchPruningOptions::null_move_pruning},
    PruningOptionBinding{.name = std::string_view{"latemovereductions"},
                         .flag = &SearchPruningOptions::late_move_reductions},
    PruningOptionBinding{.name = std::string_view{"reversefutilitypruning"},
                         .flag = &SearchPruningOptions::reverse_futility_pruning},
    PruningOptionBinding{.name = std::string_view{"latemovepruning"},
                         .flag = &SearchPruningOptions::late_move_pruning},
    PruningOptionBinding{.name = std::string_view{"futilitypruning"},
                         .flag = &SearchPruningOptions::futility_pruning},
    PruningOptionBinding{.name = std::string_view{"seepruning"},
                         .flag = &SearchPruningOptions::see_pruning},
};

inline constexpr std::array extension_option_bindings{
    ExtensionOptionBinding{.name = std::string_view{"singularextensions"},
                           .flag = &SearchExtensionOptions::singular_extensions},
    ExtensionOptionBinding{.name = std::string_view{"multicut"},
                           .flag = &SearchExtensionOptions::multi_cut},
    ExtensionOptionBinding{.name = std::string_view{"internaliterativereductions"},
                           .flag = &SearchExtensionOptions::internal_iterative_reductions},
    ExtensionOptionBinding{.name = std::string_view{"checkextensions"},
                           .flag = &SearchExtensionOptions::check_extensions},
};

[[nodiscard]] constexpr bool is_ascii_space(char value) noexcept {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r' || value == '\f' ||
           value == '\v';
}

[[nodiscard]] constexpr char to_lower_ascii(char value) noexcept {
    if (value >= 'A' && value <= 'Z') {
        return static_cast<char>(value - 'A' + 'a');
    }
    return value;
}

[[nodiscard]] bool equals_ci(std::string_view lhs, std::string_view rhs) noexcept {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (std::size_t index = 0; index < lhs.size(); ++index) {
        if (to_lower_ascii(lhs[index]) != to_lower_ascii(rhs[index])) {
            return false;
        }
    }

    return true;
}

[[nodiscard]] std::string_view trim(std::string_view text) noexcept {
    while (!text.empty() && is_ascii_space(text.front())) {
        text.remove_prefix(1);
    }
    while (!text.empty() && is_ascii_space(text.back())) {
        text.remove_suffix(1);
    }
    return text;
}

[[nodiscard]] std::vector<std::string_view> split_tokens(std::string_view text) {
    std::vector<std::string_view> tokens;
    text = trim(text);
    while (!text.empty()) {
        std::size_t token_end = 0;
        while (token_end < text.size() && !is_ascii_space(text[token_end])) {
            ++token_end;
        }

        tokens.push_back(text.substr(0, token_end));
        text.remove_prefix(token_end);
        while (!text.empty() && is_ascii_space(text.front())) {
            text.remove_prefix(1);
        }
    }
    return tokens;
}

[[nodiscard]] std::string join_tokens(const std::vector<std::string_view>& tokens,
                                      std::size_t first, std::size_t last) {
    std::string result;
    for (std::size_t index = first; index < last && index < tokens.size(); ++index) {
        if (!result.empty()) {
            result.push_back(' ');
        }
        result.append(tokens[index]);
    }
    return result;
}

[[nodiscard]] std::string normalize_option_name(std::string_view name) {
    std::string result;
    for (const auto symbol : name) {
        if (!is_ascii_space(symbol)) {
            result.push_back(to_lower_ascii(symbol));
        }
    }
    return result;
}

[[nodiscard]] bool parse_int(std::string_view text, int& value) noexcept {
    text = trim(text);
    if (text.empty()) {
        return false;
    }

    int parsed = 0;
    const auto* first = text.data();
    const auto* last = text.data() + text.size();
    const auto [ptr, error] = std::from_chars(first, last, parsed);
    if (error != std::errc{} || ptr != last) {
        return false;
    }

    value = parsed;
    return true;
}

[[nodiscard]] bool parse_uint64(std::string_view text, std::uint64_t& value) noexcept {
    text = trim(text);
    if (text.empty()) {
        return false;
    }

    std::uint64_t parsed = 0;
    const auto* first = text.data();
    const auto* last = text.data() + text.size();
    const auto [ptr, error] = std::from_chars(first, last, parsed);
    if (error != std::errc{} || ptr != last) {
        return false;
    }

    value = parsed;
    return true;
}

[[nodiscard]] bool parse_bool(std::string_view text, bool& value) noexcept {
    text = trim(text);
    if (equals_ci(text, "true") || text == "1") {
        value = true;
        return true;
    }
    if (equals_ci(text, "false") || text == "0") {
        value = false;
        return true;
    }
    return false;
}

bool apply_pruning_option(UciOptions& options, ParsedOptionValue option) noexcept {
    const auto found =
        std::ranges::find_if(pruning_option_bindings, [option](const PruningOptionBinding entry) {
            return option.name == entry.name;
        });
    if (found == pruning_option_bindings.end()) {
        return false;
    }

    bool value = false;
    if (parse_bool(option.value, value)) {
        options.pruning.*(found->flag) = value;
    }
    return true;
}

bool apply_extension_option(UciOptions& options, ParsedOptionValue option) noexcept {
    const auto found = std::ranges::find_if(
        extension_option_bindings,
        [option](const ExtensionOptionBinding entry) { return option.name == entry.name; });
    if (found == extension_option_bindings.end()) {
        return false;
    }

    bool value = false;
    if (parse_bool(option.value, value)) {
        options.extensions.*(found->flag) = value;
    }
    return true;
}

[[nodiscard]] std::optional<NnueInferenceBackend> parse_nnue_backend(std::string_view text) {
    const auto normalized = normalize_option_name(text);
    if (normalized == "scalar") {
        return NnueInferenceBackend::Scalar;
    }
    if (normalized == "neon") {
        return NnueInferenceBackend::Neon;
    }
    if (normalized == "auto") {
        return NnueInferenceBackend::Auto;
    }
    return std::nullopt;
}

[[nodiscard]] bool in_range(int value, int minimum, int maximum) noexcept {
    return value >= minimum && value <= maximum;
}

[[nodiscard]] bool is_go_keyword(std::string_view token) noexcept {
    return std::ranges::any_of(
        go_keywords, [token](std::string_view keyword) { return equals_ci(token, keyword); });
}

[[nodiscard]] std::string normalize_move_text(std::string_view notation) {
    std::string result;
    result.reserve(notation.size());
    for (const auto symbol : notation) {
        result.push_back(to_lower_ascii(symbol));
    }
    return result;
}

[[nodiscard]] char promotion_char(PieceType type) noexcept {
    switch (type) {
    case PieceType::Knight:
        return 'n';
    case PieceType::Bishop:
        return 'b';
    case PieceType::Rook:
        return 'r';
    case PieceType::Queen:
        return 'q';
    case PieceType::Pawn:
    case PieceType::King:
    case PieceType::None:
        break;
    }
    return '\0';
}

[[nodiscard]] std::string square_to_uci(Square square) {
    if (!square.is_valid()) {
        return "??";
    }

    std::string result;
    result.push_back(static_cast<char>('a' + square.file().index()));
    result.push_back(static_cast<char>('1' + square.rank().index()));
    return result;
}

[[nodiscard]] std::vector<std::string_view>
parse_searchmoves(const std::vector<std::string_view>& tokens) {
    std::vector<std::string_view> result;
    for (std::size_t index = 1; index < tokens.size(); ++index) {
        if (!equals_ci(tokens[index], "searchmoves")) {
            continue;
        }

        ++index;
        while (index < tokens.size() && !is_go_keyword(tokens[index])) {
            result.push_back(tokens[index]);
            ++index;
        }
        break;
    }
    return result;
}

[[nodiscard]] GoValueField go_value_field(std::string_view token) noexcept {
    const auto found =
        std::ranges::find_if(go_value_options, [token](const GoValueOption option) {
            return equals_ci(token, option.name);
        });
    if (found == go_value_options.end()) {
        return GoValueField::None;
    }
    return found->field;
}

void apply_go_value(GoParameters& parameters, GoValueField field, std::string_view value) {
    int parsed = 0;
    std::uint64_t parsed_nodes = 0;

    switch (field) {
    case GoValueField::Depth:
        if (parse_int(value, parsed)) {
            parameters.depth = std::clamp(parsed, 1, max_uci_depth);
            parameters.has_depth = true;
        }
        break;
    case GoValueField::Nodes:
        if (parse_uint64(value, parsed_nodes) && parsed_nodes > 0) {
            parameters.nodes = parsed_nodes;
        }
        break;
    case GoValueField::MoveTime:
        if (parse_int(value, parsed) && parsed > 0) {
            parameters.movetime_ms = parsed;
            parameters.has_movetime = true;
        }
        break;
    case GoValueField::WhiteTime:
        if (parse_int(value, parsed) && parsed > 0) {
            parameters.wtime_ms = parsed;
            parameters.has_wtime = true;
        }
        break;
    case GoValueField::BlackTime:
        if (parse_int(value, parsed) && parsed > 0) {
            parameters.btime_ms = parsed;
            parameters.has_btime = true;
        }
        break;
    case GoValueField::WhiteIncrement:
        if (parse_int(value, parsed) && parsed > 0) {
            parameters.winc_ms = parsed;
            parameters.has_winc = true;
        }
        break;
    case GoValueField::BlackIncrement:
        if (parse_int(value, parsed) && parsed > 0) {
            parameters.binc_ms = parsed;
            parameters.has_binc = true;
        }
        break;
    case GoValueField::MovesToGo:
        if (parse_int(value, parsed) && parsed > 0) {
            parameters.movestogo = parsed;
            parameters.has_movestogo = true;
        }
        break;
    case GoValueField::None:
        break;
    }
}

void apply_go_flag(GoParameters& parameters, std::string_view token) noexcept {
    if (equals_ci(token, "infinite")) {
        parameters.infinite = true;
    } else if (equals_ci(token, "ponder")) {
        parameters.ponder = true;
    }
}

[[nodiscard]] GoParameters parse_go_parameters(const std::vector<std::string_view>& tokens) {
    GoParameters result;
    result.searchmoves = parse_searchmoves(tokens);

    for (std::size_t index = 1; index + 1U < tokens.size(); ++index) {
        apply_go_value(result, go_value_field(tokens[index]), tokens[index + 1U]);
    }
    for (std::size_t index = 1; index < tokens.size(); ++index) {
        apply_go_flag(result, tokens[index]);
    }

    return result;
}

[[nodiscard]] bool has_clock_limit(const GoParameters& parameters,
                                   const Position& position) noexcept {
    if (position.side_to_move() == Color::White) {
        return parameters.has_wtime;
    }
    return parameters.has_btime;
}

[[nodiscard]] int selected_clock_ms(const GoParameters& parameters,
                                    const Position& position) noexcept {
    if (position.side_to_move() == Color::White) {
        return parameters.wtime_ms;
    }
    return parameters.btime_ms;
}

[[nodiscard]] int selected_increment_ms(const GoParameters& parameters,
                                        const Position& position) noexcept {
    if (position.side_to_move() == Color::White) {
        return parameters.has_winc ? parameters.winc_ms : 0;
    }
    return parameters.has_binc ? parameters.binc_ms : 0;
}

[[nodiscard]] int choose_search_depth(const GoParameters& parameters,
                                      const Position& position) noexcept {
    if (parameters.has_depth) {
        return parameters.depth;
    }
    if (parameters.nodes != 0 || parameters.has_movetime || has_clock_limit(parameters, position) ||
        parameters.infinite || parameters.ponder) {
        return max_uci_depth;
    }
    return default_search_depth;
}

[[nodiscard]] bool should_use_iterative_search(const GoParameters& parameters,
                                               const Position& position) noexcept {
    return (parameters.has_depth && parameters.depth > 1) || parameters.nodes != 0 ||
           parameters.has_movetime || has_clock_limit(parameters, position) ||
           parameters.infinite || parameters.ponder;
}

[[nodiscard]] TimeBudget compute_time_budget_ms(const GoParameters& parameters,
                                                const Position& position,
                                                const UciOptions& options) noexcept {
    if (parameters.has_movetime) {
        return TimeBudget{.hard_ms =
                              std::max(1, parameters.movetime_ms - options.move_overhead_ms)};
    }
    if (!has_clock_limit(parameters, position)) {
        return {};
    }

    const auto remaining = selected_clock_ms(parameters, position);
    const auto safe_remaining = std::max(1, remaining - options.move_overhead_ms);
    const auto moves_to_go = parameters.has_movestogo
                                 ? std::clamp(parameters.movestogo, 1, max_clock_moves_to_go)
                                 : default_clock_moves_to_go;
    const auto increment = selected_increment_ms(parameters, position);
    const auto budget = (safe_remaining / moves_to_go) +
                        ((increment * increment_budget_numerator) / increment_budget_denominator);
    const auto soft_budget = std::clamp(std::max(1, budget), 1, safe_remaining);
    const auto hard_budget =
        std::clamp(static_cast<int>(std::min<std::int64_t>(
                       safe_remaining, std::max<std::int64_t>(soft_budget, soft_budget * 4LL))),
                   1, safe_remaining);
    return TimeBudget{.soft_ms = soft_budget, .hard_ms = hard_budget};
}

[[nodiscard]] std::vector<Move>
parse_root_search_moves(const Position& position,
                        const std::vector<std::string_view>& searchmoves) {
    std::vector<Move> moves;
    moves.reserve(searchmoves.size());
    for (const auto notation : searchmoves) {
        const auto parsed = move_from_uci(position, notation);
        if (parsed.has_value()) {
            moves.push_back(*parsed);
        }
    }
    return moves;
}

[[nodiscard]] bool command_requests_pending_search(const GoParameters& parameters) noexcept {
    return parameters.infinite || parameters.ponder;
}

[[nodiscard]] SearchLimits make_search_limits(const GoParameters& parameters,
                                              const Position& position, const UciOptions& options,
                                              std::atomic_bool& stop_requested) {
    SearchLimits limits;
    limits.depth = Depth::plies(choose_search_depth(parameters, position));
    limits.nodes = parameters.nodes;
    limits.stop_requested = &stop_requested;
    limits.iterative = should_use_iterative_search(parameters, position);
    limits.threads = options.threads;
    limits.pruning = options.pruning;
    limits.extensions = options.extensions;

    const auto budget = compute_time_budget_ms(parameters, position, options);
    if (budget.soft_ms.has_value() || budget.hard_ms.has_value()) {
        const auto now = std::chrono::steady_clock::now();
        if (budget.soft_ms.has_value()) {
            limits.soft_deadline = now + std::chrono::milliseconds{*budget.soft_ms};
            limits.has_soft_deadline = true;
        }
        if (budget.hard_ms.has_value()) {
            limits.hard_deadline = now + std::chrono::milliseconds{*budget.hard_ms};
            limits.has_hard_deadline = true;
        }
    }

    return limits;
}

[[nodiscard]] bool apply_uci_moves(Position& position, const std::vector<std::string_view>& tokens,
                                   std::size_t first_move_index) {
    for (std::size_t index = first_move_index; index < tokens.size(); ++index) {
        const auto move = move_from_uci(position, tokens[index]);
        if (!move.has_value()) {
            return false;
        }

        StateInfo state;
        if (!position.make_move(*move, state)) {
            return false;
        }
    }
    return true;
}

void emit_search_info(std::ostream& output, const SearchResult& result) {
    output << "info depth " << result.depth.value() << " seldepth " << result.seldepth.value()
           << " time " << result.elapsed_ms << " nodes " << result.nodes << " nps " << result.nps
           << " score cp " << result.score.value();
    if (result.bound == SearchScoreBound::Lower) {
        output << " lowerbound";
    } else if (result.bound == SearchScoreBound::Upper) {
        output << " upperbound";
    }
    output << " hashfull " << result.hashfull_permill;
    if (result.principal_variation_size != 0U) {
        output << " pv";
        for (std::size_t index = 0; index < result.principal_variation_size; ++index) {
            output << ' ' << move_to_uci(result.principal_variation[index]);
        }
    }
    output << '\n';
    output.flush();
}

void emit_iteration_search_info(const SearchResult& result, void* userdata) {
    auto* sink = static_cast<SearchInfoSink*>(userdata);
    if (sink == nullptr || sink->output == nullptr || sink->output_mutex == nullptr) {
        return;
    }

    std::scoped_lock output_lock{*sink->output_mutex};
    emit_search_info(*sink->output, result);
    sink->last_emitted = result;
    ++sink->emitted_iterations;
}

[[nodiscard]] bool should_emit_final_search_info(const SearchInfoSink& sink,
                                                 const SearchResult& result) noexcept {
    return sink.emitted_iterations == 0U || sink.last_emitted.depth != result.depth;
}

} // namespace

std::string move_to_uci(Move move) {
    if (!move.is_valid() || move.is_null()) {
        return "0000";
    }

    auto result = square_to_uci(move.from()) + square_to_uci(move.to());
    if (move.is_promotion()) {
        result.push_back(promotion_char(move.promotion_piece()));
    }
    return result;
}

std::optional<Move> move_from_uci(const Position& position, std::string_view notation) {
    const auto normalized = normalize_move_text(trim(notation));
    if (normalized.size() != uci_move_text_length &&
        normalized.size() != uci_promotion_move_text_length) {
        return std::nullopt;
    }

    const auto legal_moves = generate_legal_moves(position);
    for (const auto move : legal_moves) {
        if (move_to_uci(move) == normalized) {
            return move;
        }
    }

    return std::nullopt;
}

UciSession::UciSession(std::istream& input, std::ostream& output)
    : input_{&input}, output_{&output}, options_{.hash_mb = uci_default_hash_mb,
                                                 .threads = uci_default_threads,
                                                 .multi_pv = uci_default_multi_pv,
                                                 .move_overhead_ms = uci_default_move_overhead_ms,
                                                 .eval_file = std::string{default_eval_file},
                                                 .nnue_backend = NnueInferenceBackend::Scalar,
                                                 .use_nnue = false} {
    const auto tt_resized =
        transposition_table_.resize_mb(static_cast<std::size_t>(uci_default_hash_mb));
    (void)tt_resized;
}

UciSession::~UciSession() noexcept { stop_requested_.store(true, std::memory_order_relaxed); }

void UciSession::run() {
    std::string line;
    while (!quit_requested_ && std::getline(*input_, line)) {
        handle_line(line);
    }
}

void UciSession::wait_for_search() {
    if (search_thread_.joinable()) {
        search_thread_.join();
    }
    stop_requested_.store(false, std::memory_order_relaxed);
}

void UciSession::handle_line(std::string_view line) {
    const auto text = trim(line);
    if (text.empty()) {
        return;
    }

    const auto tokens = split_tokens(text);
    if (tokens.empty()) {
        return;
    }

    const auto command = tokens.front();
    if (equals_ci(command, "uci")) {
        handle_uci();
    } else if (equals_ci(command, "debug")) {
        handle_debug(text);
    } else if (equals_ci(command, "isready")) {
        init_bitboards();
        emit("readyok");
    } else if (equals_ci(command, "setoption")) {
        handle_setoption(text);
    } else if (equals_ci(command, "ucinewgame")) {
        finish_pending_search();
        position_ = Position::startpos();
        transposition_table_.clear();
        has_pending_search_ = false;
        pending_bestmove_ = Move::none();
    } else if (equals_ci(command, "position")) {
        handle_position(text);
    } else if (equals_ci(command, "go")) {
        handle_go(text);
    } else if (equals_ci(command, "stop")) {
        handle_stop();
    } else if (equals_ci(command, "quit")) {
        finish_pending_search();
        quit_requested_ = true;
    }
}

void UciSession::handle_uci() {
    emit("id name LocalChessEngine");
    emit("id author LocalChessEngine contributors");
    emit("option name Hash type spin default 16 min 1 max 65536");
    emit("option name Threads type spin default 10 min 1 max 512");
    emit("option name MultiPV type spin default 1 min 1 max 256");
    emit("option name Move Overhead type spin default 10 min 0 max 5000");
    emit("option name Use NNUE type check default false");
    emit("option name EvalFile type string default ");
    emit("option name NNUE Backend type combo default Scalar var Scalar var Neon var Auto");
    emit("option name Null Move Pruning type check default false");
    emit("option name Late Move Reductions type check default false");
    emit("option name Reverse Futility Pruning type check default false");
    emit("option name Late Move Pruning type check default false");
    emit("option name Futility Pruning type check default false");
    emit("option name SEE Pruning type check default false");
    emit("option name Singular Extensions type check default false");
    emit("option name Multi Cut type check default false");
    emit("option name Internal Iterative Reductions type check default false");
    emit("option name Check Extensions type check default false");
    emit("uciok");
}

void UciSession::handle_debug(std::string_view line) {
    const auto tokens = split_tokens(line);
    if (tokens.size() < 2U) {
        return;
    }
    if (equals_ci(tokens[1], "on")) {
        debug_enabled_ = true;
    } else if (equals_ci(tokens[1], "off")) {
        debug_enabled_ = false;
    }
}

void UciSession::handle_setoption(std::string_view line) {
    const auto tokens = split_tokens(line);
    if (tokens.size() < 4U) {
        return;
    }

    std::size_t name_index = tokens.size();
    std::size_t value_index = tokens.size();
    for (std::size_t index = 1; index < tokens.size(); ++index) {
        if (name_index == tokens.size() && equals_ci(tokens[index], "name")) {
            name_index = index;
        } else if (name_index != tokens.size() && equals_ci(tokens[index], "value")) {
            value_index = index;
            break;
        }
    }

    if (name_index + 1U >= tokens.size()) {
        return;
    }

    const auto name_end = value_index == tokens.size() ? tokens.size() : value_index;
    const auto option_name = normalize_option_name(join_tokens(tokens, name_index + 1U, name_end));
    if (option_name.empty() || value_index + 1U >= tokens.size()) {
        return;
    }

    const auto value_text = join_tokens(tokens, value_index + 1U, tokens.size());

    if (apply_pruning_option(options_,
                             ParsedOptionValue{.name = option_name, .value = value_text})) {
        return;
    }
    if (apply_extension_option(options_,
                               ParsedOptionValue{.name = option_name, .value = value_text})) {
        return;
    }

    if (apply_nnue_option(OptionValue{.name = option_name, .value = value_text})) {
        return;
    }

    int value = 0;
    if (!parse_int(value_text, value)) {
        return;
    }

    if (option_name == "hash" && in_range(value, hash_min_mb, hash_max_mb)) {
        finish_pending_search();
        if (transposition_table_.resize_mb(static_cast<std::size_t>(value))) {
            options_.hash_mb = value;
        }
    } else if (option_name == "threads" && in_range(value, threads_min, threads_max)) {
        options_.threads = value;
    } else if (option_name == "multipv" && in_range(value, multi_pv_min, multi_pv_max)) {
        options_.multi_pv = value;
    } else if (option_name == "moveoverhead" &&
               in_range(value, move_overhead_min_ms, move_overhead_max_ms)) {
        options_.move_overhead_ms = value;
    }
}

bool UciSession::apply_nnue_option(OptionValue option) {
    if (option.name == "usennue") {
        bool value = false;
        if (parse_bool(option.value, value)) {
            finish_pending_search();
            options_.use_nnue = value;
        }
        return true;
    }
    if (option.name == "evalfile") {
        finish_pending_search();
        options_.eval_file = option.value;
        auto loaded = NnueNetwork::load(options_.eval_file);
        if (loaded.has_value()) {
            nnue_network_ = std::move(*loaded);
        } else {
            nnue_network_ = NnueNetwork{};
        }
        return true;
    }
    if (option.name == "nnuebackend") {
        if (const auto backend = parse_nnue_backend(option.value); backend.has_value()) {
            finish_pending_search();
            options_.nnue_backend = *backend;
        }
        return true;
    }
    return false;
}

void UciSession::handle_position(std::string_view line) {
    finish_pending_search();

    const auto tokens = split_tokens(line);
    if (tokens.size() < 2U) {
        return;
    }

    Position candidate;
    std::size_t next_token = 2;
    if (equals_ci(tokens[1], "startpos")) {
        candidate = Position::startpos();
    } else if (equals_ci(tokens[1], "fen")) {
        if (tokens.size() < 2U + fen_field_count) {
            return;
        }

        const auto fen = join_tokens(tokens, 2, 2U + fen_field_count);
        const auto parsed = Position::from_fen(fen);
        if (!parsed.has_value()) {
            return;
        }

        candidate = *parsed;
        next_token = 2U + fen_field_count;
    } else {
        return;
    }

    if (next_token < tokens.size()) {
        if (!equals_ci(tokens[next_token], "moves")) {
            return;
        }
        if (!apply_uci_moves(candidate, tokens, next_token + 1U)) {
            return;
        }
    }

    position_ = candidate;
}

void UciSession::handle_go(std::string_view line) {
    finish_pending_search();
    init_bitboards();

    const auto tokens = split_tokens(line);
    const auto parameters = parse_go_parameters(tokens);
    auto root_moves = parse_root_search_moves(position_, parameters.searchmoves);
    auto limits = make_search_limits(parameters, position_, options_, stop_requested_);
    if (options_.use_nnue && nnue_network_.is_loaded()) {
        limits.nnue_network = &nnue_network_;
        limits.nnue_backend = options_.nnue_backend;
    }
    transposition_table_.new_search();
    limits.transposition_table = &transposition_table_;
    auto search_position = position_;
    const auto restrict_root_moves = !parameters.searchmoves.empty();
    const auto defer_bestmove = command_requests_pending_search(parameters);

    stop_requested_.store(false, std::memory_order_relaxed);
    {
        std::scoped_lock lock{search_mutex_};
        pending_bestmove_ = Move::none();
        has_pending_search_ = false;
    }

    search_thread_ =
        std::jthread{[this, search_position, limits, root_moves = std::move(root_moves),
                      restrict_root_moves, defer_bestmove]() mutable {
            SearchInfoSink info_sink{.output = output_, .output_mutex = &output_mutex_};
            auto search_limits = limits;
            search_limits.iteration_callback = emit_iteration_search_info;
            search_limits.iteration_callback_userdata = &info_sink;
            const auto result =
                restrict_root_moves
                    ? search(search_position, search_limits,
                             std::span<const Move>{root_moves.data(), root_moves.size()})
                    : search(search_position, search_limits);
            auto final_result = result;
            if (defer_bestmove && final_result.stop_reason == SearchStopReason::None) {
                while (!stop_requested_.load(std::memory_order_relaxed)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds{1});
                }
                final_result.stopped = true;
                final_result.stop_reason = SearchStopReason::ExternalStop;
            }
            if (should_emit_final_search_info(info_sink, final_result)) {
                std::scoped_lock output_lock{output_mutex_};
                emit_search_info(*output_, final_result);
            }

            if (defer_bestmove) {
                std::scoped_lock lock{search_mutex_};
                pending_bestmove_ = final_result.best_move;
                has_pending_search_ = true;
                return;
            }

            emit_bestmove(final_result.best_move);
        }};
}

void UciSession::handle_stop() { finish_pending_search(); }

void UciSession::finish_pending_search() {
    request_stop_and_join();

    Move bestmove = Move::none();
    bool should_emit_bestmove = false;
    {
        std::scoped_lock lock{search_mutex_};
        if (has_pending_search_) {
            bestmove = pending_bestmove_;
            pending_bestmove_ = Move::none();
            has_pending_search_ = false;
            should_emit_bestmove = true;
        }
    }

    if (should_emit_bestmove) {
        emit_bestmove(bestmove);
    }
}

void UciSession::request_stop_and_join() {
    stop_requested_.store(true, std::memory_order_relaxed);
    if (search_thread_.joinable()) {
        search_thread_.join();
    }
    stop_requested_.store(false, std::memory_order_relaxed);
}

void UciSession::emit(std::string_view line) {
    std::scoped_lock output_lock{output_mutex_};
    *output_ << line << '\n';
    output_->flush();
}

void UciSession::emit_bestmove(Move move) { emit("bestmove " + move_to_uci(move)); }

int run_uci(std::istream& input, std::ostream& output) {
    UciSession session{input, output};
    session.run();
    return 0;
}

} // namespace eng
