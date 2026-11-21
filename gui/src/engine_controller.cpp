#include "engine_controller.hpp"

#include <QFileInfo>
#include <QRegularExpression>

#include <algorithm>

namespace {

#ifdef __has_feature
#if __has_feature(thread_sanitizer)
#define LCE_ENGINE_CONTROLLER_THREAD_SANITIZER 1
#endif
#endif
#ifdef __SANITIZE_THREAD__
#define LCE_ENGINE_CONTROLLER_THREAD_SANITIZER 1
#endif

#ifdef LCE_ENGINE_CONTROLLER_THREAD_SANITIZER
constexpr int kHandshakeTimeoutMs = 90000;
#else
constexpr int kHandshakeTimeoutMs = 15000;
#endif
constexpr int kShutdownTimeoutMs = 2000;
constexpr int kDestructorWaitMs = 500;
constexpr int kMaxGoDepth = 64;
constexpr int kMaxGoMovetimeMs = 24 * 60 * 60 * 1000;
constexpr int kHashMinMb = 1;
constexpr int kHashMaxMb = 65536;
constexpr int kThreadsMin = 1;
constexpr int kThreadsMax = 512;
constexpr int kMultiPvMin = 1;
constexpr int kMultiPvMax = 256;
constexpr int kMoveOverheadMinMs = 0;
constexpr int kMoveOverheadMaxMs = 5000;
constexpr qsizetype kMaxProtocolLineLength = 8192;
constexpr qsizetype kMaxProtocolBufferLength = 32768;

[[nodiscard]] QString trimmed_line(QString line) {
    if (line.endsWith(QLatin1Char('\r'))) {
        line.chop(1);
    }
    return line.trimmed();
}

[[nodiscard]] bool is_running_state(EngineController::State state) noexcept {
    return state == EngineController::State::Starting || state == EngineController::State::Ready ||
           state == EngineController::State::Searching ||
           state == EngineController::State::Stopping;
}

[[nodiscard]] QString process_error_name(QProcess::ProcessError error) {
    switch (error) {
    case QProcess::FailedToStart:
        return QStringLiteral("failed_to_start");
    case QProcess::Crashed:
        return QStringLiteral("crashed");
    case QProcess::Timedout:
        return QStringLiteral("timed_out");
    case QProcess::WriteError:
        return QStringLiteral("write_error");
    case QProcess::ReadError:
        return QStringLiteral("read_error");
    case QProcess::UnknownError:
        return QStringLiteral("unknown_error");
    }
    return QStringLiteral("unknown_error");
}

[[nodiscard]] bool parse_int_token(const QString& token, int& value) noexcept {
    bool parsed_success = false;
    const auto parsed = token.toInt(&parsed_success);
    if (!parsed_success) {
        return false;
    }
    value = parsed;
    return true;
}

[[nodiscard]] bool parse_qint64_token(const QString& token, qint64& value) noexcept {
    bool parsed_success = false;
    const auto parsed = token.toLongLong(&parsed_success);
    if (!parsed_success) {
        return false;
    }
    value = parsed;
    return true;
}

[[nodiscard]] bool parse_quint64_token(const QString& token, quint64& value) noexcept {
    bool parsed_success = false;
    const auto parsed = token.toULongLong(&parsed_success);
    if (!parsed_success) {
        return false;
    }
    value = parsed;
    return true;
}

[[nodiscard]] bool is_uci_move_token(const QString& token) {
    static const QRegularExpression move_pattern{
        QStringLiteral("^(0000|[a-h][1-8][a-h][1-8][nbrq]?)$")};
    return move_pattern.match(token.toLower()).hasMatch();
}

[[nodiscard]] QString normalized_nnue_backend(QString backend) {
    backend = backend.trimmed();
    if (backend.compare(QStringLiteral("Neon"), Qt::CaseInsensitive) == 0) {
        return QStringLiteral("Neon");
    }
    if (backend.compare(QStringLiteral("Auto"), Qt::CaseInsensitive) == 0) {
        return QStringLiteral("Auto");
    }
    return QStringLiteral("Scalar");
}

[[nodiscard]] QString single_line_value(QString value) {
    value = value.trimmed();
    value.replace(QLatin1Char('\n'), QLatin1Char(' '));
    value.replace(QLatin1Char('\r'), QLatin1Char(' '));
    return value.trimmed();
}

[[nodiscard]] EngineController::EngineOptions
sanitized_options(EngineController::EngineOptions options) {
    options.hash_mb = std::clamp(options.hash_mb, kHashMinMb, kHashMaxMb);
    options.threads = std::clamp(options.threads, kThreadsMin, kThreadsMax);
    options.multi_pv = std::clamp(options.multi_pv, kMultiPvMin, kMultiPvMax);
    options.move_overhead_ms =
        std::clamp(options.move_overhead_ms, kMoveOverheadMinMs, kMoveOverheadMaxMs);
    options.eval_file = single_line_value(options.eval_file);
    options.nnue_backend = normalized_nnue_backend(options.nnue_backend);
    return options;
}

[[nodiscard]] QString bounded_detail(QString detail) {
    constexpr qsizetype max_detail = 240;
    detail.replace(QLatin1Char('\n'), QLatin1Char(' '));
    detail.replace(QLatin1Char('\r'), QLatin1Char(' '));
    if (detail.size() > max_detail) {
        detail.truncate(max_detail);
        detail.append(QStringLiteral("..."));
    }
    return detail;
}

} // namespace

EngineController::EngineController(QObject* parent)
    : QObject(parent), process_{this}, handshake_timer_{this}, shutdown_timer_{this} {
    qRegisterMetaType<EngineController::State>();
    qRegisterMetaType<EngineController::SearchInfo>();
    qRegisterMetaType<EngineController::BestMove>();

    handshake_timer_.setSingleShot(true);
    handshake_timer_.setInterval(kHandshakeTimeoutMs);
    shutdown_timer_.setSingleShot(true);
    shutdown_timer_.setInterval(kShutdownTimeoutMs);

    wire_process_signals();
    connect(&handshake_timer_, &QTimer::timeout, this, [this]() { handle_handshake_timeout(); });
    connect(&shutdown_timer_, &QTimer::timeout, this, [this]() { handle_shutdown_timeout(); });
}

EngineController::~EngineController() {
    handshake_timer_.stop();
    shutdown_timer_.stop();
    if (process_.state() == QProcess::NotRunning) {
        return;
    }

    process_.write("quit\n");
    process_.closeWriteChannel();
    if (!process_.waitForFinished(kDestructorWaitMs)) {
        process_.kill();
        process_.waitForFinished(kDestructorWaitMs);
    }
}

void EngineController::set_engine_path(const QString& path) { engine_path_ = path.trimmed(); }

void EngineController::set_engine_options(const EngineOptions& options) {
    const EngineOptions sanitized = sanitized_options(options);
    if (engine_options_ == sanitized && !engine_options_dirty_) {
        return;
    }

    engine_options_ = sanitized;
    engine_options_dirty_ = true;
    if (state_ == State::Ready) {
        (void)apply_engine_options_if_ready();
    }
}

bool EngineController::start_engine() { return start_engine(engine_path_); }

bool EngineController::start_engine(const QString& path) {
    if (is_running()) {
        return true;
    }

    set_engine_path(path);
    if (engine_path_.isEmpty()) {
        fail_controller(QStringLiteral("Engine path is empty."), false);
        return false;
    }

    const QFileInfo engine_info{engine_path_};
    if (!engine_info.exists() || !engine_info.isFile() || !engine_info.isExecutable()) {
        fail_controller(QStringLiteral("Engine executable is not available: %1").arg(engine_path_),
                        false);
        return false;
    }

    clear_protocol_state();
    intentional_shutdown_ = false;
    engine_options_dirty_ = true;
    process_.setProgram(engine_path_);
    process_.setArguments({});
    process_.setProcessChannelMode(QProcess::SeparateChannels);
    set_state(State::Starting);
    emit engineEvent(QStringLiteral("start"), engine_path_);
    process_.start(QIODevice::ReadWrite);
    handshake_timer_.start();
    return true;
}

bool EngineController::restart_engine(const QString& path) {
    if (!path.trimmed().isEmpty()) {
        set_engine_path(path);
    }
    if (is_running()) {
        quit_engine();
    }
    return start_engine();
}

void EngineController::quit_engine() {
    handshake_timer_.stop();
    if (process_.state() == QProcess::NotRunning) {
        clear_protocol_state();
        intentional_shutdown_ = false;
        set_state(State::Stopped);
        return;
    }

    intentional_shutdown_ = true;
    set_state(State::Stopping);
    (void)send_command(QStringLiteral("quit"));
    process_.closeWriteChannel();
    shutdown_timer_.start();
}

void EngineController::stop_search() {
    if (state_ != State::Searching) {
        return;
    }
    set_state(State::Stopping);
    (void)send_command(QStringLiteral("stop"));
}

void EngineController::new_game() {
    if (!is_running()) {
        return;
    }
    if (state_ == State::Searching) {
        stop_search();
    }
    (void)send_command(QStringLiteral("ucinewgame"));
}

bool EngineController::set_position_fen(const QString& fen) {
    if (state_ != State::Ready || fen.contains(QLatin1Char('\n')) ||
        fen.contains(QLatin1Char('\r')) || fen.trimmed().isEmpty()) {
        return false;
    }
    if (!apply_engine_options_if_ready()) {
        return false;
    }
    return send_command(QStringLiteral("position fen %1").arg(fen.trimmed()));
}

bool EngineController::go_depth(int depth) {
    if (state_ != State::Ready) {
        return false;
    }
    if (!apply_engine_options_if_ready()) {
        return false;
    }
    const int safe_depth = std::clamp(depth, 1, kMaxGoDepth);
    if (!send_command(QStringLiteral("go depth %1").arg(safe_depth))) {
        return false;
    }
    set_state(State::Searching);
    return true;
}

bool EngineController::go_infinite() {
    if (state_ != State::Ready) {
        return false;
    }
    if (!apply_engine_options_if_ready()) {
        return false;
    }
    if (!send_command(QStringLiteral("go infinite"))) {
        return false;
    }
    set_state(State::Searching);
    return true;
}

bool EngineController::go_nodes(quint64 nodes) {
    if (state_ != State::Ready || nodes == 0U) {
        return false;
    }
    if (!apply_engine_options_if_ready()) {
        return false;
    }
    if (!send_command(QStringLiteral("go nodes %1").arg(nodes))) {
        return false;
    }
    set_state(State::Searching);
    return true;
}

bool EngineController::go_movetime(int movetime_ms) {
    if (state_ != State::Ready) {
        return false;
    }
    if (!apply_engine_options_if_ready()) {
        return false;
    }
    const int safe_movetime = std::clamp(movetime_ms, 1, kMaxGoMovetimeMs);
    if (!send_command(QStringLiteral("go movetime %1").arg(safe_movetime))) {
        return false;
    }
    set_state(State::Searching);
    return true;
}

bool EngineController::is_running() const noexcept {
    return process_.state() != QProcess::NotRunning && is_running_state(state_);
}

EngineController::SearchInfo EngineController::parse_search_info_line(const QString& line) {
    SearchInfo info;
    (void)parse_search_info_line(line, info);
    return info;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool EngineController::parse_search_info_line(const QString& line, SearchInfo& info) {
    const QString normalized = trimmed_line(line);
    const QStringList tokens = normalized.split(QLatin1Char(' '), Qt::SkipEmptyParts);
    if (tokens.isEmpty() || tokens.first() != QStringLiteral("info")) {
        return false;
    }

    SearchInfo parsed;
    parsed.raw_line = normalized;
    for (qsizetype index = 1; index < tokens.size(); ++index) {
        const QString& token = tokens.at(index);
        if (token == QStringLiteral("string")) {
            break;
        }
        if (token == QStringLiteral("depth") && index + 1 < tokens.size()) {
            (void)parse_int_token(tokens.at(++index), parsed.depth);
        } else if (token == QStringLiteral("seldepth") && index + 1 < tokens.size()) {
            (void)parse_int_token(tokens.at(++index), parsed.seldepth);
        } else if (token == QStringLiteral("time") && index + 1 < tokens.size()) {
            (void)parse_qint64_token(tokens.at(++index), parsed.time_ms);
        } else if (token == QStringLiteral("nodes") && index + 1 < tokens.size()) {
            (void)parse_quint64_token(tokens.at(++index), parsed.nodes);
        } else if (token == QStringLiteral("nps") && index + 1 < tokens.size()) {
            (void)parse_qint64_token(tokens.at(++index), parsed.nps);
        } else if (token == QStringLiteral("multipv") && index + 1 < tokens.size()) {
            int multipv = 1;
            if (parse_int_token(tokens.at(++index), multipv) && multipv > 0) {
                parsed.multipv = multipv;
            }
        } else if (token == QStringLiteral("hashfull") && index + 1 < tokens.size()) {
            (void)parse_int_token(tokens.at(++index), parsed.hashfull);
        } else if (token == QStringLiteral("lowerbound")) {
            parsed.bound = ScoreBound::Lower;
        } else if (token == QStringLiteral("upperbound")) {
            parsed.bound = ScoreBound::Upper;
        } else if (token == QStringLiteral("score") && index + 2 < tokens.size()) {
            const QString& score_type = tokens.at(++index);
            int score_value = 0;
            if (parse_int_token(tokens.at(++index), score_value)) {
                if (score_type == QStringLiteral("cp")) {
                    parsed.has_score_cp = true;
                    parsed.score_cp = score_value;
                } else if (score_type == QStringLiteral("mate")) {
                    parsed.has_mate = true;
                    parsed.mate = score_value;
                }
            }
        } else if (token == QStringLiteral("pv")) {
            while (index + 1 < tokens.size()) {
                const QString move = tokens.at(++index).toLower();
                if (!is_uci_move_token(move) || move == QStringLiteral("0000")) {
                    break;
                }
                parsed.pv.push_back(move);
            }
        }
    }

    info = parsed;
    return true;
}

bool EngineController::parse_bestmove_line(const QString& line, BestMove& bestmove) {
    const QString normalized = trimmed_line(line);
    const QStringList tokens = normalized.split(QLatin1Char(' '), Qt::SkipEmptyParts);
    if (tokens.size() < 2 || tokens.first() != QStringLiteral("bestmove")) {
        return false;
    }

    const QString move = tokens.at(1).toLower();
    if (!is_uci_move_token(move)) {
        return false;
    }

    BestMove parsed{.move = move};
    if (tokens.size() >= 4 && tokens.at(2) == QStringLiteral("ponder")) {
        const QString ponder = tokens.at(3).toLower();
        if (!is_uci_move_token(ponder) || ponder == QStringLiteral("0000")) {
            return false;
        }
        parsed.ponder = ponder;
    }

    bestmove = parsed;
    return true;
}

EngineController::LineExtractionResult
EngineController::extract_complete_lines(const QString& previous, const QString& chunk) {
    QString combined = previous + chunk;
    LineExtractionResult result;
    while (true) {
        const qsizetype newline_index = combined.indexOf(QLatin1Char('\n'));
        if (newline_index < 0) {
            break;
        }

        QString line = combined.left(newline_index);
        combined.remove(0, newline_index + 1);
        if (line.endsWith(QLatin1Char('\r'))) {
            line.chop(1);
        }
        if (line.size() > kMaxProtocolLineLength) {
            result.overflow = true;
            result.remaining.clear();
            return result;
        }
        result.lines.push_back(line);
    }

    if (combined.size() > kMaxProtocolBufferLength || combined.size() > kMaxProtocolLineLength) {
        result.overflow = true;
        result.remaining.clear();
        return result;
    }
    result.remaining = combined;
    return result;
}

QString EngineController::state_name(State state) {
    switch (state) {
    case State::Stopped:
        return QStringLiteral("Stopped");
    case State::Starting:
        return QStringLiteral("Starting");
    case State::Ready:
        return QStringLiteral("Ready");
    case State::Searching:
        return QStringLiteral("Searching");
    case State::Stopping:
        return QStringLiteral("Stopping");
    case State::Failed:
        return QStringLiteral("Failed");
    }
    return QStringLiteral("Unknown");
}

void EngineController::wire_process_signals() {
    connect(&process_, &QProcess::started, this, [this]() { handle_process_started(); });
    connect(&process_, &QProcess::readyReadStandardOutput, this,
            [this]() { handle_stdout_ready(); });
    connect(&process_, &QProcess::readyReadStandardError, this,
            [this]() { handle_stderr_ready(); });
    connect(&process_, &QProcess::errorOccurred, this,
            [this](QProcess::ProcessError error) { handle_process_error(error); });
    connect(&process_, &QProcess::finished, this,
            [this](int exit_code, QProcess::ExitStatus exit_status) {
                handle_process_finished(exit_code, exit_status);
            });
}

void EngineController::set_state(State state) {
    if (state_ == state) {
        return;
    }
    state_ = state;
    emit stateChanged(state_);
    emit engineEvent(QStringLiteral("state"), state_name(state_));
}

void EngineController::send_uci_handshake() { (void)send_command(QStringLiteral("uci")); }

bool EngineController::send_command(const QString& command) {
    if (process_.state() == QProcess::NotRunning || command.contains(QLatin1Char('\n')) ||
        command.contains(QLatin1Char('\r'))) {
        return false;
    }

    const QByteArray bytes = command.toUtf8() + '\n';
    const auto written = process_.write(bytes);
    if (written != bytes.size()) {
        emit errorOccurred(QStringLiteral("Failed to write UCI command to engine."));
        emit engineEvent(QStringLiteral("write_error"), bounded_detail(command));
        return false;
    }
    emit engineEvent(QStringLiteral("command"), bounded_detail(command));
    return true;
}

bool EngineController::send_setoption(const QString& name, const QString& value) {
    if (name.trimmed().isEmpty() || value.contains(QLatin1Char('\n')) ||
        value.contains(QLatin1Char('\r'))) {
        return false;
    }
    return send_command(QStringLiteral("setoption name %1 value %2").arg(name, value));
}

bool EngineController::apply_engine_options_if_ready() {
    if (!engine_options_dirty_) {
        return true;
    }
    if (state_ != State::Ready) {
        return false;
    }

    const EngineOptions options = sanitized_options(engine_options_);
    bool applied = true;
    applied = send_setoption(QStringLiteral("Hash"), QString::number(options.hash_mb)) && applied;
    applied =
        send_setoption(QStringLiteral("Threads"), QString::number(options.threads)) && applied;
    applied =
        send_setoption(QStringLiteral("MultiPV"), QString::number(options.multi_pv)) && applied;
    applied = send_setoption(QStringLiteral("Move Overhead"),
                             QString::number(options.move_overhead_ms)) &&
              applied;
    applied = send_setoption(QStringLiteral("NNUE Backend"), options.nnue_backend) && applied;
    if (!options.eval_file.isEmpty()) {
        applied = send_setoption(QStringLiteral("EvalFile"), options.eval_file) && applied;
    }
    applied = send_setoption(QStringLiteral("Use NNUE"),
                             options.use_nnue ? QStringLiteral("true") : QStringLiteral("false")) &&
              applied;

    if (applied) {
        engine_options_ = options;
        engine_options_dirty_ = false;
    }
    return applied;
}

void EngineController::handle_stdout_ready() {
    const QString chunk = QString::fromUtf8(process_.readAllStandardOutput());
    const auto extracted = extract_complete_lines(stdout_buffer_, chunk);
    stdout_buffer_ = extracted.remaining;
    if (extracted.overflow) {
        fail_controller(QStringLiteral("Engine stdout protocol line exceeded safety limit."), true);
        return;
    }

    for (const auto& line : extracted.lines) {
        handle_stdout_line(line);
    }
}

void EngineController::handle_stderr_ready() {
    const QString chunk = QString::fromUtf8(process_.readAllStandardError());
    const auto extracted = extract_complete_lines(stderr_buffer_, chunk);
    stderr_buffer_ = extracted.remaining;
    if (extracted.overflow) {
        fail_controller(QStringLiteral("Engine stderr protocol line exceeded safety limit."), true);
        return;
    }

    for (const auto& line : extracted.lines) {
        if (!line.trimmed().isEmpty()) {
            emit engineEvent(QStringLiteral("stderr"), bounded_detail(line));
        }
    }
}

void EngineController::handle_stdout_line(const QString& line) {
    const QString normalized = trimmed_line(line);
    if (normalized.isEmpty()) {
        return;
    }
    emit engineEvent(QStringLiteral("stdout"), bounded_detail(normalized));

    if (normalized == QStringLiteral("uciok")) {
        saw_uciok_ = true;
        (void)send_command(QStringLiteral("isready"));
        return;
    }
    if (normalized == QStringLiteral("readyok")) {
        saw_readyok_ = true;
        handshake_timer_.stop();
        if (state_ == State::Starting || state_ == State::Ready) {
            set_state(State::Ready);
        }
        (void)apply_engine_options_if_ready();
        return;
    }
    if (normalized.startsWith(QStringLiteral("info "))) {
        SearchInfo info;
        if (parse_search_info_line(normalized, info)) {
            emit searchInfoReceived(info);
        }
        return;
    }
    if (normalized.startsWith(QStringLiteral("bestmove "))) {
        BestMove bestmove;
        if (!parse_bestmove_line(normalized, bestmove)) {
            emit errorOccurred(QStringLiteral("Engine returned malformed bestmove: %1")
                                   .arg(bounded_detail(normalized)));
            emit engineEvent(QStringLiteral("protocol_error"), bounded_detail(normalized));
            set_state(State::Ready);
            return;
        }
        emit bestMoveReceived(bestmove);
        if (state_ == State::Searching || state_ == State::Stopping) {
            set_state(State::Ready);
        }
    }
}

void EngineController::handle_process_started() {
    emit engineEvent(QStringLiteral("started"), QString::number(process_.processId()));
    send_uci_handshake();
}

void EngineController::handle_process_error(QProcess::ProcessError error) {
    if (intentional_shutdown_ && error == QProcess::UnknownError) {
        return;
    }
    if (error == QProcess::FailedToStart || error == QProcess::Crashed ||
        error == QProcess::WriteError || error == QProcess::ReadError) {
        fail_controller(QStringLiteral("Engine process error: %1").arg(process_error_name(error)),
                        false);
    }
}

void EngineController::handle_process_finished(int exit_code, QProcess::ExitStatus exit_status) {
    handshake_timer_.stop();
    shutdown_timer_.stop();
    clear_protocol_state();
    emit engineEvent(QStringLiteral("finished"), QStringLiteral("exit_code=%1").arg(exit_code));

    if (state_ == State::Failed) {
        intentional_shutdown_ = false;
        return;
    }
    if (intentional_shutdown_) {
        intentional_shutdown_ = false;
        set_state(State::Stopped);
        return;
    }
    if (exit_status == QProcess::CrashExit) {
        fail_controller(QStringLiteral("Engine process crashed."), false);
        return;
    }
    fail_controller(
        QStringLiteral("Engine process exited unexpectedly with code %1.").arg(exit_code), false);
}

void EngineController::handle_handshake_timeout() {
    if (state_ != State::Starting) {
        return;
    }
    const QString phase = saw_uciok_ ? QStringLiteral("readyok") : QStringLiteral("uciok");
    fail_controller(QStringLiteral("Engine UCI handshake timed out waiting for %1.").arg(phase),
                    true);
}

void EngineController::handle_shutdown_timeout() {
    if (process_.state() != QProcess::NotRunning) {
        emit engineEvent(QStringLiteral("kill"), QStringLiteral("shutdown_timeout"));
        process_.kill();
    }
}

void EngineController::fail_controller(const QString& message, bool kill_process) {
    handshake_timer_.stop();
    shutdown_timer_.stop();
    emit errorOccurred(message);
    emit engineEvent(QStringLiteral("error"), bounded_detail(message));
    if (kill_process && process_.state() != QProcess::NotRunning) {
        intentional_shutdown_ = false;
        process_.kill();
    }
    set_state(State::Failed);
}

void EngineController::clear_protocol_state() {
    stdout_buffer_.clear();
    stderr_buffer_.clear();
    saw_uciok_ = false;
    saw_readyok_ = false;
}
