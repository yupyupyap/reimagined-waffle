#include "app_settings.hpp"
#include "engine_controller.hpp"
#include "evaluation_display.hpp"

#include <eng/position.hpp>

#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QSettings>
#include <QString>
#include <QTemporaryDir>
#include <QThread>

#include <cmath>
#include <string_view>

namespace {

#ifdef __has_feature
#if __has_feature(thread_sanitizer)
#define LCE_ENGINE_CONTROLLER_TEST_THREAD_SANITIZER 1
#endif
#endif
#ifdef __SANITIZE_THREAD__
#define LCE_ENGINE_CONTROLLER_TEST_THREAD_SANITIZER 1
#endif

constexpr int kShortTimeoutMs = 3000;
#ifdef LCE_ENGINE_CONTROLLER_TEST_THREAD_SANITIZER
constexpr int kLongTimeoutMs = 120000;
#else
constexpr int kLongTimeoutMs = 20000;
#endif
constexpr int kEventSliceMs = 10;
constexpr int kSleepMs = 2;
constexpr int kExpectedDepth = 3;
constexpr int kExpectedSeldepth = 5;
constexpr qint64 kExpectedTimeMs = 42;
constexpr quint64 kExpectedNodes = 1234;
constexpr qint64 kExpectedNps = 5678;
constexpr int kExpectedScoreCp = -18;
constexpr int kExpectedHashfull = 7;
constexpr int kExpectedMultipv = 2;
constexpr int kSettingsHashMb = 64;
constexpr int kSettingsThreads = 3;
constexpr int kSettingsMultiPv = 2;
constexpr int kSettingsMoveOverheadMs = 25;
constexpr int kSettingsMoveTimeMs = 250;
constexpr int kControllerHashMb = 2;
constexpr int kControllerThreads = 1;
constexpr int kControllerMultiPv = 2;
constexpr int kControllerMoveOverheadMs = 30;
constexpr int kControllerMoveTimeMs = 50;

[[nodiscard]] bool require(bool condition, std::string_view message) {
    if (!condition) {
        qCritical() << "FAIL:"
                    << QString::fromUtf8(message.data(), static_cast<qsizetype>(message.size()));
    }
    return condition;
}

template <typename Predicate> [[nodiscard]] bool wait_until(Predicate predicate, int timeout_ms) {
    QElapsedTimer timer;
    timer.start();
    while (timer.elapsed() < timeout_ms) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, kEventSliceMs);
        if (predicate()) {
            return true;
        }
        QThread::msleep(kSleepMs);
    }
    QCoreApplication::processEvents(QEventLoop::AllEvents, kEventSliceMs);
    return predicate();
}

[[nodiscard]] QString startpos_fen() {
    return QString::fromLatin1(eng::Position::startpos_fen.data(),
                               static_cast<qsizetype>(eng::Position::startpos_fen.size()));
}

[[nodiscard]] bool test_parser() {
    EngineController::SearchInfo info;
    bool passed = EngineController::parse_search_info_line(
        QStringLiteral("info depth 3 seldepth 5 time 42 nodes 1234 nps 5678 score cp -18 "
                       "lowerbound hashfull 7 pv g1f3 b8c6"),
        info);
    passed = passed && info.depth == kExpectedDepth && info.seldepth == kExpectedSeldepth &&
             info.time_ms == kExpectedTimeMs && info.nodes == kExpectedNodes &&
             info.nps == kExpectedNps && info.has_score_cp && info.score_cp == kExpectedScoreCp &&
             info.bound == EngineController::ScoreBound::Lower && info.multipv == 1 &&
             info.hashfull == kExpectedHashfull &&
             info.pv == QStringList{QStringLiteral("g1f3"), QStringLiteral("b8c6")};

    EngineController::SearchInfo mate_info;
    passed = passed &&
             EngineController::parse_search_info_line(
                 QStringLiteral("info depth 4 multipv 2 score mate -2 upperbound pv a7a8q"),
                 mate_info) &&
             mate_info.has_mate && mate_info.mate == -2 &&
             mate_info.bound == EngineController::ScoreBound::Upper &&
             mate_info.multipv == kExpectedMultipv &&
             mate_info.pv == QStringList{QStringLiteral("a7a8q")};

    EngineController::BestMove bestmove;
    passed = passed &&
             EngineController::parse_bestmove_line(QStringLiteral("bestmove e2e4 ponder e7e5"),
                                                   bestmove) &&
             bestmove.move == QStringLiteral("e2e4") && bestmove.ponder == QStringLiteral("e7e5");
    passed = passed &&
             EngineController::parse_bestmove_line(QStringLiteral("bestmove 0000"), bestmove) &&
             bestmove.is_null();
    passed =
        passed && !EngineController::parse_bestmove_line(QStringLiteral("bestmove bad"), bestmove);

    return require(passed, "parser handles info, bestmove and malformed bestmove");
}

[[nodiscard]] bool test_incremental_lines() {
    auto extracted = EngineController::extract_complete_lines({}, QStringLiteral("info depth 1"));
    bool passed = !extracted.overflow && extracted.lines.isEmpty() &&
                  extracted.remaining == QStringLiteral("info depth 1");

    extracted = EngineController::extract_complete_lines(
        extracted.remaining, QStringLiteral(" nodes 1\r\nbestmove e2e4\npartial"));
    passed = passed && !extracted.overflow &&
             extracted.lines == QStringList{QStringLiteral("info depth 1 nodes 1"),
                                            QStringLiteral("bestmove e2e4")} &&
             extracted.remaining == QStringLiteral("partial");

    return require(passed, "line extraction preserves partial stdout and CRLF");
}

[[nodiscard]] bool test_evaluation_display_model() {
    EngineController::SearchInfo cp_info;
    cp_info.has_score_cp = true;
    cp_info.score_cp = 125;
    cp_info.bound = EngineController::ScoreBound::Lower;

    const auto neutral = gui::neutral_evaluation_display();
    const auto white_display = gui::evaluation_display_from_search_info(cp_info, eng::Color::White);
    const auto black_display = gui::evaluation_display_from_search_info(cp_info, eng::Color::Black);

    bool passed = neutral.has_value && neutral.text == QStringLiteral("+0.00") &&
                  std::abs(neutral.white_fraction - 0.5) < 0.000001;
    passed = passed && white_display.has_value && white_display.white_score_cp == 125 &&
             white_display.bound == EngineController::ScoreBound::Lower &&
             white_display.text == QStringLiteral(">= +1.25") && white_display.white_fraction > 0.5;
    passed = passed && black_display.has_value && black_display.white_score_cp == -125 &&
             black_display.bound == EngineController::ScoreBound::Upper &&
             black_display.text == QStringLiteral("<= -1.25") && black_display.white_fraction < 0.5;

    EngineController::SearchInfo mate_info;
    mate_info.has_mate = true;
    mate_info.mate = 3;
    const auto white_mate = gui::evaluation_display_from_search_info(mate_info, eng::Color::White);
    mate_info.mate = -2;
    const auto black_mate = gui::evaluation_display_from_search_info(mate_info, eng::Color::White);

    passed = passed && white_mate.has_value && white_mate.is_mate &&
             white_mate.text == QStringLiteral("+M3") && white_mate.white_fraction > 0.9 &&
             white_mate.graph_score_cp == gui::evaluation_graph_score_limit_cp;
    passed = passed && black_mate.has_value && black_mate.is_mate &&
             black_mate.text == QStringLiteral("-M2") && black_mate.white_fraction < 0.1 &&
             black_mate.graph_score_cp == -gui::evaluation_graph_score_limit_cp;

    return require(passed, "evaluation display model formats score, mate and bar fraction");
}

[[nodiscard]] bool test_invalid_engine_path() {
    EngineController controller;
    QString error;
    QObject::connect(&controller, &EngineController::errorOccurred, &controller,
                     [&error](const QString& message) { error = message; });

    const QString missing_path =
        QDir::temp().absoluteFilePath(QStringLiteral("localchessengine-missing-for-test"));
    const bool started = controller.start_engine(missing_path);
    const bool passed = !started && controller.state() == EngineController::State::Failed &&
                        error.contains(QStringLiteral("not available"));
    return require(passed, "invalid engine path fails without launching a process");
}

[[nodiscard]] bool test_gui_settings_persistence(const QString& engine_path) {
    QTemporaryDir temp_dir;
    if (!require(temp_dir.isValid(), "temporary settings directory is available")) {
        return false;
    }

    const QString eval_file_path = temp_dir.filePath(QStringLiteral("network.nnue"));
    QFile eval_file{eval_file_path};
    bool passed = eval_file.open(QIODevice::WriteOnly);
    if (passed) {
        passed = eval_file.write("nnue") == 4;
        eval_file.close();
    }

    GuiSettings settings = GuiSettings::defaults(engine_path);
    settings.hash_mb = kSettingsHashMb;
    settings.threads = kSettingsThreads;
    settings.multi_pv = kSettingsMultiPv;
    settings.move_overhead_ms = kSettingsMoveOverheadMs;
    settings.use_nnue = true;
    settings.eval_file = eval_file_path;
    settings.nnue_backend = GuiSettings::NnueBackend::Auto;
    settings.engine_move_time_ms = kSettingsMoveTimeMs;
    passed = passed && settings.is_valid();

    const QString settings_path = temp_dir.filePath(QStringLiteral("gui-settings.ini"));
    {
        QSettings store{settings_path, QSettings::IniFormat};
        settings.save_to(store);
    }

    QSettings reload{settings_path, QSettings::IniFormat};
    const GuiSettings loaded =
        GuiSettings::load_from(reload, QStringLiteral("/missing/fallback-engine"));
    passed = passed && loaded == settings;

    GuiSettings invalid = settings;
    invalid.use_nnue = true;
    invalid.eval_file = temp_dir.filePath(QStringLiteral("missing.nnue"));
    passed = passed && !invalid.is_valid();

    return require(passed, "GUI settings persist, reload and validate option ranges");
}

[[nodiscard]] bool wait_for_ready(EngineController& controller) {
    return wait_until(
        [&controller]() { return controller.state() == EngineController::State::Ready; },
        kLongTimeoutMs);
}

[[nodiscard]] bool wait_for_stopped(EngineController& controller) {
    return wait_until(
        [&controller]() { return controller.state() == EngineController::State::Stopped; },
        kShortTimeoutMs);
}

[[nodiscard]] bool test_handshake(const QString& engine_path) {
    EngineController controller;
    QString error;
    QObject::connect(&controller, &EngineController::errorOccurred, &controller,
                     [&error](const QString& message) { error = message; });

    bool passed = controller.start_engine(engine_path);
    passed = passed && wait_for_ready(controller);
    controller.quit_engine();
    passed = passed && wait_for_stopped(controller) && error.isEmpty();
    return require(passed, "real engine reaches UCI ready state and quits cleanly");
}

[[nodiscard]] bool test_go_nodes(const QString& engine_path) {
    EngineController controller;
    QString bestmove;
    int info_count = 0;
    QObject::connect(&controller, &EngineController::bestMoveReceived, &controller,
                     [&bestmove](const EngineController::BestMove& move) { bestmove = move.move; });
    QObject::connect(&controller, &EngineController::searchInfoReceived, &controller,
                     [&info_count](const EngineController::SearchInfo&) { ++info_count; });

    bool passed = controller.start_engine(engine_path);
    passed = passed && wait_for_ready(controller);
    passed = passed && controller.set_position_fen(startpos_fen());
    passed = passed && controller.go_nodes(1);
    passed = passed && wait_until(
                           [&controller, &bestmove]() {
                               return !bestmove.isEmpty() &&
                                      controller.state() == EngineController::State::Ready;
                           },
                           kLongTimeoutMs);
    controller.quit_engine();
    passed = passed && wait_for_stopped(controller) && !bestmove.isEmpty() && info_count > 0;
    return require(passed, "go nodes 1 emits info and bestmove without blocking the event loop");
}

[[nodiscard]] bool test_options_and_movetime(const QString& engine_path) {
    EngineController controller;
    QString bestmove;
    QStringList commands;
    QObject::connect(&controller, &EngineController::bestMoveReceived, &controller,
                     [&bestmove](const EngineController::BestMove& move) { bestmove = move.move; });
    QObject::connect(&controller, &EngineController::engineEvent, &controller,
                     [&commands](const QString& event_type, const QString& detail) {
                         if (event_type == QStringLiteral("command")) {
                             commands.push_back(detail);
                         }
                     });

    EngineController::EngineOptions options;
    options.hash_mb = kControllerHashMb;
    options.threads = kControllerThreads;
    options.multi_pv = kControllerMultiPv;
    options.move_overhead_ms = kControllerMoveOverheadMs;
    options.use_nnue = false;
    options.nnue_backend = QStringLiteral("Scalar");
    controller.set_engine_options(options);

    bool passed = controller.start_engine(engine_path);
    passed = passed && wait_for_ready(controller);
    passed = passed && commands.contains(
                           QStringLiteral("setoption name Hash value %1").arg(kControllerHashMb));
    passed =
        passed && commands.contains(
                      QStringLiteral("setoption name Threads value %1").arg(kControllerThreads));
    passed =
        passed && commands.contains(
                      QStringLiteral("setoption name MultiPV value %1").arg(kControllerMultiPv));
    passed =
        passed &&
        commands.contains(
            QStringLiteral("setoption name Move Overhead value %1").arg(kControllerMoveOverheadMs));
    passed = passed && commands.contains(QStringLiteral("setoption name Use NNUE value false"));
    passed = passed && controller.set_position_fen(startpos_fen());
    passed = passed && controller.go_movetime(kControllerMoveTimeMs);
    passed = passed && wait_until(
                           [&controller, &bestmove]() {
                               return !bestmove.isEmpty() &&
                                      controller.state() == EngineController::State::Ready;
                           },
                           kLongTimeoutMs);
    controller.quit_engine();
    passed = passed && wait_for_stopped(controller) &&
             commands.contains(QStringLiteral("go movetime %1").arg(kControllerMoveTimeMs)) &&
             !bestmove.isEmpty();
    return require(passed, "UCI options are applied and go movetime returns a bestmove");
}

[[nodiscard]] bool test_go_infinite_stop(const QString& engine_path) {
    EngineController controller;
    QString bestmove;
    QStringList commands;
    int info_count = 0;
    QObject::connect(&controller, &EngineController::bestMoveReceived, &controller,
                     [&bestmove](const EngineController::BestMove& move) { bestmove = move.move; });
    QObject::connect(&controller, &EngineController::searchInfoReceived, &controller,
                     [&info_count](const EngineController::SearchInfo&) { ++info_count; });
    QObject::connect(&controller, &EngineController::engineEvent, &controller,
                     [&commands](const QString& event_type, const QString& detail) {
                         if (event_type == QStringLiteral("command")) {
                             commands.push_back(detail);
                         }
                     });

    bool passed = controller.start_engine(engine_path);
    passed = passed && wait_for_ready(controller);
    passed = passed && controller.set_position_fen(startpos_fen());
    passed = passed && controller.go_infinite();
    passed = passed && wait_until(
                           [&controller, &info_count]() {
                               return info_count > 0 &&
                                      controller.state() == EngineController::State::Searching;
                           },
                           kLongTimeoutMs);
    controller.stop_search();
    passed = passed && wait_until(
                           [&controller, &bestmove]() {
                               return !bestmove.isEmpty() &&
                                      controller.state() == EngineController::State::Ready;
                           },
                           kLongTimeoutMs);
    controller.quit_engine();
    passed = passed && wait_for_stopped(controller) &&
             commands.contains(QStringLiteral("go infinite")) && !bestmove.isEmpty() &&
             info_count > 0;
    return require(passed, "go infinite emits info and returns bestmove after stop");
}

} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
int main(int argc, char** argv) {
    QCoreApplication app{argc, argv};
    if (QCoreApplication::arguments().size() < 2) {
        qCritical() << "usage: localchessengine_gui_tests /path/to/localchessengine";
        return 2;
    }

    const QString engine_path = QCoreApplication::arguments().at(1);
    int failures = 0;
    failures += test_parser() ? 0 : 1;
    failures += test_incremental_lines() ? 0 : 1;
    failures += test_evaluation_display_model() ? 0 : 1;
    failures += test_invalid_engine_path() ? 0 : 1;
    failures += test_gui_settings_persistence(engine_path) ? 0 : 1;
    failures += test_handshake(engine_path) ? 0 : 1;
    failures += test_go_nodes(engine_path) ? 0 : 1;
    failures += test_options_and_movetime(engine_path) ? 0 : 1;
    failures += test_go_infinite_stop(engine_path) ? 0 : 1;
    return failures == 0 ? 0 : 1;
}
