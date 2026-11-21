#include "app_icon.hpp"
#include "main_window.hpp"

#include <QApplication>
#include <QCoreApplication>
#include <QDebug>
#include <QStringList>
#include <QTimer>

namespace {

[[nodiscard]] QString option_value(const QStringList& arguments, const QString& option_name) {
    const qsizetype option_index = arguments.indexOf(option_name);
    if (option_index < 0 || option_index + 1 >= arguments.size()) {
        return {};
    }
    return arguments.at(option_index + 1);
}

} // namespace

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("LocalChessEngine"));
    QApplication::setOrganizationName(QStringLiteral("LocalChessEngine"));
    QApplication::setApplicationVersion(QStringLiteral("0.2.0"));
    QApplication::setWindowIcon(make_app_icon());

    MainWindow window;
    window.show();

    const QStringList arguments = QCoreApplication::arguments();
    const bool smoke_exit = arguments.contains(QStringLiteral("--smoke-exit"));
    const bool smoke_analyze = arguments.contains(QStringLiteral("--smoke-analyze"));
    const QString board_snapshot_path = option_value(arguments, QStringLiteral("--board-snapshot"));
    const QString window_snapshot_path =
        option_value(arguments, QStringLiteral("--window-snapshot"));
    constexpr int kSmokeAnalyzeDelayMs = 6000;

    if (smoke_analyze) {
        QTimer::singleShot(0, &window, [&window]() { window.resume_analysis(); });
    }

    if (!board_snapshot_path.isEmpty() || !window_snapshot_path.isEmpty()) {
        const int snapshot_delay = smoke_analyze ? kSmokeAnalyzeDelayMs : 0;
        QTimer::singleShot(snapshot_delay, &window,
                           [&window, board_snapshot_path, window_snapshot_path, smoke_exit]() {
                               if (!board_snapshot_path.isEmpty() &&
                                   !window.save_board_snapshot(board_snapshot_path)) {
                                   qWarning()
                                       << "Failed to save board snapshot:" << board_snapshot_path;
                                   QCoreApplication::exit(2);
                                   return;
                               }
                               if (!window_snapshot_path.isEmpty() &&
                                   !window.save_window_snapshot(window_snapshot_path)) {
                                   qWarning()
                                       << "Failed to save window snapshot:" << window_snapshot_path;
                                   QCoreApplication::exit(3);
                                   return;
                               }
                               if (smoke_exit) {
                                   QCoreApplication::quit();
                               }
                           });
    } else if (smoke_exit) {
        QTimer::singleShot(smoke_analyze ? kSmokeAnalyzeDelayMs : 0, &app, &QApplication::quit);
    }

    return QApplication::exec();
}
