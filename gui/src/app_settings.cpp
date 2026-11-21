#include "app_settings.hpp"

#include <QFileInfo>
#include <QSettings>
#include <QStringList>

namespace {

constexpr auto* kEnginePathKey = "engine/path";
constexpr auto* kHashMbKey = "engine/hash_mb";
constexpr auto* kThreadsKey = "engine/threads";
constexpr auto* kMultiPvKey = "engine/multi_pv";
constexpr auto* kMoveOverheadMsKey = "engine/move_overhead_ms";
constexpr auto* kUseNnueKey = "engine/use_nnue";
constexpr auto* kEvalFileKey = "engine/eval_file";
constexpr auto* kNnueBackendKey = "engine/nnue_backend";
constexpr auto* kEngineMoveTimeMsKey = "game/engine_move_time_ms";

[[nodiscard]] bool contains_line_break(const QString& text) {
    return text.contains(QLatin1Char('\n')) || text.contains(QLatin1Char('\r'));
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] int setting_int(QSettings& settings, const char* key, int fallback, int minimum,
                              int maximum) {
    bool parsed = false;
    const int value = settings.value(QString::fromLatin1(key), fallback).toInt(&parsed);
    if (!parsed || value < minimum || value > maximum) {
        return fallback;
    }
    return value;
}

[[nodiscard]] QString single_line_setting(QSettings& settings, const char* key,
                                          const QString& fallback) {
    QString value = settings.value(QString::fromLatin1(key), fallback).toString().trimmed();
    if (value.isEmpty() || contains_line_break(value)) {
        return fallback;
    }
    return value;
}

[[nodiscard]] QString validate_range(int value, int minimum, int maximum, const QString& label) {
    if (value < minimum || value > maximum) {
        return QStringLiteral("%1 must be between %2 and %3.").arg(label).arg(minimum).arg(maximum);
    }
    return {};
}

} // namespace

GuiSettings GuiSettings::defaults(const QString& fallback_engine_path) {
    GuiSettings settings;
    settings.engine_path = fallback_engine_path.trimmed();
    return settings;
}

GuiSettings GuiSettings::load(const QString& fallback_engine_path) {
    QSettings settings;
    return load_from(settings, fallback_engine_path);
}

GuiSettings GuiSettings::load_from(QSettings& settings, const QString& fallback_engine_path) {
    GuiSettings loaded = defaults(fallback_engine_path);
    loaded.engine_path = single_line_setting(settings, kEnginePathKey, loaded.engine_path);
    loaded.hash_mb = setting_int(settings, kHashMbKey, default_hash_mb, min_hash_mb, max_hash_mb);
    loaded.threads = setting_int(settings, kThreadsKey, default_threads, min_threads, max_threads);
    loaded.multi_pv =
        setting_int(settings, kMultiPvKey, default_multi_pv, min_multi_pv, max_multi_pv);
    loaded.move_overhead_ms = setting_int(settings, kMoveOverheadMsKey, default_move_overhead_ms,
                                          min_move_overhead_ms, max_move_overhead_ms);
    loaded.use_nnue = settings.value(QString::fromLatin1(kUseNnueKey), false).toBool();
    loaded.eval_file = single_line_setting(settings, kEvalFileKey, {});
    loaded.nnue_backend =
        nnue_backend_from_string(settings
                                     .value(QString::fromLatin1(kNnueBackendKey),
                                            nnue_backend_to_string(NnueBackend::Scalar))
                                     .toString(),
                                 NnueBackend::Scalar);
    loaded.engine_move_time_ms =
        setting_int(settings, kEngineMoveTimeMsKey, default_engine_move_time_ms,
                    min_engine_move_time_ms, max_engine_move_time_ms);
    return loaded;
}

void GuiSettings::save() const {
    QSettings settings;
    save_to(settings);
}

void GuiSettings::save_to(QSettings& settings) const {
    settings.setValue(QString::fromLatin1(kEnginePathKey), engine_path.trimmed());
    settings.setValue(QString::fromLatin1(kHashMbKey), hash_mb);
    settings.setValue(QString::fromLatin1(kThreadsKey), threads);
    settings.setValue(QString::fromLatin1(kMultiPvKey), multi_pv);
    settings.setValue(QString::fromLatin1(kMoveOverheadMsKey), move_overhead_ms);
    settings.setValue(QString::fromLatin1(kUseNnueKey), use_nnue);
    settings.setValue(QString::fromLatin1(kEvalFileKey), eval_file.trimmed());
    settings.setValue(QString::fromLatin1(kNnueBackendKey), nnue_backend_to_string(nnue_backend));
    settings.setValue(QString::fromLatin1(kEngineMoveTimeMsKey), engine_move_time_ms);
    settings.sync();
}

QString GuiSettings::validation_error() const {
    const QString trimmed_engine_path = engine_path.trimmed();
    if (trimmed_engine_path.isEmpty()) {
        return QStringLiteral("Engine path is required.");
    }
    if (contains_line_break(trimmed_engine_path)) {
        return QStringLiteral("Engine path must be a single line.");
    }
    const QFileInfo engine_info{trimmed_engine_path};
    if (!engine_info.exists() || !engine_info.isFile() || !engine_info.isExecutable()) {
        return QStringLiteral("Engine path must point to an executable file.");
    }

    for (const auto& error : QStringList{
             validate_range(hash_mb, min_hash_mb, max_hash_mb, QStringLiteral("Hash")),
             validate_range(threads, min_threads, max_threads, QStringLiteral("Threads")),
             validate_range(multi_pv, min_multi_pv, max_multi_pv, QStringLiteral("MultiPV")),
             validate_range(move_overhead_ms, min_move_overhead_ms, max_move_overhead_ms,
                            QStringLiteral("Move Overhead")),
             validate_range(engine_move_time_ms, min_engine_move_time_ms, max_engine_move_time_ms,
                            QStringLiteral("Engine move time"))}) {
        if (!error.isEmpty()) {
            return error;
        }
    }

    const QString trimmed_eval_file = eval_file.trimmed();
    if (contains_line_break(trimmed_eval_file)) {
        return QStringLiteral("NNUE file path must be a single line.");
    }
    if (use_nnue) {
        if (trimmed_eval_file.isEmpty()) {
            return QStringLiteral("An NNUE file is required when Use NNUE is enabled.");
        }
        const QFileInfo eval_info{trimmed_eval_file};
        if (!eval_info.exists() || !eval_info.isFile()) {
            return QStringLiteral("NNUE file must point to a readable file.");
        }
    }

    return {};
}

bool GuiSettings::is_valid() const { return validation_error().isEmpty(); }

QString GuiSettings::nnue_backend_uci_name() const { return nnue_backend_to_string(nnue_backend); }

QString GuiSettings::nnue_backend_to_string(NnueBackend backend) {
    switch (backend) {
    case NnueBackend::Scalar:
        return QStringLiteral("Scalar");
    case NnueBackend::Neon:
        return QStringLiteral("Neon");
    case NnueBackend::Auto:
        return QStringLiteral("Auto");
    }
    return QStringLiteral("Scalar");
}

GuiSettings::NnueBackend GuiSettings::nnue_backend_from_string(const QString& text,
                                                               NnueBackend fallback) {
    const QString normalized = text.trimmed().toLower();
    if (normalized == QStringLiteral("scalar")) {
        return NnueBackend::Scalar;
    }
    if (normalized == QStringLiteral("neon")) {
        return NnueBackend::Neon;
    }
    if (normalized == QStringLiteral("auto")) {
        return NnueBackend::Auto;
    }
    return fallback;
}
