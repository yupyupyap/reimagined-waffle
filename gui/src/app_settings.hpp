#ifndef GUI_APP_SETTINGS_HPP
#define GUI_APP_SETTINGS_HPP

#include <QString>

class QSettings;

struct GuiSettings final {
    enum class NnueBackend : unsigned char { Scalar, Neon, Auto };

    static constexpr int min_hash_mb = 1;
    static constexpr int max_hash_mb = 65536;
    static constexpr int default_hash_mb = 16;
    static constexpr int min_threads = 1;
    static constexpr int max_threads = 512;
    static constexpr int default_threads = 10;
    static constexpr int min_multi_pv = 1;
    static constexpr int max_multi_pv = 256;
    static constexpr int default_multi_pv = 1;
    static constexpr int min_move_overhead_ms = 0;
    static constexpr int max_move_overhead_ms = 5000;
    static constexpr int default_move_overhead_ms = 10;
    static constexpr int min_engine_move_time_ms = 50;
    static constexpr int max_engine_move_time_ms = 600000;
    static constexpr int default_engine_move_time_ms = 500;

    QString engine_path;
    int hash_mb{default_hash_mb};
    int threads{default_threads};
    int multi_pv{default_multi_pv};
    int move_overhead_ms{default_move_overhead_ms};
    bool use_nnue{false};
    QString eval_file;
    NnueBackend nnue_backend{NnueBackend::Scalar};
    int engine_move_time_ms{default_engine_move_time_ms};

    friend bool operator==(const GuiSettings& lhs, const GuiSettings& rhs) = default;

    [[nodiscard]] static GuiSettings defaults(const QString& fallback_engine_path);
    [[nodiscard]] static GuiSettings load(const QString& fallback_engine_path);
    [[nodiscard]] static GuiSettings load_from(QSettings& settings,
                                               const QString& fallback_engine_path);
    void save() const;
    void save_to(QSettings& settings) const;

    [[nodiscard]] QString validation_error() const;
    [[nodiscard]] bool is_valid() const;
    [[nodiscard]] QString nnue_backend_uci_name() const;

    [[nodiscard]] static QString nnue_backend_to_string(NnueBackend backend);
    [[nodiscard]] static NnueBackend nnue_backend_from_string(const QString& text,
                                                              NnueBackend fallback);
};

#endif // GUI_APP_SETTINGS_HPP
