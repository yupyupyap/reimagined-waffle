#ifndef GUI_ENGINE_CONTROLLER_HPP
#define GUI_ENGINE_CONTROLLER_HPP

#include <QObject>
#include <QProcess>
#include <QString>
#include <QStringList>
#include <QTimer>

#include <cstdint>

class EngineController final : public QObject {
    Q_OBJECT

  public:
    enum class State : std::uint8_t { Stopped, Starting, Ready, Searching, Stopping, Failed };
    Q_ENUM(State)

    enum class ScoreBound : std::uint8_t { Exact, Lower, Upper };
    Q_ENUM(ScoreBound)

    struct SearchInfo {
        int depth{-1};
        int seldepth{-1};
        qint64 time_ms{-1};
        quint64 nodes{0};
        qint64 nps{0};
        int multipv{1};
        bool has_score_cp{false};
        int score_cp{0};
        bool has_mate{false};
        int mate{0};
        ScoreBound bound{ScoreBound::Exact};
        int hashfull{-1};
        QStringList pv;
        QString raw_line;
    };

    struct BestMove {
        QString move;
        QString ponder;

        [[nodiscard]] bool is_null() const noexcept { return move == QStringLiteral("0000"); }
    };

    struct EngineOptions {
        static constexpr int default_hash_mb = 16;
        static constexpr int default_threads = 10;
        static constexpr int default_multi_pv = 1;
        static constexpr int default_move_overhead_ms = 10;

        int hash_mb{default_hash_mb};
        int threads{default_threads};
        int multi_pv{default_multi_pv};
        int move_overhead_ms{default_move_overhead_ms};
        bool use_nnue{false};
        QString eval_file;
        QString nnue_backend{QStringLiteral("Scalar")};

        friend bool operator==(const EngineOptions& lhs, const EngineOptions& rhs) = default;
    };

    struct LineExtractionResult {
        QStringList lines;
        QString remaining;
        bool overflow{false};
    };

    explicit EngineController(QObject* parent = nullptr);
    ~EngineController() override;

    EngineController(const EngineController&) = delete;
    EngineController& operator=(const EngineController&) = delete;
    EngineController(EngineController&&) = delete;
    EngineController& operator=(EngineController&&) = delete;

    [[nodiscard]] State state() const noexcept { return state_; }
    [[nodiscard]] QString engine_path() const { return engine_path_; }
    void set_engine_path(const QString& path);
    [[nodiscard]] EngineOptions engine_options() const { return engine_options_; }
    void set_engine_options(const EngineOptions& options);

    [[nodiscard]] bool start_engine();
    [[nodiscard]] bool start_engine(const QString& path);
    [[nodiscard]] bool restart_engine(const QString& path = {});
    void quit_engine();
    void stop_search();
    void new_game();
    [[nodiscard]] bool set_position_fen(const QString& fen);
    [[nodiscard]] bool go_depth(int depth);
    [[nodiscard]] bool go_infinite();
    [[nodiscard]] bool go_nodes(quint64 nodes);
    [[nodiscard]] bool go_movetime(int movetime_ms);
    [[nodiscard]] bool is_running() const noexcept;

    [[nodiscard]] static SearchInfo parse_search_info_line(const QString& line);
    [[nodiscard]] static bool parse_search_info_line(const QString& line, SearchInfo& info);
    [[nodiscard]] static bool parse_bestmove_line(const QString& line, BestMove& bestmove);
    [[nodiscard]] static LineExtractionResult extract_complete_lines(const QString& previous,
                                                                     const QString& chunk);
    [[nodiscard]] static QString state_name(State state);

  signals:
    void stateChanged(EngineController::State state);
    void searchInfoReceived(EngineController::SearchInfo info);
    void bestMoveReceived(EngineController::BestMove bestmove);
    void errorOccurred(QString message);
    void engineEvent(QString event_type, QString detail);

  private:
    void wire_process_signals();
    void set_state(State state);
    void send_uci_handshake();
    [[nodiscard]] bool send_command(const QString& command);
    [[nodiscard]] bool apply_engine_options_if_ready();
    [[nodiscard]] bool send_setoption(const QString& name, const QString& value);
    void handle_stdout_ready();
    void handle_stderr_ready();
    void handle_stdout_line(const QString& line);
    void handle_process_started();
    void handle_process_error(QProcess::ProcessError error);
    void handle_process_finished(int exit_code, QProcess::ExitStatus exit_status);
    void handle_handshake_timeout();
    void handle_shutdown_timeout();
    void fail_controller(const QString& message, bool kill_process);
    void clear_protocol_state();

    QProcess process_;
    QTimer handshake_timer_;
    QTimer shutdown_timer_;
    QString engine_path_;
    EngineOptions engine_options_;
    QString stdout_buffer_;
    QString stderr_buffer_;
    State state_{State::Stopped};
    bool saw_uciok_{false};
    bool saw_readyok_{false};
    bool intentional_shutdown_{false};
    bool engine_options_dirty_{true};
};

Q_DECLARE_METATYPE(EngineController::State)
Q_DECLARE_METATYPE(EngineController::SearchInfo)
Q_DECLARE_METATYPE(EngineController::BestMove)

#endif // GUI_ENGINE_CONTROLLER_HPP
