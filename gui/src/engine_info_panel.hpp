#ifndef GUI_ENGINE_INFO_PANEL_HPP
#define GUI_ENGINE_INFO_PANEL_HPP

#include "engine_controller.hpp"
#include "evaluation_display.hpp"

#include <eng/types.hpp>

#include <QWidget>

class EvaluationGraphWidget;
class QLabel;
class QPlainTextEdit;
class QPushButton;

class EngineInfoPanel final : public QWidget {
    Q_OBJECT

  public:
    struct Controls {
        bool can_toggle_analysis{false};
        bool analysis_paused{false};
    };

    explicit EngineInfoPanel(QWidget* parent = nullptr);
    ~EngineInfoPanel() override = default;
    EngineInfoPanel(const EngineInfoPanel&) = delete;
    EngineInfoPanel& operator=(const EngineInfoPanel&) = delete;
    EngineInfoPanel(EngineInfoPanel&&) = delete;
    EngineInfoPanel& operator=(EngineInfoPanel&&) = delete;

    void clear_for_new_game();
    void begin_search(const QString& label);
    void set_engine_state(EngineController::State state);
    void set_controls(Controls controls);
    void record_search_info(const EngineController::SearchInfo& info, eng::Color side_to_move);
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void append_engine_event(const QString& event_type, const QString& detail);

  signals:
    void analysisToggleRequested();

  private:
    void reset_metrics();
    void append_log_line(const QString& line);
    [[nodiscard]] static QString format_nodes(quint64 nodes);
    [[nodiscard]] static QString format_nps(qint64 nps);
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    static void append_score_to_graph(EvaluationGraphWidget* graph,
                                      const gui::EvaluationDisplay& display, int multipv);
    void update_depth_and_counters(const EngineController::SearchInfo& info);
    void update_pv(const EngineController::SearchInfo& info);
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    [[nodiscard]] static bool should_log_event(const QString& event_type, const QString& detail);
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    [[nodiscard]] static QString format_event_line(const QString& event_type,
                                                   const QString& detail);

    QLabel* state_value_{nullptr};
    QLabel* depth_value_{nullptr};
    QLabel* nodes_value_{nullptr};
    QLabel* nps_value_{nullptr};
    QLabel* hashfull_value_{nullptr};
    QLabel* pv_value_{nullptr};
    EvaluationGraphWidget* graph_{nullptr};
    QPlainTextEdit* log_{nullptr};
    QPushButton* analysis_toggle_button_{nullptr};
};

#endif // GUI_ENGINE_INFO_PANEL_HPP
