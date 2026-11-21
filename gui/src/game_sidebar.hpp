#ifndef GUI_GAME_SIDEBAR_HPP
#define GUI_GAME_SIDEBAR_HPP

#include "engine_controller.hpp"

#include <eng/types.hpp>

#include <QString>
#include <QWidget>

#include <cstdint>

class QAction;
class QButtonGroup;
class QHBoxLayout;
class QLabel;
class QPushButton;
class QStackedWidget;
class QTabBar;
class EngineInfoPanel;

class GameSidebar final : public QWidget {
    Q_OBJECT

  public:
    enum class PlayMode : std::uint8_t { HumanVsHuman, HumanVsEngine };

    struct Controls {
        bool can_toggle_analysis{false};
        bool analysis_paused{false};
        bool analysis_active{false};
        bool can_stop_search{false};
    };

    struct GameSnapshot {
        PlayMode play_mode{PlayMode::HumanVsHuman};
        QString game_status;
        QString engine_status;
        bool awaiting_engine_move{false};
        bool engine_to_move{false};
        bool terminal{false};
    };

    explicit GameSidebar(QWidget* parent = nullptr);
    ~GameSidebar() override = default;
    GameSidebar(const GameSidebar&) = delete;
    GameSidebar& operator=(const GameSidebar&) = delete;
    GameSidebar(GameSidebar&&) = delete;
    GameSidebar& operator=(GameSidebar&&) = delete;

    void set_navigation_actions(QAction* first, QAction* previous, QAction* next, QAction* last);
    void set_play_mode(PlayMode mode);
    void set_game_snapshot(const GameSnapshot& snapshot);
    void clear_for_new_game();
    void begin_search(const QString& label);
    void set_engine_state(EngineController::State state);
    void set_controls(Controls controls);
    void record_search_info(const EngineController::SearchInfo& info, eng::Color side_to_move);
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void append_engine_event(const QString& event_type, const QString& detail);

  signals:
    void playModeChangeRequested(GameSidebar::PlayMode mode);
    void newGameRequested();
    void analysisToggleRequested();
    void stopSearchRequested();

  private:
    QWidget* create_top_area();
    QWidget* create_simple_page();
    void add_navigation_button(QAction* action, const QString& text);
    void reset_simple_analysis();
    [[nodiscard]] static QString mode_text(PlayMode mode);
    [[nodiscard]] static QString snapshot_detail(const GameSnapshot& snapshot);
    [[nodiscard]] static QString format_pv(const QStringList& pv);
    void update_simple_depth_and_line(const EngineController::SearchInfo& info);

    QButtonGroup* mode_group_{nullptr};
    QPushButton* human_mode_button_{nullptr};
    QPushButton* engine_mode_button_{nullptr};
    QTabBar* view_tabs_{nullptr};
    QStackedWidget* view_stack_{nullptr};
    QLabel* game_status_value_{nullptr};
    QLabel* game_detail_value_{nullptr};
    QLabel* engine_state_value_{nullptr};
    QLabel* depth_value_{nullptr};
    QLabel* line_value_{nullptr};
    QPushButton* analysis_toggle_button_{nullptr};
    QPushButton* stop_search_button_{nullptr};
    QHBoxLayout* navigation_layout_{nullptr};
    EngineInfoPanel* advanced_panel_{nullptr};
    PlayMode play_mode_{PlayMode::HumanVsHuman};
};

#endif // GUI_GAME_SIDEBAR_HPP
