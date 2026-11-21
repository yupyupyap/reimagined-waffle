#ifndef GUI_MAIN_WINDOW_HPP
#define GUI_MAIN_WINDOW_HPP

#include "app_settings.hpp"
#include "engine_controller.hpp"
#include "game_sidebar.hpp"
#include "game_state.hpp"

#include <QMainWindow>
#include <QString>

#include <cstdint>
#include <optional>

class QAction;
class BoardWidget;
class EvaluationBarWidget;
class GameSidebar;

class MainWindow final : public QMainWindow {
  public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;
    MainWindow(const MainWindow&) = delete;
    MainWindow& operator=(const MainWindow&) = delete;
    MainWindow(MainWindow&&) = delete;
    MainWindow& operator=(MainWindow&&) = delete;

    [[nodiscard]] bool save_board_snapshot(const QString& path) const;
    [[nodiscard]] bool save_window_snapshot(const QString& path);
    void resume_analysis();

  private:
    enum class SearchPurpose : std::uint8_t { None, Analysis, EngineMove };

    void create_engine_controller();
    void create_menus();
    void create_central_area();
    void new_game();
    void open_pgn();
    void save_pgn();
    void save_pgn_as();
    [[nodiscard]] bool save_pgn_to_path(const QString& path);
    void navigate_first();
    void navigate_previous();
    void navigate_next();
    void navigate_last();
    void start_engine();
    void stop_engine();
    void restart_engine();
    void stop_engine_search();
    void toggle_analysis_pause();
    void pause_analysis();
    void open_preferences();
    void set_play_mode(GameSidebar::PlayMode mode, bool require_confirmation = true);
    void leave_engine_play_mode_for_pgn();
    void handle_square_clicked(int square_index);
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void handle_move_gesture(int from_square_index, int to_square_index);
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void handle_move_attempt(eng::Square from_square, eng::Square to_square);
    void handle_engine_state_changed();
    void handle_engine_search_info(const EngineController::SearchInfo& info);
    void handle_engine_event(const QString& event_type, const QString& detail);
    void handle_engine_bestmove(const QString& notation);
    void request_engine_move_if_needed();
    void start_engine_search_for_current_position();
    void start_analysis_for_current_position();
    void refresh_analysis_after_position_change();
    void request_analysis_for_current_position();
    [[nodiscard]] bool human_can_move() const;
    [[nodiscard]] bool engine_to_move() const noexcept;
    [[nodiscard]] std::optional<eng::PieceType>
    choose_promotion_piece(const GuiGameState::MoveAttempt& attempt);
    void sync_board_from_game(const QString& status_override = {});
    [[nodiscard]] QString game_status_text() const;
    [[nodiscard]] QString engine_status_text() const;
    [[nodiscard]] static QString default_engine_path();
    [[nodiscard]] static EngineController::EngineOptions
    engine_options_from_settings(const GuiSettings& settings);
    void apply_settings_to_engine();
    void update_engine_actions();
    void update_pgn_actions();
    void install_status_bar_branding();
    void show_about_dialog();
    void show_license_dialog();

    BoardWidget* board_widget_{nullptr};
    EvaluationBarWidget* evaluation_bar_{nullptr};
    GameSidebar* game_sidebar_{nullptr};
    EngineController* engine_controller_{nullptr};
    QAction* open_pgn_action_{nullptr};
    QAction* save_pgn_action_{nullptr};
    QAction* save_pgn_as_action_{nullptr};
    QAction* first_move_action_{nullptr};
    QAction* previous_move_action_{nullptr};
    QAction* next_move_action_{nullptr};
    QAction* last_move_action_{nullptr};
    QAction* start_engine_action_{nullptr};
    QAction* analysis_toggle_action_{nullptr};
    QAction* stop_search_action_{nullptr};
    QAction* stop_engine_action_{nullptr};
    QAction* restart_engine_action_{nullptr};
    QAction* preferences_action_{nullptr};
    QAction* play_engine_action_{nullptr};
    GuiSettings settings_;
    GuiGameState game_;
    QString pgn_path_;
    GameSidebar::PlayMode play_mode_{GameSidebar::PlayMode::HumanVsHuman};
    bool awaiting_engine_move_{false};
    bool pending_engine_turn_{false};
    bool pending_analysis_{false};
    bool restart_engine_when_stopped_{false};
    bool analysis_paused_{false};
    SearchPurpose active_search_purpose_{SearchPurpose::None};
    std::optional<eng::Color> active_search_side_to_move_;
};

#endif // GUI_MAIN_WINDOW_HPP
