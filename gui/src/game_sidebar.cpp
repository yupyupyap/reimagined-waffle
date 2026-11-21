#include "game_sidebar.hpp"

#include "engine_info_panel.hpp"

#include <QAbstractButton>
#include <QAction>
#include <QButtonGroup>
#include <QFont>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSize>
#include <QSizePolicy>
#include <QStackedWidget>
#include <QStringList>
#include <QStyle>
#include <QTabBar>
#include <QToolButton>
#include <QVBoxLayout>

namespace {

constexpr int kSidebarMinimumWidth = 320;
constexpr int kTopMargin = 0;
constexpr int kSideMargin = 12;
constexpr int kSimplePageTopMargin = 12;
constexpr int kSectionSpacing = 14;
constexpr int kTightSpacing = 6;
constexpr int kButtonSpacing = 8;
constexpr int kNavigationButtonSide = 36;
constexpr qsizetype kMaxPvCharacters = 140;

[[nodiscard]] QString sidebar_button_style() {
    return QStringLiteral(
        "QPushButton, QToolButton { background: palette(base); border: 1px solid palette(mid); "
        "color: palette(text); min-height: 28px; padding: 5px 10px; }"
        "QPushButton:hover, QToolButton:hover { background: palette(alternate-base); }"
        "QPushButton:checked, QToolButton:checked { background: palette(highlight); "
        "color: palette(highlighted-text); }"
        "QPushButton:disabled, QToolButton:disabled { color: palette(mid); }");
}

void apply_sidebar_button_style(QAbstractButton* button) {
    if (button != nullptr) {
        button->setStyleSheet(sidebar_button_style());
    }
}

[[nodiscard]] QLabel* section_label(const QString& text, QWidget* parent) {
    auto* label = new QLabel(text, parent); // NOLINT(cppcoreguidelines-owning-memory)
    QFont font = label->font();
    font.setBold(true);
    label->setFont(font);
    return label;
}

[[nodiscard]] QLabel* value_label(const QString& text, QWidget* parent) {
    auto* label = new QLabel(text, parent); // NOLINT(cppcoreguidelines-owning-memory)
    label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    label->setWordWrap(true);
    return label;
}

[[nodiscard]] QPushButton* mode_button(const QString& text, QWidget* parent) {
    auto* button = new QPushButton(text, parent); // NOLINT(cppcoreguidelines-owning-memory)
    button->setCheckable(true);
    button->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    apply_sidebar_button_style(button);
    return button;
}

[[nodiscard]] QString clean_action_text(QString text) {
    text.remove(QLatin1Char('&'));
    return text.trimmed();
}

} // namespace

// NOLINTBEGIN(cppcoreguidelines-owning-memory, cppcoreguidelines-prefer-member-initializer)
GameSidebar::GameSidebar(QWidget* parent) : QWidget(parent) {
    setObjectName(QStringLiteral("gameSidebar"));
    setAccessibleName(QStringLiteral("Game sidebar"));
    setMinimumWidth(kSidebarMinimumWidth);
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(kSectionSpacing);

    root->addWidget(create_top_area());

    view_tabs_ = new QTabBar(this);
    view_tabs_->setExpanding(true);
    view_tabs_->setDocumentMode(true);
    view_tabs_->setDrawBase(false);
    view_tabs_->setStyleSheet(QStringLiteral(
        "QTabBar::tab { background: rgb(26, 26, 26); border: 1px solid rgb(58, 58, 58); "
        "color: rgb(242, 242, 242); min-height: 24px; padding: 6px 14px; }"
        "QTabBar::tab:selected { background: rgb(245, 245, 245); color: rgb(24, 24, 24); }"
        "QTabBar::tab:disabled { color: rgb(112, 112, 112); }"));
    view_tabs_->addTab(QStringLiteral("Play"));
    view_tabs_->addTab(QStringLiteral("Advanced"));
    root->addWidget(view_tabs_);

    view_stack_ = new QStackedWidget(this);
    view_stack_->addWidget(create_simple_page());
    advanced_panel_ = new EngineInfoPanel(view_stack_);
    view_stack_->addWidget(advanced_panel_);
    root->addWidget(view_stack_, 1);

    connect(view_tabs_, &QTabBar::currentChanged, view_stack_, &QStackedWidget::setCurrentIndex);
    connect(advanced_panel_, &EngineInfoPanel::analysisToggleRequested, this,
            &GameSidebar::analysisToggleRequested);

    clear_for_new_game();
    set_play_mode(PlayMode::HumanVsHuman);
}
// NOLINTEND(cppcoreguidelines-owning-memory, cppcoreguidelines-prefer-member-initializer)

void GameSidebar::set_navigation_actions(QAction* first, QAction* previous, QAction* next,
                                         QAction* last) {
    if (navigation_layout_ == nullptr) {
        return;
    }

    while (QLayoutItem* item = navigation_layout_->takeAt(0)) {
        if (QWidget* widget = item->widget(); widget != nullptr) {
            widget->deleteLater();
        }
        delete item;
    }

    navigation_layout_->addStretch(1);
    add_navigation_button(first, QStringLiteral("|<"));
    add_navigation_button(previous, QStringLiteral("<"));
    add_navigation_button(next, QStringLiteral(">"));
    add_navigation_button(last, QStringLiteral(">|"));
    navigation_layout_->addStretch(1);
}

void GameSidebar::set_play_mode(PlayMode mode) {
    play_mode_ = mode;
    if (human_mode_button_ == nullptr || engine_mode_button_ == nullptr) {
        return;
    }

    const QSignalBlocker human_blocker{human_mode_button_};
    const QSignalBlocker engine_blocker{engine_mode_button_};
    human_mode_button_->setChecked(mode == PlayMode::HumanVsHuman);
    engine_mode_button_->setChecked(mode == PlayMode::HumanVsEngine);
}

void GameSidebar::set_game_snapshot(const GameSnapshot& snapshot) {
    set_play_mode(snapshot.play_mode);
    if (game_status_value_ != nullptr) {
        game_status_value_->setText(snapshot.game_status.isEmpty() ? QStringLiteral("Ready")
                                                                   : snapshot.game_status);
    }
    if (game_detail_value_ != nullptr) {
        game_detail_value_->setText(snapshot_detail(snapshot));
    }
    if (engine_state_value_ != nullptr) {
        engine_state_value_->setText(snapshot.engine_status);
    }
}

void GameSidebar::clear_for_new_game() {
    reset_simple_analysis();
    if (advanced_panel_ != nullptr) {
        advanced_panel_->clear_for_new_game();
    }
}

void GameSidebar::begin_search(const QString& label) {
    if (depth_value_ != nullptr) {
        depth_value_->setText(label);
    }
    if (line_value_ != nullptr) {
        line_value_->setText(QStringLiteral("-"));
    }
    if (advanced_panel_ != nullptr) {
        advanced_panel_->begin_search(label);
    }
}

void GameSidebar::set_engine_state(EngineController::State state) {
    if (engine_state_value_ != nullptr) {
        engine_state_value_->setText(
            QStringLiteral("Engine: %1").arg(EngineController::state_name(state)));
    }
    if (advanced_panel_ != nullptr) {
        advanced_panel_->set_engine_state(state);
    }
}

void GameSidebar::set_controls(Controls controls) {
    if (analysis_toggle_button_ != nullptr) {
        analysis_toggle_button_->setEnabled(controls.can_toggle_analysis);
        if (controls.analysis_paused) {
            analysis_toggle_button_->setText(QStringLiteral("Resume Analysis"));
            analysis_toggle_button_->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
        } else if (controls.analysis_active) {
            analysis_toggle_button_->setText(QStringLiteral("Pause Analysis"));
            analysis_toggle_button_->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
        } else {
            analysis_toggle_button_->setText(QStringLiteral("Analyze Position"));
            analysis_toggle_button_->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
        }
    }
    if (stop_search_button_ != nullptr) {
        stop_search_button_->setEnabled(controls.can_stop_search);
    }
    if (advanced_panel_ != nullptr) {
        advanced_panel_->set_controls(EngineInfoPanel::Controls{
            .can_toggle_analysis = controls.can_toggle_analysis,
            .analysis_paused = controls.analysis_paused,
        });
    }
}

void GameSidebar::record_search_info(const EngineController::SearchInfo& info,
                                     eng::Color side_to_move) {
    update_simple_depth_and_line(info);
    if (advanced_panel_ != nullptr) {
        advanced_panel_->record_search_info(info, side_to_move);
    }
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void GameSidebar::append_engine_event(const QString& event_type, const QString& detail) {
    if (advanced_panel_ != nullptr) {
        advanced_panel_->append_engine_event(event_type, detail);
    }
}

QWidget* GameSidebar::create_top_area() {
    auto* top = new QWidget(this); // NOLINT(cppcoreguidelines-owning-memory)
    auto* layout = new QVBoxLayout(top);
    layout->setContentsMargins(kSideMargin, kTopMargin, 0, 0);
    layout->setSpacing(kTightSpacing);

    auto* title = section_label(QStringLiteral("Game"), top);
    layout->addWidget(title);

    auto* mode_row = new QHBoxLayout(); // NOLINT(cppcoreguidelines-owning-memory)
    mode_row->setSpacing(kButtonSpacing);
    human_mode_button_ = mode_button(mode_text(PlayMode::HumanVsHuman), top);
    engine_mode_button_ = mode_button(mode_text(PlayMode::HumanVsEngine), top);
    mode_group_ = new QButtonGroup(this); // NOLINT(cppcoreguidelines-owning-memory)
    mode_group_->setExclusive(true);
    mode_group_->addButton(human_mode_button_, static_cast<int>(PlayMode::HumanVsHuman));
    mode_group_->addButton(engine_mode_button_, static_cast<int>(PlayMode::HumanVsEngine));
    connect(mode_group_, &QButtonGroup::idClicked, this, [this](int id) {
        const auto requested = static_cast<PlayMode>(id);
        if (requested != play_mode_) {
            emit playModeChangeRequested(requested);
        }
    });
    mode_row->addWidget(human_mode_button_);
    mode_row->addWidget(engine_mode_button_);
    layout->addLayout(mode_row);
    return top;
}

QWidget* GameSidebar::create_simple_page() {
    auto* page = new QWidget(this); // NOLINT(cppcoreguidelines-owning-memory)
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(kSideMargin, kSimplePageTopMargin, 0, 0);
    layout->setSpacing(kSectionSpacing);

    auto* status_title = section_label(QStringLiteral("Position"), page);
    layout->addWidget(status_title);
    game_status_value_ = value_label(QStringLiteral("Ready"), page);
    game_detail_value_ = value_label(QStringLiteral("-"), page);
    layout->addWidget(game_status_value_);
    layout->addWidget(game_detail_value_);

    auto* action_title = section_label(QStringLiteral("Actions"), page);
    layout->addWidget(action_title);
    auto* action_layout = new QVBoxLayout(); // NOLINT(cppcoreguidelines-owning-memory)
    action_layout->setSpacing(kButtonSpacing);

    auto* new_game_button =
        new QPushButton(style()->standardIcon(QStyle::SP_FileIcon), QStringLiteral("New Game"),
                        page); // NOLINT(cppcoreguidelines-owning-memory)
    apply_sidebar_button_style(new_game_button);
    connect(new_game_button, &QPushButton::clicked, this, &GameSidebar::newGameRequested);
    action_layout->addWidget(new_game_button);

    analysis_toggle_button_ = new QPushButton(style()->standardIcon(QStyle::SP_MediaPause),
                                              QStringLiteral("Pause Analysis"),
                                              page); // NOLINT(cppcoreguidelines-owning-memory)
    apply_sidebar_button_style(analysis_toggle_button_);
    connect(analysis_toggle_button_, &QPushButton::clicked, this,
            &GameSidebar::analysisToggleRequested);
    action_layout->addWidget(analysis_toggle_button_);

    stop_search_button_ =
        new QPushButton(style()->standardIcon(QStyle::SP_MediaStop), QStringLiteral("Stop Search"),
                        page); // NOLINT(cppcoreguidelines-owning-memory)
    apply_sidebar_button_style(stop_search_button_);
    connect(stop_search_button_, &QPushButton::clicked, this, &GameSidebar::stopSearchRequested);
    action_layout->addWidget(stop_search_button_);
    layout->addLayout(action_layout);

    auto* analysis_title = section_label(QStringLiteral("Analysis"), page);
    layout->addWidget(analysis_title);
    depth_value_ = value_label(QStringLiteral("-"), page);
    line_value_ = value_label(QStringLiteral("-"), page);
    layout->addWidget(depth_value_);
    layout->addWidget(line_value_);

    auto* navigation_title = section_label(QStringLiteral("Moves"), page);
    layout->addWidget(navigation_title);
    navigation_layout_ = new QHBoxLayout(); // NOLINT(cppcoreguidelines-owning-memory)
    navigation_layout_->setSpacing(kButtonSpacing);
    layout->addLayout(navigation_layout_);

    engine_state_value_ = value_label(QStringLiteral("Engine: Stopped"), page);
    layout->addWidget(engine_state_value_);
    layout->addStretch(1);
    return page;
}

void GameSidebar::add_navigation_button(QAction* action, const QString& text) {
    if (navigation_layout_ == nullptr || action == nullptr) {
        return;
    }
    auto* button = new QToolButton(this); // NOLINT(cppcoreguidelines-owning-memory)
    button->setText(text);
    button->setToolTip(clean_action_text(action->text()));
    button->setAccessibleName(clean_action_text(action->text()));
    button->setAutoRaise(false);
    button->setMinimumSize(QSize{kNavigationButtonSide, kNavigationButtonSide});
    apply_sidebar_button_style(button);
    button->setEnabled(action->isEnabled());
    connect(button, &QToolButton::clicked, action, &QAction::trigger);
    connect(action, &QAction::changed, button, [button, action]() {
        button->setEnabled(action->isEnabled());
        button->setToolTip(clean_action_text(action->text()));
        button->setAccessibleName(clean_action_text(action->text()));
    });
    navigation_layout_->addWidget(button);
}

void GameSidebar::reset_simple_analysis() {
    if (depth_value_ != nullptr) {
        depth_value_->setText(QStringLiteral("-"));
    }
    if (line_value_ != nullptr) {
        line_value_->setText(QStringLiteral("-"));
    }
}

QString GameSidebar::mode_text(PlayMode mode) {
    switch (mode) {
    case PlayMode::HumanVsHuman:
        return QStringLiteral("Human vs Human");
    case PlayMode::HumanVsEngine:
        return QStringLiteral("Human vs Engine");
    }
    return QStringLiteral("Human vs Human");
}

QString GameSidebar::snapshot_detail(const GameSnapshot& snapshot) {
    if (snapshot.terminal) {
        return QStringLiteral("Game complete");
    }
    if (snapshot.awaiting_engine_move) {
        return QStringLiteral("Engine thinking");
    }
    if (snapshot.engine_to_move) {
        return QStringLiteral("Engine to move");
    }
    return mode_text(snapshot.play_mode);
}

QString GameSidebar::format_pv(const QStringList& pv) {
    QString text = pv.join(QLatin1Char(' ')).trimmed();
    if (text.size() > kMaxPvCharacters) {
        text.truncate(kMaxPvCharacters);
        text.append(QStringLiteral("..."));
    }
    return text.isEmpty() ? QStringLiteral("-") : text;
}

void GameSidebar::update_simple_depth_and_line(const EngineController::SearchInfo& info) {
    if (info.multipv != 1) {
        return;
    }
    if (depth_value_ != nullptr && info.depth >= 0) {
        depth_value_->setText(
            info.seldepth >= 0 ? QStringLiteral("Depth %1 / %2").arg(info.depth).arg(info.seldepth)
                               : QStringLiteral("Depth %1").arg(info.depth));
    }
    if (line_value_ != nullptr && !info.pv.isEmpty()) {
        line_value_->setText(format_pv(info.pv));
    }
}
