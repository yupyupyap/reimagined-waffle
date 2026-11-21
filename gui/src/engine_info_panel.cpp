#include "engine_info_panel.hpp"

#include <QFont>
#include <QGridLayout>
#include <QLabel>
#include <QLocale>
#include <QPainter>
#include <QPainterPath>
#include <QPalette>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QTime>
#include <QVBoxLayout>
#include <QVector>

#include <algorithm>

namespace {

constexpr int kGraphMinHeight = 120;
constexpr int kGraphPreferredHeight = 140;
constexpr int kGraphPadding = 10;
constexpr qsizetype kGraphMaxPoints = 160;
constexpr int kGraphPointRadius = 3;
constexpr int kGraphLineWidth = 2;
constexpr int kGraphAccentRed = 38;
constexpr int kGraphAccentGreen = 114;
constexpr int kGraphAccentBlue = 182;
constexpr int kLogMaxBlocks = 220;
constexpr int kMetricValueMinWidth = 72;
constexpr int kCompactPanelWidth = 280;
constexpr qsizetype kMaxLogDetailLength = 220;
constexpr int kPanelLeftMargin = 12;
constexpr int kPanelSpacing = 10;
constexpr int kMetricHorizontalSpacing = 10;
constexpr int kMetricVerticalSpacing = 5;
constexpr int kPvMinHeight = 48;
constexpr int kLogMinHeight = 140;

[[nodiscard]] QLabel* metric_label(const QString& text, QWidget* parent) {
    auto* label = new QLabel(text, parent); // NOLINT(cppcoreguidelines-owning-memory)
    label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    return label;
}

[[nodiscard]] QLabel* metric_value(QWidget* parent) {
    auto* label =
        new QLabel(QStringLiteral("-"), parent); // NOLINT(cppcoreguidelines-owning-memory)
    label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    label->setMinimumWidth(kMetricValueMinWidth);
    return label;
}

[[nodiscard]] QString clipped_detail(QString detail) {
    detail.replace(QLatin1Char('\n'), QLatin1Char(' '));
    detail.replace(QLatin1Char('\r'), QLatin1Char(' '));
    if (detail.size() > kMaxLogDetailLength) {
        detail.truncate(kMaxLogDetailLength);
        detail.append(QStringLiteral("..."));
    }
    return detail.trimmed();
}

[[nodiscard]] QColor graph_accent_color() {
    return QColor{kGraphAccentRed, kGraphAccentGreen, kGraphAccentBlue};
}

} // namespace

class EvaluationGraphWidget final : public QWidget {
  public:
    explicit EvaluationGraphWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setMinimumHeight(kGraphMinHeight);
        setAccessibleName(QStringLiteral("Evaluation graph"));
    }
    ~EvaluationGraphWidget() override = default;
    EvaluationGraphWidget(const EvaluationGraphWidget&) = delete;
    EvaluationGraphWidget& operator=(const EvaluationGraphWidget&) = delete;
    EvaluationGraphWidget(EvaluationGraphWidget&&) = delete;
    EvaluationGraphWidget& operator=(EvaluationGraphWidget&&) = delete;

    [[nodiscard]] QSize sizeHint() const override {
        return QSize{kCompactPanelWidth, kGraphPreferredHeight};
    }

    void clear_points() {
        scores_.clear();
        update();
    }

    void append_score(int score_cp) {
        scores_.push_back(std::clamp(score_cp, -gui::evaluation_graph_score_limit_cp,
                                     gui::evaluation_graph_score_limit_cp));
        if (scores_.size() > kGraphMaxPoints) {
            scores_.remove(0, scores_.size() - kGraphMaxPoints);
        }
        update();
    }

  protected:
    void paintEvent(QPaintEvent* event) override {
        (void)event;
        QPainter painter{this};
        painter.setRenderHint(QPainter::Antialiasing, true);

        const QRect plot =
            rect().adjusted(kGraphPadding, kGraphPadding, -kGraphPadding, -kGraphPadding);
        painter.fillRect(rect(), palette().base());
        painter.setPen(QPen{palette().mid().color(), 1});
        painter.drawRect(plot.adjusted(0, 0, -1, -1));

        const int zero_y = plot.center().y();
        painter.setPen(QPen{palette().mid().color(), 1, Qt::DashLine});
        painter.drawLine(plot.left(), zero_y, plot.right(), zero_y);

        if (scores_.empty()) {
            painter.setPen(palette().placeholderText().color());
            painter.drawText(plot, Qt::AlignCenter, QStringLiteral("No eval"));
            return;
        }

        if (scores_.size() == 1) {
            painter.setPen(Qt::NoPen);
            painter.setBrush(QBrush{graph_accent_color()});
            painter.drawEllipse(point_for_index(0, plot), kGraphPointRadius, kGraphPointRadius);
            return;
        }

        QPainterPath path;
        path.moveTo(point_for_index(0, plot));
        for (qsizetype index = 1; index < scores_.size(); ++index) {
            path.lineTo(point_for_index(index, plot));
        }
        painter.setPen(QPen{graph_accent_color(), kGraphLineWidth});
        painter.drawPath(path);
    }

  private:
    [[nodiscard]] QPointF point_for_index(qsizetype index, QRect plot) const {
        const auto x_ratio = scores_.size() <= 1 ? 0.0
                                                 : static_cast<double>(index) /
                                                       (static_cast<double>(scores_.size()) - 1.0);
        const auto y_ratio =
            (static_cast<double>(scores_.at(index)) + gui::evaluation_graph_score_limit_cp) /
            (2.0 * gui::evaluation_graph_score_limit_cp);
        const double point_x = plot.left() + (x_ratio * plot.width());
        const double point_y = plot.bottom() - (y_ratio * plot.height());
        return QPointF{point_x, point_y};
    }

    QVector<int> scores_;
};

// NOLINTBEGIN(cppcoreguidelines-owning-memory, cppcoreguidelines-prefer-member-initializer)
EngineInfoPanel::EngineInfoPanel(QWidget* parent) : QWidget(parent) {
    setObjectName(QStringLiteral("engineInfoPanel"));
    setAccessibleName(QStringLiteral("Engine analysis panel"));
    setMinimumWidth(kCompactPanelWidth);

    auto* root = new QVBoxLayout(this); // NOLINT(cppcoreguidelines-owning-memory)
    root->setContentsMargins(kPanelLeftMargin, 0, 0, 0);
    root->setSpacing(kPanelSpacing);

    auto* title =
        new QLabel(QStringLiteral("Engine"), this); // NOLINT(cppcoreguidelines-owning-memory)
    QFont title_font = title->font();
    title_font.setBold(true);
    title->setFont(title_font);
    root->addWidget(title);

    auto* grid = new QGridLayout(); // NOLINT(cppcoreguidelines-owning-memory)
    grid->setColumnStretch(1, 1);
    grid->setHorizontalSpacing(kMetricHorizontalSpacing);
    grid->setVerticalSpacing(kMetricVerticalSpacing);

    state_value_ = metric_value(this);
    depth_value_ = metric_value(this);
    nodes_value_ = metric_value(this);
    nps_value_ = metric_value(this);
    hashfull_value_ = metric_value(this);

    int row = 0;
    grid->addWidget(metric_label(QStringLiteral("State"), this), row, 0);
    grid->addWidget(state_value_, row++, 1);
    grid->addWidget(metric_label(QStringLiteral("Depth"), this), row, 0);
    grid->addWidget(depth_value_, row++, 1);
    grid->addWidget(metric_label(QStringLiteral("Nodes"), this), row, 0);
    grid->addWidget(nodes_value_, row++, 1);
    grid->addWidget(metric_label(QStringLiteral("NPS"), this), row, 0);
    grid->addWidget(nps_value_, row++, 1);
    grid->addWidget(metric_label(QStringLiteral("Hash"), this), row, 0);
    grid->addWidget(hashfull_value_, row++, 1);
    root->addLayout(grid);

    graph_ = new EvaluationGraphWidget(this); // NOLINT(cppcoreguidelines-owning-memory)
    root->addWidget(graph_);

    auto* pv_title =
        new QLabel(QStringLiteral("PV"), this); // NOLINT(cppcoreguidelines-owning-memory)
    pv_title->setFont(title_font);
    root->addWidget(pv_title);
    pv_value_ = new QLabel(QStringLiteral("-"), this); // NOLINT(cppcoreguidelines-owning-memory)
    pv_value_->setWordWrap(true);
    pv_value_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    pv_value_->setMinimumHeight(kPvMinHeight);
    root->addWidget(pv_value_);

    auto* controls = new QHBoxLayout(); // NOLINT(cppcoreguidelines-owning-memory)
    analysis_toggle_button_ =
        new QPushButton(QStringLiteral("Pause"), this); // NOLINT(cppcoreguidelines-owning-memory)
    connect(analysis_toggle_button_, &QPushButton::clicked, this,
            &EngineInfoPanel::analysisToggleRequested);
    controls->addWidget(analysis_toggle_button_);
    root->addLayout(controls);

    auto* log_title =
        new QLabel(QStringLiteral("Log"), this); // NOLINT(cppcoreguidelines-owning-memory)
    log_title->setFont(title_font);
    root->addWidget(log_title);

    log_ = new QPlainTextEdit(this); // NOLINT(cppcoreguidelines-owning-memory)
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(kLogMaxBlocks);
    log_->setLineWrapMode(QPlainTextEdit::WidgetWidth);
    log_->setMinimumHeight(kLogMinHeight);
    root->addWidget(log_, 1);

    clear_for_new_game();
}
// NOLINTEND(cppcoreguidelines-owning-memory, cppcoreguidelines-prefer-member-initializer)

void EngineInfoPanel::clear_for_new_game() {
    reset_metrics();
    if (graph_ != nullptr) {
        graph_->clear_points();
    }
    if (log_ != nullptr) {
        log_->clear();
    }
}

void EngineInfoPanel::begin_search(const QString& label) {
    reset_metrics();
    if (graph_ != nullptr) {
        graph_->clear_points();
    }
    append_log_line(QStringLiteral("Search: %1").arg(label));
}

void EngineInfoPanel::set_engine_state(EngineController::State state) {
    if (state_value_ != nullptr) {
        state_value_->setText(EngineController::state_name(state));
    }
}

void EngineInfoPanel::set_controls(Controls controls) {
    if (analysis_toggle_button_ != nullptr) {
        analysis_toggle_button_->setEnabled(controls.can_toggle_analysis);
        analysis_toggle_button_->setText(controls.analysis_paused ? QStringLiteral("Resume")
                                                                  : QStringLiteral("Pause"));
    }
}

void EngineInfoPanel::record_search_info(const EngineController::SearchInfo& info,
                                         eng::Color side_to_move) {
    const auto display = gui::evaluation_display_from_search_info(info, side_to_move);
    append_score_to_graph(graph_, display, info.multipv);
    update_depth_and_counters(info);
    update_pv(info);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void EngineInfoPanel::append_engine_event(const QString& event_type, const QString& detail) {
    if (!should_log_event(event_type, detail)) {
        return;
    }
    append_log_line(format_event_line(event_type, detail));
}

void EngineInfoPanel::reset_metrics() {
    if (depth_value_ != nullptr) {
        depth_value_->setText(QStringLiteral("-"));
    }
    if (nodes_value_ != nullptr) {
        nodes_value_->setText(QStringLiteral("-"));
    }
    if (nps_value_ != nullptr) {
        nps_value_->setText(QStringLiteral("-"));
    }
    if (hashfull_value_ != nullptr) {
        hashfull_value_->setText(QStringLiteral("-"));
    }
    if (pv_value_ != nullptr) {
        pv_value_->setText(QStringLiteral("-"));
    }
}

void EngineInfoPanel::append_log_line(const QString& line) {
    if (log_ == nullptr || line.trimmed().isEmpty()) {
        return;
    }
    log_->appendPlainText(QStringLiteral("%1  %2").arg(
        QTime::currentTime().toString(QStringLiteral("HH:mm:ss")), line.trimmed()));
}

QString EngineInfoPanel::format_nodes(quint64 nodes) {
    return QLocale::system().toString(static_cast<qulonglong>(nodes));
}

QString EngineInfoPanel::format_nps(qint64 nps) {
    return QLocale::system().toString(static_cast<qlonglong>(nps));
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void EngineInfoPanel::append_score_to_graph(EvaluationGraphWidget* graph,
                                            const gui::EvaluationDisplay& display, int multipv) {
    if (graph == nullptr || multipv != 1 || !display.has_graph_score) {
        return;
    }
    graph->append_score(display.graph_score_cp);
}

void EngineInfoPanel::update_depth_and_counters(const EngineController::SearchInfo& info) {
    if (info.depth >= 0) {
        depth_value_->setText(info.seldepth >= 0
                                  ? QStringLiteral("%1 / %2").arg(info.depth).arg(info.seldepth)
                                  : QString::number(info.depth));
    }
    if (info.nodes > 0U) {
        nodes_value_->setText(format_nodes(info.nodes));
    }
    if (info.nps >= 0) {
        nps_value_->setText(format_nps(info.nps));
    }
    if (info.hashfull >= 0) {
        hashfull_value_->setText(QStringLiteral("%1/1000").arg(info.hashfull));
    }
}

void EngineInfoPanel::update_pv(const EngineController::SearchInfo& info) {
    if (!info.pv.isEmpty()) {
        const QString prefix =
            info.multipv > 1 ? QStringLiteral("PV %1: ").arg(info.multipv) : QString{};
        pv_value_->setText(prefix + info.pv.join(QLatin1Char(' ')));
    }
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
bool EngineInfoPanel::should_log_event(const QString& event_type, const QString& detail) {
    const QString trimmed = detail.trimmed();
    if (trimmed.isEmpty()) {
        return false;
    }
    if (event_type == QStringLiteral("stdout")) {
        return !trimmed.startsWith(QStringLiteral("info ")) &&
               !trimmed.startsWith(QStringLiteral("option "));
    }
    return true;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
QString EngineInfoPanel::format_event_line(const QString& event_type, const QString& detail) {
    const QString clipped = clipped_detail(detail);
    if (event_type == QStringLiteral("command")) {
        return QStringLiteral("> %1").arg(clipped);
    }
    if (event_type == QStringLiteral("stdout")) {
        return QStringLiteral("< %1").arg(clipped);
    }
    if (event_type == QStringLiteral("stderr")) {
        return QStringLiteral("stderr: %1").arg(clipped);
    }
    if (event_type == QStringLiteral("state")) {
        return QStringLiteral("State: %1").arg(clipped);
    }
    if (event_type == QStringLiteral("error")) {
        return QStringLiteral("Error: %1").arg(clipped);
    }
    return QStringLiteral("%1: %2").arg(event_type, clipped);
}
