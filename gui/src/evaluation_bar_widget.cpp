#include "evaluation_bar_widget.hpp"

#include <QFont>
#include <QFontMetrics>
#include <QPaintEvent>
#include <QPainter>
#include <QPalette>
#include <QPen>
#include <QSizePolicy>

#include <algorithm>
#include <cmath>

namespace {

constexpr int kPreferredBarWidth = 64;
constexpr int kMinimumBarWidth = 56;
constexpr int kPreferredBarHeight = 640;
constexpr int kMinimumBarHeight = 320;
constexpr int kBarPadding = 2;
constexpr int kDivisionLineWidth = 2;
constexpr int kBorderLineWidth = 1;
constexpr int kPillHorizontalPadding = 5;
constexpr int kPillVerticalPadding = 3;
constexpr int kPillMinHeight = 22;
constexpr int kPillRadius = 5;
constexpr int kLabelMinPointSize = 7;
constexpr int kDarkRed = 20;
constexpr int kDarkGreen = 20;
constexpr int kDarkBlue = 20;
constexpr int kLightRed = 246;
constexpr int kLightGreen = 243;
constexpr int kLightBlue = 235;
constexpr int kBorderAlpha = 190;
constexpr int kPillAlpha = 232;

[[nodiscard]] QColor black_bar_color() { return QColor{kDarkRed, kDarkGreen, kDarkBlue}; }
[[nodiscard]] QColor white_bar_color() { return QColor{kLightRed, kLightGreen, kLightBlue}; }

[[nodiscard]] QFont fitted_label_font(const QFont& base_font, const QString& text, int max_width) {
    QFont font = base_font;
    font.setBold(true);
    for (int point_size = font.pointSize(); point_size > kLabelMinPointSize; --point_size) {
        font.setPointSize(point_size);
        if (QFontMetrics{font}.horizontalAdvance(text) <= max_width) {
            return font;
        }
    }
    font.setPointSize(kLabelMinPointSize);
    return font;
}

} // namespace

EvaluationBarWidget::EvaluationBarWidget(QWidget* parent) : QWidget(parent) {
    setObjectName(QStringLiteral("evaluationBar"));
    setAccessibleName(QStringLiteral("Evaluation bar"));
    setAccessibleDescription(display_.text);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    setMinimumWidth(kMinimumBarWidth);
}

void EvaluationBarWidget::reset_to_neutral() { set_evaluation(gui::neutral_evaluation_display()); }

void EvaluationBarWidget::set_evaluation(const gui::EvaluationDisplay& display) {
    if (!display.has_value) {
        return;
    }
    display_ = display;
    setAccessibleDescription(display_.text);
    update();
}

void EvaluationBarWidget::set_reference_height(int height) {
    const int safe_height = std::max(kMinimumBarHeight, height);
    if (reference_height_ == safe_height) {
        return;
    }

    reference_height_ = safe_height;
    setFixedHeight(reference_height_);
    updateGeometry();
}

QSize EvaluationBarWidget::sizeHint() const {
    return QSize{kPreferredBarWidth,
                 reference_height_ > 0 ? reference_height_ : kPreferredBarHeight};
}

QSize EvaluationBarWidget::minimumSizeHint() const {
    return QSize{kMinimumBarWidth, kMinimumBarHeight};
}

void EvaluationBarWidget::paintEvent(QPaintEvent* event) {
    (void)event;
    QPainter painter{this};
    painter.setRenderHint(QPainter::Antialiasing, true);

    const QRect bar = rect().adjusted(kBarPadding, 0, -kBarPadding, -1);
    if (bar.isEmpty()) {
        return;
    }

    const double white_fraction = std::clamp(display_.white_fraction, 0.0, 1.0);
    const int white_height = static_cast<int>(std::lround(white_fraction * bar.height()));
    const int split_y = std::clamp(bar.bottom() - white_height + 1, bar.top(), bar.bottom());
    const QRect black_rect{bar.left(), bar.top(), bar.width(), split_y - bar.top()};
    const QRect white_rect{bar.left(), split_y, bar.width(), bar.bottom() - split_y + 1};

    painter.fillRect(black_rect, black_bar_color());
    painter.fillRect(white_rect, white_bar_color());

    painter.setPen(QPen{palette().mid().color(), kDivisionLineWidth});
    painter.drawLine(bar.left(), split_y, bar.right(), split_y);

    QColor border_color = palette().mid().color();
    border_color.setAlpha(kBorderAlpha);
    painter.setPen(QPen{border_color, kBorderLineWidth});
    painter.setBrush(Qt::NoBrush);
    painter.drawRect(bar);

    const QString label = display_.text.isEmpty() ? gui::format_evaluation_cp(0) : display_.text;
    QFont label_font = fitted_label_font(font(), label, bar.width() - (2 * kPillHorizontalPadding));
    const QFontMetrics metrics{label_font};
    const int pill_width =
        std::min(bar.width() - 2, metrics.horizontalAdvance(label) + (2 * kPillHorizontalPadding));
    const int pill_height = std::max(kPillMinHeight, metrics.height() + (2 * kPillVerticalPadding));
    const int pill_x = bar.left() + ((bar.width() - pill_width) / 2);
    const int pill_y =
        std::clamp(split_y - (pill_height / 2), bar.top() + 1, bar.bottom() - pill_height);
    const QRect pill{pill_x, pill_y, pill_width, pill_height};

    QColor pill_color = palette().window().color();
    pill_color.setAlpha(kPillAlpha);
    painter.setPen(QPen{palette().mid().color(), 1});
    painter.setBrush(pill_color);
    painter.drawRoundedRect(pill, kPillRadius, kPillRadius);

    painter.setFont(label_font);
    painter.setPen(palette().text().color());
    painter.drawText(pill, Qt::AlignCenter, label);
}
