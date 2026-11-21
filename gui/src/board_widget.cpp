#include "board_widget.hpp"

#include <QApplication>
#include <QColor>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QPen>
#include <QResizeEvent>
#include <QSizePolicy>
#include <QSvgRenderer>

#include <algorithm>
#include <cmath>
#include <utility>

namespace {

constexpr int kBoardFiles = 8;
constexpr int kBoardRanks = 8;
constexpr int kPreferredBoardSize = 640;
constexpr int kMinimumBoardSize = 320;
constexpr int kLegalQuietHintMinimumDiameter = 8;
constexpr int kLegalQuietHintDivisor = 5;
constexpr int kLegalCaptureHintMinimumInset = 4;
constexpr int kLegalCaptureHintInsetDivisor = 12;
constexpr int kMinimumCaptureHintPenWidth = 3;
constexpr int kCaptureHintPenDivisor = 18;
constexpr qreal kPieceInsetRatio = 0.075;
constexpr qreal kDragPieceOpacity = 0.82;

constexpr std::array<const char*, eng::Piece::count> kPieceResourcePaths{
    ":/pieces/white_pawn.svg", ":/pieces/white_knight.svg", ":/pieces/white_bishop.svg",
    ":/pieces/white_rook.svg", ":/pieces/white_queen.svg",  ":/pieces/white_king.svg",
    ":/pieces/black_pawn.svg", ":/pieces/black_knight.svg", ":/pieces/black_bishop.svg",
    ":/pieces/black_rook.svg", ":/pieces/black_queen.svg",  ":/pieces/black_king.svg",
};

[[nodiscard]] QColor light_square_color() { return QColor{QStringLiteral("#ebe2ce")}; }
[[nodiscard]] QColor dark_square_color() { return QColor{QStringLiteral("#658a73")}; }
[[nodiscard]] QColor board_border_color() { return QColor{QStringLiteral("#262d2a")}; }
[[nodiscard]] QColor last_move_color() { return QColor{QStringLiteral("#68f5c645")}; }
[[nodiscard]] QColor selected_square_color() { return QColor{QStringLiteral("#7e4682b4")}; }
[[nodiscard]] QColor selected_border_color() { return QColor{QStringLiteral("#23527c")}; }
[[nodiscard]] QColor hover_square_color() { return QColor{QStringLiteral("#3effffff")}; }
[[nodiscard]] QColor legal_quiet_hint_color() { return QColor{QStringLiteral("#2d4148a0")}; }
[[nodiscard]] QColor legal_capture_hint_color() { return QColor{QStringLiteral("#f6f0dccc")}; }
[[nodiscard]] QColor check_square_color() { return QColor{QStringLiteral("#d84f4f9c")}; }

[[nodiscard]] int device_ratio_key(qreal ratio) {
    return static_cast<int>(std::lround(ratio * 100.0));
}

[[nodiscard]] bool is_left_button(QMouseEvent* event) {
    return event != nullptr && event->button() == Qt::LeftButton;
}

} // namespace

BoardWidget::BoardWidget(QWidget* parent) : QWidget(parent) {
    setObjectName(QStringLiteral("boardWidget"));
    setAccessibleName(QStringLiteral("Chess board"));
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    load_piece_renderers();
}

BoardWidget::~BoardWidget() = default;

void BoardWidget::set_position(const eng::Position& position) {
    position_ = position;
    clear_interaction();
    update();
}

void BoardWidget::set_orientation(eng::Color bottom_color) {
    if (!eng::is_valid_color(bottom_color) || bottom_color_ == bottom_color) {
        return;
    }

    bottom_color_ = bottom_color;
    update();
}

void BoardWidget::set_active_side(eng::Color color) {
    if (!eng::is_valid_color(color)) {
        return;
    }

    setAccessibleDescription(color == eng::Color::White ? QStringLiteral("White to move")
                                                        : QStringLiteral("Black to move"));
    update();
}

void BoardWidget::set_selected_square(std::optional<eng::Square> square) {
    if (square.has_value() && !square->is_valid()) {
        square.reset();
    }

    selected_square_ = square;
    update();
}

void BoardWidget::set_legal_destinations(std::span<const eng::Square> squares) {
    legal_destinations_.clear();
    legal_destinations_.reserve(squares.size());
    for (const auto square : squares) {
        if (square.is_valid() &&
            std::ranges::find(legal_destinations_, square) == legal_destinations_.end()) {
            legal_destinations_.push_back(square);
        }
    }
    update();
}

void BoardWidget::set_check_square(std::optional<eng::Square> square) {
    if (square.has_value() && !square->is_valid()) {
        square.reset();
    }

    check_square_ = square;
    update();
}

void BoardWidget::set_last_move(std::optional<eng::Move> move) {
    if (move.has_value() && move->is_valid() && !move->is_null()) {
        highlighted_move_ = SquarePair{.from = move->from(), .to = move->to()};
    } else {
        highlighted_move_.reset();
    }
    update();
}

void BoardWidget::clear_interaction() {
    selected_square_.reset();
    pressed_square_.reset();
    hover_square_.reset();
    check_square_.reset();
    highlighted_move_.reset();
    legal_destinations_.clear();
    dragging_ = false;
}

QSize BoardWidget::sizeHint() const { return QSize{kPreferredBoardSize, kPreferredBoardSize}; }

QSize BoardWidget::minimumSizeHint() const { return QSize{kMinimumBoardSize, kMinimumBoardSize}; }

void BoardWidget::paintEvent(QPaintEvent* event) {
    QPainter painter(this);
    painter.setClipRect(event->rect());
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);

    const QRect board = board_rect();
    painter.fillRect(rect(), palette().window());
    if (board.isEmpty()) {
        return;
    }

    const int square_size = board.width() / kBoardFiles;
    draw_board(painter, board);
    draw_highlights(painter, board, square_size);
    draw_pieces(painter, board, square_size);
}

void BoardWidget::mousePressEvent(QMouseEvent* event) {
    if (!is_left_button(event)) {
        QWidget::mousePressEvent(event);
        return;
    }

    const auto square = square_at(event->position().toPoint());
    if (!square.has_value()) {
        clear_interaction();
        update();
        return;
    }

    pressed_square_ = square;
    hover_square_ = square;
    drag_start_ = event->position().toPoint();
    drag_position_ = drag_start_;
    dragging_ = false;

    update();
}

void BoardWidget::mouseMoveEvent(QMouseEvent* event) {
    if (event == nullptr || !pressed_square_.has_value()) {
        QWidget::mouseMoveEvent(event);
        return;
    }

    drag_position_ = event->position().toPoint();
    hover_square_ = square_at(drag_position_);

    const int movement = (drag_position_ - drag_start_).manhattanLength();
    if (!dragging_ && movement >= QApplication::startDragDistance() &&
        !position_.piece_on(*pressed_square_).is_none()) {
        dragging_ = true;
        emit squareSelected(pressed_square_->index());
    }

    update();
}

void BoardWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (!is_left_button(event)) {
        QWidget::mouseReleaseEvent(event);
        return;
    }

    const auto release_square = square_at(event->position().toPoint());
    const auto press_square = pressed_square_;
    pressed_square_.reset();
    hover_square_.reset();

    if (!release_square.has_value()) {
        dragging_ = false;
        update();
        return;
    }

    if (dragging_) {
        dragging_ = false;
        if (press_square.has_value() && *press_square != *release_square &&
            !position_.piece_on(*press_square).is_none()) {
            emit_move_gesture(*press_square, *release_square);
        }
        update();
        return;
    }

    if (!press_square.has_value() || *press_square != *release_square) {
        update();
        return;
    }

    emit squareClicked(release_square->index());
    update();
}

void BoardWidget::resizeEvent(QResizeEvent* event) {
    pixmap_cache_.clear();
    QWidget::resizeEvent(event);
    emit boardGeometryChanged(board_rect());
}

void BoardWidget::load_piece_renderers() {
    for (int index = 0; index < eng::Piece::count; ++index) {
        piece_renderers_[index] =
            std::make_unique<QSvgRenderer>(QString::fromLatin1(kPieceResourcePaths[index]));
    }
}

void BoardWidget::draw_board(QPainter& painter, QRect board) {
    const int square_size = board.width() / kBoardFiles;

    for (int row = 0; row < kBoardRanks; ++row) {
        for (int column = 0; column < kBoardFiles; ++column) {
            const bool light_square = ((row + column) % 2) == 0;
            const QRect square{board.left() + (column * square_size),
                               board.top() + (row * square_size), square_size, square_size};
            painter.fillRect(square, light_square ? light_square_color() : dark_square_color());
        }
    }

    painter.setPen(QPen{board_border_color(), 2});
    painter.setBrush(Qt::NoBrush);
    painter.drawRect(board.adjusted(0, 0, -1, -1));
}

void BoardWidget::draw_highlights(QPainter& painter, QRect board, int square_size) const {
    const auto fill_square = [&](eng::Square square, QColor color) {
        painter.fillRect(square_rect(square, board, square_size), color);
    };

    if (highlighted_move_.has_value()) {
        fill_square(highlighted_move_->from, last_move_color());
        fill_square(highlighted_move_->to, last_move_color());
    }

    if (check_square_.has_value()) {
        fill_square(*check_square_, check_square_color());
    }

    painter.save();
    painter.setRenderHint(QPainter::Antialiasing, true);
    for (const auto destination : legal_destinations_) {
        const QRect destination_rect = square_rect(destination, board, square_size);
        const eng::Piece target_piece = position_.piece_on(destination);
        if (target_piece.is_none()) {
            const int diameter =
                std::max(kLegalQuietHintMinimumDiameter, square_size / kLegalQuietHintDivisor);
            const QPoint center = destination_rect.center();
            painter.setPen(Qt::NoPen);
            painter.setBrush(legal_quiet_hint_color());
            painter.drawEllipse(center, diameter / 2, diameter / 2);
        } else {
            const int inset = std::max(kLegalCaptureHintMinimumInset,
                                       square_size / kLegalCaptureHintInsetDivisor);
            painter.setPen(
                QPen{legal_capture_hint_color(),
                     static_cast<qreal>(std::max(kMinimumCaptureHintPenWidth,
                                                 square_size / kCaptureHintPenDivisor))});
            painter.setBrush(Qt::NoBrush);
            painter.drawEllipse(destination_rect.adjusted(inset, inset, -inset, -inset));
        }
    }
    painter.restore();

    if (selected_square_.has_value()) {
        const QRect selected_rect = square_rect(*selected_square_, board, square_size);
        painter.fillRect(selected_rect, selected_square_color());
        painter.setPen(QPen{selected_border_color(), 3});
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(selected_rect.adjusted(2, 2, -3, -3));
    }

    if (hover_square_.has_value() && dragging_) {
        fill_square(*hover_square_, hover_square_color());
    }
}

void BoardWidget::draw_pieces(QPainter& painter, QRect board, int square_size) const {
    for (int square_index = 0; square_index < eng::Square::count; ++square_index) {
        const eng::Square square = eng::Square::from_index(square_index);
        const eng::Piece piece = position_.piece_on(square);
        if (piece.is_none()) {
            continue;
        }
        if (dragging_ && pressed_square_.has_value() && *pressed_square_ == square) {
            continue;
        }

        draw_piece(painter, piece, square_rect(square, board, square_size));
    }

    if (dragging_ && pressed_square_.has_value()) {
        const eng::Piece dragged_piece = position_.piece_on(*pressed_square_);
        if (!dragged_piece.is_none()) {
            const int side = square_size;
            const QRect drag_rect{drag_position_.x() - (side / 2), drag_position_.y() - (side / 2),
                                  side, side};
            draw_piece(painter, dragged_piece, drag_rect, kDragPieceOpacity);
        }
    }
}

void BoardWidget::draw_piece(QPainter& painter, eng::Piece piece, QRect square_rect,
                             qreal opacity) const {
    const int inset = static_cast<int>(std::round(square_rect.width() * kPieceInsetRatio));
    const QRect piece_rect = square_rect.adjusted(inset, inset, -inset, -inset);
    const QPixmap pixmap = piece_pixmap(piece, piece_rect.size());

    painter.save();
    painter.setOpacity(opacity);
    if (!pixmap.isNull()) {
        painter.drawPixmap(piece_rect.topLeft(), pixmap);
    }
    painter.restore();
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void BoardWidget::emit_move_gesture(eng::Square from_square, eng::Square to_square) {
    emit moveGestureCompleted(from_square.index(), to_square.index());
}

QRect BoardWidget::board_rect() const {
    const int usable_side = std::min(width(), height());
    const int square_size = usable_side / kBoardFiles;
    if (square_size <= 0) {
        return {};
    }

    const int board_side = square_size * kBoardFiles;
    return QRect{(width() - board_side) / 2, (height() - board_side) / 2, board_side, board_side};
}

QRect BoardWidget::square_rect(eng::Square square, QRect board, int square_size) const {
    const int file = square.file().index();
    const int rank = square.rank().index();
    const int column = bottom_color_ == eng::Color::White ? file : (kBoardFiles - 1 - file);
    const int row = bottom_color_ == eng::Color::White ? (kBoardRanks - 1 - rank) : rank;

    return QRect{board.left() + (column * square_size), board.top() + (row * square_size),
                 square_size, square_size};
}

std::optional<eng::Square> BoardWidget::square_at(QPoint point) const {
    const QRect board = board_rect();
    if (board.isEmpty() || !board.contains(point)) {
        return std::nullopt;
    }

    const int square_size = board.width() / kBoardFiles;
    if (square_size <= 0) {
        return std::nullopt;
    }
    const int column = (point.x() - board.left()) / square_size;
    const int row = (point.y() - board.top()) / square_size;
    if (column < 0 || column >= kBoardFiles || row < 0 || row >= kBoardRanks) {
        return std::nullopt;
    }

    const int file = bottom_color_ == eng::Color::White ? column : (kBoardFiles - 1 - column);
    const int rank = bottom_color_ == eng::Color::White ? (kBoardRanks - 1 - row) : row;
    return eng::Square::from_file_rank(eng::File::from_index(file), eng::Rank::from_index(rank));
}

QPixmap BoardWidget::piece_pixmap(eng::Piece piece, QSize logical_size) const {
    if (piece.is_none() || logical_size.isEmpty()) {
        return {};
    }

    const qreal dpr = devicePixelRatioF();
    const QSize pixel_size =
        QSize{std::max(1, static_cast<int>(std::ceil(logical_size.width() * dpr))),
              std::max(1, static_cast<int>(std::ceil(logical_size.height() * dpr)))};
    const QString cache_key = QStringLiteral("%1:%2x%3@%4")
                                  .arg(piece.index())
                                  .arg(pixel_size.width())
                                  .arg(pixel_size.height())
                                  .arg(device_ratio_key(dpr));

    const auto cached = pixmap_cache_.constFind(cache_key);
    if (cached != pixmap_cache_.cend()) {
        return cached.value();
    }

    QPixmap pixmap{pixel_size};
    pixmap.setDevicePixelRatio(dpr);
    pixmap.fill(Qt::transparent);

    QPainter pixmap_painter{&pixmap};
    pixmap_painter.setRenderHint(QPainter::Antialiasing, true);
    pixmap_painter.setRenderHint(QPainter::SmoothPixmapTransform, true);

    const auto& renderer = piece_renderers_[piece.index()];
    if (renderer != nullptr && renderer->isValid()) {
        renderer->render(&pixmap_painter, QRectF{QPointF{0.0, 0.0}, QSizeF{logical_size}});
    } else {
        pixmap_painter.setPen(QPen{board_border_color(), 2});
        pixmap_painter.setBrush(piece.color() == eng::Color::White ? Qt::white : Qt::black);
        const QRectF fallback_rect{QPointF{4.0, 4.0},
                                   QSizeF{std::max(1.0, logical_size.width() - 8.0),
                                          std::max(1.0, logical_size.height() - 8.0)}};
        pixmap_painter.drawEllipse(fallback_rect);
    }

    pixmap_cache_.insert(cache_key, pixmap);
    return pixmap;
}
