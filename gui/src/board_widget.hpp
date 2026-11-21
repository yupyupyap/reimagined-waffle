#ifndef GUI_BOARD_WIDGET_HPP
#define GUI_BOARD_WIDGET_HPP

#include "eng/move.hpp"
#include "eng/position.hpp"
#include "eng/types.hpp"

#include <QHash>
#include <QPixmap>
#include <QPoint>
#include <QRect>
#include <QString>
#include <QSvgRenderer>
#include <QWidget>

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

class QMouseEvent;
class QPaintEvent;
class QResizeEvent;

class BoardWidget final : public QWidget {
    Q_OBJECT

  public:
    explicit BoardWidget(QWidget* parent = nullptr);
    ~BoardWidget() override;
    BoardWidget(const BoardWidget&) = delete;
    BoardWidget& operator=(const BoardWidget&) = delete;
    BoardWidget(BoardWidget&&) = delete;
    BoardWidget& operator=(BoardWidget&&) = delete;

    void set_position(const eng::Position& position);
    void set_orientation(eng::Color bottom_color);
    void set_active_side(eng::Color color);
    void set_selected_square(std::optional<eng::Square> square);
    void set_legal_destinations(std::span<const eng::Square> squares);
    void set_check_square(std::optional<eng::Square> square);
    void set_last_move(std::optional<eng::Move> move);
    void clear_interaction();

    [[nodiscard]] QSize sizeHint() const override;
    [[nodiscard]] QSize minimumSizeHint() const override;

  signals:
    void squareSelected(int square_index);
    void squareClicked(int square_index);
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void moveGestureCompleted(int from_square_index, int to_square_index);
    void boardGeometryChanged(QRect board_rect);

  protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

  private:
    struct SquarePair {
        eng::Square from;
        eng::Square to;
    };

    void load_piece_renderers();
    static void draw_board(QPainter& painter, QRect board);
    void draw_highlights(QPainter& painter, QRect board, int square_size) const;
    void draw_pieces(QPainter& painter, QRect board, int square_size) const;
    void draw_piece(QPainter& painter, eng::Piece piece, QRect square_rect,
                    qreal opacity = 1.0) const;
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void emit_move_gesture(eng::Square from_square, eng::Square to_square);

    [[nodiscard]] QRect board_rect() const;
    [[nodiscard]] QRect square_rect(eng::Square square, QRect board, int square_size) const;
    [[nodiscard]] std::optional<eng::Square> square_at(QPoint point) const;
    [[nodiscard]] QPixmap piece_pixmap(eng::Piece piece, QSize logical_size) const;

    eng::Position position_{eng::Position::startpos()};
    eng::Color bottom_color_{eng::Color::White};
    std::optional<eng::Square> selected_square_;
    std::optional<eng::Square> pressed_square_;
    std::optional<eng::Square> hover_square_;
    std::optional<eng::Square> check_square_;
    std::optional<SquarePair> highlighted_move_;
    std::vector<eng::Square> legal_destinations_;
    QPoint drag_start_;
    QPoint drag_position_;
    bool dragging_{false};
    std::array<std::unique_ptr<QSvgRenderer>, eng::Piece::count> piece_renderers_{};
    mutable QHash<QString, QPixmap> pixmap_cache_;
};

#endif // GUI_BOARD_WIDGET_HPP
