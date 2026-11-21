#ifndef GUI_EVALUATION_BAR_WIDGET_HPP
#define GUI_EVALUATION_BAR_WIDGET_HPP

#include "evaluation_display.hpp"

#include <QWidget>

class EvaluationBarWidget final : public QWidget {
  public:
    explicit EvaluationBarWidget(QWidget* parent = nullptr);
    ~EvaluationBarWidget() override = default;
    EvaluationBarWidget(const EvaluationBarWidget&) = delete;
    EvaluationBarWidget& operator=(const EvaluationBarWidget&) = delete;
    EvaluationBarWidget(EvaluationBarWidget&&) = delete;
    EvaluationBarWidget& operator=(EvaluationBarWidget&&) = delete;

    void reset_to_neutral();
    void set_evaluation(const gui::EvaluationDisplay& display);
    void set_reference_height(int height);

    [[nodiscard]] QSize sizeHint() const override;
    [[nodiscard]] QSize minimumSizeHint() const override;

  protected:
    void paintEvent(QPaintEvent* event) override;

  private:
    gui::EvaluationDisplay display_{gui::neutral_evaluation_display()};
    int reference_height_{0};
};

#endif // GUI_EVALUATION_BAR_WIDGET_HPP
