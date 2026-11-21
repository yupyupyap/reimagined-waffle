#ifndef GUI_EVALUATION_DISPLAY_HPP
#define GUI_EVALUATION_DISPLAY_HPP

#include "engine_controller.hpp"

#include <eng/types.hpp>

#include <QString>

namespace gui {

inline constexpr int evaluation_graph_score_limit_cp = 1000;

struct EvaluationDisplay {
    bool has_value{false};
    bool has_graph_score{false};
    bool is_mate{false};
    int white_score_cp{0};
    int white_mate{0};
    int graph_score_cp{0};
    double white_fraction{0.5};
    EngineController::ScoreBound bound{EngineController::ScoreBound::Exact};
    QString text;
};

[[nodiscard]] EngineController::ScoreBound
invert_score_bound(EngineController::ScoreBound bound) noexcept;
[[nodiscard]] QString format_evaluation_cp(int score_cp);
[[nodiscard]] QString format_evaluation_mate(int mate);
[[nodiscard]] QString format_bound_prefix(EngineController::ScoreBound bound);
[[nodiscard]] double white_fraction_from_cp(int score_cp) noexcept;
[[nodiscard]] double white_fraction_from_mate(int mate) noexcept;
[[nodiscard]] EvaluationDisplay neutral_evaluation_display();
[[nodiscard]] EvaluationDisplay
evaluation_display_from_search_info(const EngineController::SearchInfo& info,
                                    eng::Color side_to_move);

} // namespace gui

#endif // GUI_EVALUATION_DISPLAY_HPP
