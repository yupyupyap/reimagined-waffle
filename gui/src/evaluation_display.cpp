#include "evaluation_display.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace gui {
namespace {

constexpr double kEvaluationBarMaxSwing = 0.46;
constexpr double kEvaluationBarScaleCp = 360.0;
constexpr double kEvaluationBarMinFraction = 0.04;
constexpr double kEvaluationBarMaxFraction = 0.96;

[[nodiscard]] bool should_invert_for_side(eng::Color side_to_move) noexcept {
    return side_to_move == eng::Color::Black;
}

[[nodiscard]] QString prefixed_score_text(EngineController::ScoreBound bound,
                                          const QString& score) {
    return QStringLiteral("%1%2").arg(format_bound_prefix(bound), score);
}

} // namespace

EngineController::ScoreBound invert_score_bound(EngineController::ScoreBound bound) noexcept {
    switch (bound) {
    case EngineController::ScoreBound::Lower:
        return EngineController::ScoreBound::Upper;
    case EngineController::ScoreBound::Upper:
        return EngineController::ScoreBound::Lower;
    case EngineController::ScoreBound::Exact:
        break;
    }
    return EngineController::ScoreBound::Exact;
}

QString format_evaluation_cp(int score_cp) {
    const double pawns = static_cast<double>(score_cp) / 100.0;
    return QStringLiteral("%1%2")
        .arg(score_cp >= 0 ? QStringLiteral("+") : QString{})
        .arg(pawns, 0, 'f', 2);
}

QString format_evaluation_mate(int mate) {
    return mate < 0 ? QStringLiteral("-M%1").arg(std::abs(mate)) : QStringLiteral("+M%1").arg(mate);
}

QString format_bound_prefix(EngineController::ScoreBound bound) {
    switch (bound) {
    case EngineController::ScoreBound::Lower:
        return QStringLiteral(">= ");
    case EngineController::ScoreBound::Upper:
        return QStringLiteral("<= ");
    case EngineController::ScoreBound::Exact:
        break;
    }
    return {};
}

double white_fraction_from_cp(int score_cp) noexcept {
    const double swing =
        kEvaluationBarMaxSwing * std::tanh(static_cast<double>(score_cp) / kEvaluationBarScaleCp);
    return std::clamp(0.5 + swing, kEvaluationBarMinFraction, kEvaluationBarMaxFraction);
}

double white_fraction_from_mate(int mate) noexcept {
    if (mate > 0) {
        return kEvaluationBarMaxFraction;
    }
    if (mate < 0) {
        return kEvaluationBarMinFraction;
    }
    return 0.5;
}

EvaluationDisplay neutral_evaluation_display() {
    return EvaluationDisplay{.has_value = true,
                             .has_graph_score = false,
                             .is_mate = false,
                             .white_score_cp = 0,
                             .white_mate = 0,
                             .graph_score_cp = 0,
                             .white_fraction = 0.5,
                             .bound = EngineController::ScoreBound::Exact,
                             .text = format_evaluation_cp(0)};
}

EvaluationDisplay evaluation_display_from_search_info(const EngineController::SearchInfo& info,
                                                      eng::Color side_to_move) {
    const bool invert_score = should_invert_for_side(side_to_move);
    const EngineController::ScoreBound bound =
        invert_score ? invert_score_bound(info.bound) : info.bound;

    if (info.has_score_cp) {
        const int white_score = invert_score ? -info.score_cp : info.score_cp;
        return EvaluationDisplay{
            .has_value = true,
            .has_graph_score = true,
            .is_mate = false,
            .white_score_cp = white_score,
            .white_mate = 0,
            .graph_score_cp = std::clamp(white_score, -evaluation_graph_score_limit_cp,
                                         evaluation_graph_score_limit_cp),
            .white_fraction = white_fraction_from_cp(white_score),
            .bound = bound,
            .text = prefixed_score_text(bound, format_evaluation_cp(white_score)),
        };
    }

    if (info.has_mate) {
        const int white_mate = invert_score ? -info.mate : info.mate;
        return EvaluationDisplay{
            .has_value = true,
            .has_graph_score = true,
            .is_mate = true,
            .white_score_cp = 0,
            .white_mate = white_mate,
            .graph_score_cp =
                white_mate < 0 ? -evaluation_graph_score_limit_cp : evaluation_graph_score_limit_cp,
            .white_fraction = white_fraction_from_mate(white_mate),
            .bound = bound,
            .text = prefixed_score_text(bound, format_evaluation_mate(white_mate)),
        };
    }

    return EvaluationDisplay{.bound = bound};
}

} // namespace gui
