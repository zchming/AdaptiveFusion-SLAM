#include "online_risk_controller.h"

#include <stdexcept>
#include <utility>

namespace adaptive_fusion_slam {

OnlineRiskController::OnlineRiskController(
    TemporalRiskPredictor predictor,
    RiskAdaptivePolicyConfig policy_config,
    double hysteresis_margin)
    : predictor_(std::move(predictor)),
      policy_(policy_config, hysteresis_margin) {
    if (!predictor_.isTrained() || predictor_.historyLength() == 0) {
        throw std::invalid_argument(
            "Online risk controller requires a trained predictor.");
    }
}

OnlineRiskResult OnlineRiskController::observe(
    const GeometricHealth& health) {
    OnlineRiskResult result;
    if (!health.has_tracking_measurement) {
        result.decision = policy_.update(0.0);
        return result;
    }
    if (!health.tracking_success) {
        history_.clear();
        result.failure_probability = 1.0;
        result.decision = policy_.update(1.0);
        return result;
    }

    history_.push_back(health);
    while (history_.size() > predictor_.historyLength()) {
        history_.pop_front();
    }
    if (history_.size() < predictor_.historyLength()) {
        result.decision = policy_.update(0.0);
        return result;
    }

    FailurePredictionSample sample;
    sample.anchor_frame_id = health.frame_id;
    sample.anchor_timestamp = health.timestamp;
    sample.history.reserve(history_.size());
    for (const auto& item : history_) {
        sample.history.push_back(makeHealthFeatureVector(item));
    }
    result.prediction_available = true;
    result.failure_probability = predictor_.predictProbability(sample);
    result.decision = policy_.update(result.failure_probability);
    return result;
}

void OnlineRiskController::reset() {
    history_.clear();
    policy_.reset();
}

}  // namespace adaptive_fusion_slam
