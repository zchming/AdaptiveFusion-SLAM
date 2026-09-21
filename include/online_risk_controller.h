#pragma once

#include <deque>

#include "geometric_health.h"
#include "risk_adaptive_policy.h"
#include "temporal_risk_predictor.h"

namespace adaptive_fusion_slam {

struct OnlineRiskResult {
    bool prediction_available = false;
    double failure_probability = 0.0;
    RiskAdaptiveDecision decision;
};

class OnlineRiskController {
public:
    OnlineRiskController(
        TemporalRiskPredictor predictor,
        RiskAdaptivePolicyConfig policy_config = {},
        double hysteresis_margin = 0.05);

    OnlineRiskResult observe(const GeometricHealth& health);
    void reset();

private:
    TemporalRiskPredictor predictor_;
    HystereticRiskAdaptivePolicy policy_;
    std::deque<GeometricHealth> history_;
};

}  // namespace adaptive_fusion_slam
