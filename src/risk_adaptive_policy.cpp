#include "risk_adaptive_policy.h"

#include <cmath>
#include <stdexcept>

namespace adaptive_fusion_slam {

RiskAdaptivePolicy::RiskAdaptivePolicy(RiskAdaptivePolicyConfig config)
    : config_(config) {
    if (config_.medium_threshold <= 0.0 ||
        config_.medium_threshold >= config_.high_threshold ||
        config_.high_threshold >= config_.critical_threshold ||
        config_.critical_threshold >= 1.0) {
        throw std::invalid_argument(
            "Risk thresholds must be strictly ordered inside (0, 1).");
    }
}

RiskAdaptiveDecision RiskAdaptivePolicy::decide(
    double failure_probability) const {
    if (!std::isfinite(failure_probability) || failure_probability < 0.0 ||
        failure_probability > 1.0) {
        throw std::invalid_argument(
            "Failure probability must be finite and inside [0, 1].");
    }

    RiskAdaptiveDecision decision;
    decision.failure_probability = failure_probability;
    if (failure_probability >= config_.critical_threshold) {
        decision.level = RiskLevel::Critical;
        decision.enable_orb_verification = true;
        decision.force_orb_redetection = true;
        decision.allow_keyframe_insertion = false;
        decision.allow_map_observations = false;
        decision.allow_new_map_points = false;
        decision.preserve_trusted_reference = true;
    } else if (failure_probability >= config_.high_threshold) {
        decision.level = RiskLevel::High;
        decision.enable_orb_verification = true;
        decision.force_orb_redetection = true;
        decision.request_early_keyframe = true;
        decision.allow_new_map_points = false;
    } else if (failure_probability >= config_.medium_threshold) {
        decision.level = RiskLevel::Medium;
        decision.enable_orb_verification = true;
        decision.request_early_keyframe = true;
    }
    return decision;
}

const RiskAdaptivePolicyConfig& RiskAdaptivePolicy::config() const {
    return config_;
}

const char* riskLevelName(RiskLevel level) {
    switch (level) {
        case RiskLevel::Low:
            return "low";
        case RiskLevel::Medium:
            return "medium";
        case RiskLevel::High:
            return "high";
        case RiskLevel::Critical:
            return "critical";
    }
    return "unknown";
}

}  // namespace adaptive_fusion_slam
