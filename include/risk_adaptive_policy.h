#pragma once

namespace adaptive_fusion_slam {

enum class RiskLevel {
    Low,
    Medium,
    High,
    Critical,
};

struct RiskAdaptivePolicyConfig {
    double medium_threshold = 0.30;
    double high_threshold = 0.60;
    double critical_threshold = 0.85;
};

struct RiskAdaptiveDecision {
    RiskLevel level = RiskLevel::Low;
    double failure_probability = 0.0;
    bool enable_orb_verification = false;
    bool force_orb_redetection = false;
    bool request_early_keyframe = false;
    bool allow_keyframe_insertion = true;
    bool allow_map_observations = true;
    bool allow_new_map_points = true;
    bool preserve_trusted_reference = false;
};

class RiskAdaptivePolicy {
public:
    explicit RiskAdaptivePolicy(RiskAdaptivePolicyConfig config = {});

    RiskAdaptiveDecision decide(double failure_probability) const;
    const RiskAdaptivePolicyConfig& config() const;

private:
    RiskAdaptivePolicyConfig config_;
};

const char* riskLevelName(RiskLevel level);

}  // namespace adaptive_fusion_slam
