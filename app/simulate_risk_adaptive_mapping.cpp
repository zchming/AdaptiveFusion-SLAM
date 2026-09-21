#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "risk_adaptive_policy.h"

int main(int argc, char* argv[]) {
    if (argc > 2) {
        std::cerr << "Usage: simulate_risk_adaptive_mapping [output.csv]"
                  << std::endl;
        return 1;
    }
    const std::string output_path =
        argc == 2 ? argv[1] : "simulation_adaptive_mapping.csv";
    const std::vector<double> risks = {
        0.05, 0.08, 0.12, 0.18, 0.28, 0.35,
        0.48, 0.62, 0.72, 0.86, 0.92, 0.97,
    };
    constexpr std::size_t candidate_points_per_frame = 100;

    adaptive_fusion_slam::RiskAdaptivePolicy policy;
    std::size_t baseline_points = 0;
    std::size_t baseline_unreliable_points = 0;
    std::size_t adaptive_points = 0;
    std::size_t adaptive_unreliable_points = 0;
    std::size_t suppressed_points = 0;
    std::size_t early_keyframe_requests = 0;
    std::size_t frozen_frames = 0;

    std::ofstream output(output_path);
    if (!output) {
        std::cerr << "Cannot open adaptive mapping output: " << output_path
                  << std::endl;
        return 1;
    }
    output << "frame,risk,level,baseline_added,adaptive_added,suppressed,"
              "early_keyframe,map_frozen\n";
    for (std::size_t frame = 0; frame < risks.size(); ++frame) {
        const auto decision = policy.decide(risks[frame]);
        const bool unreliable =
            decision.level == adaptive_fusion_slam::RiskLevel::High ||
            decision.level == adaptive_fusion_slam::RiskLevel::Critical;
        baseline_points += candidate_points_per_frame;
        baseline_unreliable_points +=
            unreliable ? candidate_points_per_frame : 0;

        const std::size_t adaptive_added = decision.allow_new_map_points
            ? candidate_points_per_frame
            : 0;
        const std::size_t suppressed =
            candidate_points_per_frame - adaptive_added;
        adaptive_points += adaptive_added;
        adaptive_unreliable_points +=
            unreliable ? adaptive_added : 0;
        suppressed_points += suppressed;
        early_keyframe_requests += decision.request_early_keyframe ? 1U : 0U;
        frozen_frames += decision.allow_map_observations ? 0U : 1U;

        output << frame << ',' << risks[frame] << ','
               << adaptive_fusion_slam::riskLevelName(decision.level) << ','
               << candidate_points_per_frame << ',' << adaptive_added << ','
               << suppressed << ','
               << static_cast<int>(decision.request_early_keyframe) << ','
               << static_cast<int>(!decision.allow_map_observations) << '\n';
    }

    std::cout << "Baseline map points: " << baseline_points << '\n'
              << "Baseline unreliable points: "
              << baseline_unreliable_points << '\n'
              << "Adaptive map points: " << adaptive_points << '\n'
              << "Adaptive unreliable points: "
              << adaptive_unreliable_points << '\n'
              << "Suppressed risky points: " << suppressed_points << '\n'
              << "Early-keyframe requests: " << early_keyframe_requests << '\n'
              << "Frozen critical frames: " << frozen_frames << '\n'
              << "Simulation file: " << output_path << std::endl;
    return 0;
}
