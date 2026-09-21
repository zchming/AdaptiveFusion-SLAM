#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include "failure_prediction_dataset.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << std::endl;
        return false;
    }
    return true;
}

}  // namespace

int main() {
    std::vector<adaptive_fusion_slam::GeometricHealth> sequence(12);
    for (std::size_t index = 0; index < sequence.size(); ++index) {
        auto& health = sequence[index];
        health.frame_id = index;
        health.timestamp = 0.1 * static_cast<double>(index);
        health.has_tracking_measurement = index > 0;
        health.tracking_success = index != 9;
        health.inlier_ratio = 0.9 - 0.05 * static_cast<double>(index);
        health.mean_reprojection_error_pixels =
            0.2 + 0.1 * static_cast<double>(index);
        health.mean_forward_backward_error_pixels = 0.1;
        health.spatial_coverage = 0.75;
        health.median_parallax_pixels = 4.0;
        health.valid_depth_ratio = 0.8;
        health.used_orb_fallback = index >= 8;
    }

    adaptive_fusion_slam::FailurePredictionDatasetConfig config;
    config.history_length = 3;
    config.prediction_horizon = 2;
    const adaptive_fusion_slam::FailurePredictionDatasetBuilder builder(config);
    const auto samples = builder.build(sequence);

    bool passed = true;
    passed &= check(samples.size() == 6,
                    "initialization and failed-history windows should be skipped");
    std::vector<adaptive_fusion_slam::FailurePredictionSample> positives;
    for (const auto& sample : samples) {
        if (sample.future_failure) {
            positives.push_back(sample);
        }
        passed &= check(sample.history.size() == 3,
                        "every sample should contain a fixed history length");
    }
    passed &= check(positives.size() == 2,
                    "two successful anchors should warn about frame nine");
    passed &= check(positives[0].anchor_frame_id == 7 &&
                        positives[0].frames_until_failure == 2 &&
                        positives[1].anchor_frame_id == 8 &&
                        positives[1].frames_until_failure == 1,
                    "lead times should identify the first future failure");
    passed &= check(std::abs(positives[1].history.back()[0] - 0.5) < 1e-12 &&
                        positives[1].history.back()[6] == 1.0,
                    "feature vectors should preserve health values and fallback");

    std::ostringstream csv;
    adaptive_fusion_slam::writeFailurePredictionCsv(
        csv, samples, config.history_length);
    passed &= check(csv.str().find("h2_valid_depth_ratio") != std::string::npos &&
                        csv.str().find("7,0.70000000000000007,1,2") !=
                            std::string::npos,
                    "flattened temporal samples should be serializable");

    if (!passed) {
        return 1;
    }
    std::cout << "Failure-dataset test passed with " << samples.size()
              << " usable windows and " << positives.size()
              << " positive warnings at lead times 2 and 1." << std::endl;
    return 0;
}
