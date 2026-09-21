#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "failure_prediction_dataset.h"

int main(int argc, char* argv[]) {
    if (argc > 2) {
        std::cerr << "Usage: simulate_failure_dataset [output.csv]" << std::endl;
        return 1;
    }
    const std::string output_path =
        argc == 2 ? argv[1] : "simulation_failure_dataset.csv";

    std::vector<adaptive_fusion_slam::GeometricHealth> sequence;
    for (std::size_t frame = 0; frame < 24; ++frame) {
        adaptive_fusion_slam::GeometricHealth health;
        health.frame_id = frame;
        health.timestamp = 0.033 * static_cast<double>(frame);
        health.has_tracking_measurement = frame > 0;
        health.tracking_success = frame != 15;
        health.feature_count = 900;

        double degradation = 0.0;
        if (frame >= 9 && frame <= 15) {
            degradation = static_cast<double>(frame - 8) / 7.0;
        }
        degradation = std::clamp(degradation, 0.0, 1.0);
        health.inlier_ratio = 0.95 - 0.65 * degradation;
        health.mean_reprojection_error_pixels = 0.30 + 2.70 * degradation;
        health.mean_forward_backward_error_pixels = 0.10 + 0.90 * degradation;
        health.spatial_coverage = 0.92 - 0.58 * degradation;
        health.median_parallax_pixels = 5.0 - 3.5 * degradation;
        health.valid_depth_ratio = 0.96 - 0.56 * degradation;
        health.used_orb_fallback = frame >= 13 && frame <= 15;
        health.correspondence_count = static_cast<std::size_t>(
            700.0 - 450.0 * degradation);
        health.pnp_inlier_count = static_cast<std::size_t>(
            health.correspondence_count * health.inlier_ratio);
        sequence.push_back(health);
    }

    adaptive_fusion_slam::FailurePredictionDatasetConfig config;
    config.history_length = 5;
    config.prediction_horizon = 3;
    const adaptive_fusion_slam::FailurePredictionDatasetBuilder builder(config);
    const auto samples = builder.build(sequence);

    std::ofstream output(output_path);
    if (!output) {
        std::cerr << "Cannot open simulation output: " << output_path << std::endl;
        return 1;
    }
    adaptive_fusion_slam::writeFailurePredictionCsv(
        output, samples, config.history_length);

    const auto positive_count = static_cast<std::size_t>(std::count_if(
        samples.begin(), samples.end(), [](const auto& sample) {
            return sample.future_failure;
        }));
    std::cout << "Synthetic health frames: " << sequence.size() << '\n'
              << "History length L: " << config.history_length << '\n'
              << "Prediction horizon H: " << config.prediction_horizon << '\n'
              << "Usable samples: " << samples.size() << '\n'
              << "Positive future-failure samples: " << positive_count << '\n'
              << "Negative samples: " << samples.size() - positive_count << '\n';
    for (const auto& sample : samples) {
        if (sample.future_failure) {
            const auto& latest = sample.history.back();
            std::cout << "Warning sample at frame " << sample.anchor_frame_id
                      << ": failure in " << sample.frames_until_failure
                      << " frame(s), current inlier ratio " << latest[0]
                      << ", reprojection error " << latest[1]
                      << " px, fallback " << latest[6] << '\n';
        }
    }
    std::cout << "Dataset file: " << output_path << std::endl;
    return 0;
}
