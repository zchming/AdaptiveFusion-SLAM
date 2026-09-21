#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "failure_prediction_dataset.h"
#include "temporal_risk_predictor.h"

namespace {

std::vector<adaptive_fusion_slam::FailurePredictionSample> makeEpisode(
    std::size_t episode_id,
    std::size_t failure_frame,
    std::size_t degradation_length,
    bool has_failure) {
    constexpr std::size_t frame_count = 26;
    std::vector<adaptive_fusion_slam::GeometricHealth> sequence;
    sequence.reserve(frame_count);
    const double variation = 0.01 * static_cast<double>(episode_id % 5);
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        adaptive_fusion_slam::GeometricHealth health;
        health.frame_id = episode_id * 100 + frame;
        health.timestamp = 0.033 * static_cast<double>(frame);
        health.has_tracking_measurement = frame > 0;
        health.tracking_success = !has_failure || frame != failure_frame;
        double severity = 0.0;
        if (has_failure && frame + degradation_length >= failure_frame &&
            frame <= failure_frame) {
            severity = static_cast<double>(
                frame + degradation_length - failure_frame) /
                static_cast<double>(degradation_length);
        }
        severity = std::clamp(severity, 0.0, 1.0);
        health.inlier_ratio = 0.94 - 0.62 * severity + variation;
        health.mean_reprojection_error_pixels =
            0.28 + 2.75 * severity + variation;
        health.mean_forward_backward_error_pixels =
            0.10 + 0.90 * severity;
        health.spatial_coverage = 0.90 - 0.55 * severity;
        health.median_parallax_pixels = 5.2 - 3.4 * severity;
        health.valid_depth_ratio = 0.95 - 0.55 * severity;
        health.used_orb_fallback = severity > 0.68;
        sequence.push_back(health);
    }
    adaptive_fusion_slam::FailurePredictionDatasetBuilder builder;
    return builder.build(sequence);
}

void append(
    std::vector<adaptive_fusion_slam::FailurePredictionSample>& destination,
    std::vector<adaptive_fusion_slam::FailurePredictionSample> source) {
    destination.insert(
        destination.end(), source.begin(), source.end());
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc > 3) {
        std::cerr << "Usage: simulate_risk_prediction [predictions.csv] "
                     "[model.txt]"
                  << std::endl;
        return 1;
    }
    const std::string output_path =
        argc >= 2 ? argv[1] : "simulation_risk_predictions.csv";
    const std::string model_path = argc == 3
        ? argv[2]
        : "simulation_risk_model.txt";

    std::vector<adaptive_fusion_slam::FailurePredictionSample> training;
    for (std::size_t episode = 0; episode < 10; ++episode) {
        append(training, makeEpisode(
            episode, 14 + episode % 5, 5 + episode % 4, true));
    }
    for (std::size_t episode = 10; episode < 14; ++episode) {
        append(training, makeEpisode(episode, 18, 6, false));
    }

    std::vector<adaptive_fusion_slam::FailurePredictionSample> testing;
    for (std::size_t episode = 20; episode < 25; ++episode) {
        append(testing, makeEpisode(
            episode, 15 + episode % 4, 6 + episode % 3, true));
    }
    for (std::size_t episode = 25; episode < 28; ++episode) {
        append(testing, makeEpisode(episode, 18, 6, false));
    }

    adaptive_fusion_slam::TemporalRiskPredictor predictor;
    predictor.train(training);
    std::ofstream model_output(model_path);
    if (!model_output) {
        std::cerr << "Cannot open model output: " << model_path << std::endl;
        return 1;
    }
    predictor.save(model_output);
    const auto probabilities = predictor.predictProbabilities(testing);
    const auto metrics = adaptive_fusion_slam::evaluateRiskPredictions(
        testing, probabilities, predictor.decisionThreshold());

    std::ofstream output(output_path);
    if (!output) {
        std::cerr << "Cannot open prediction output: " << output_path << std::endl;
        return 1;
    }
    output << "anchor_frame_id,future_failure,frames_until_failure,risk\n";
    for (std::size_t index = 0; index < testing.size(); ++index) {
        output << testing[index].anchor_frame_id << ','
               << static_cast<int>(testing[index].future_failure) << ','
               << testing[index].frames_until_failure << ','
               << probabilities[index] << '\n';
    }

    std::cout << "Training samples: " << training.size() << '\n'
              << "Held-out test samples: " << testing.size() << '\n'
              << "Test positives: " << metrics.positive_count << '\n'
              << "Accuracy: " << metrics.accuracy << '\n'
              << "AUROC: " << metrics.auroc << '\n'
              << "AUPRC: " << metrics.auprc << '\n'
              << "Brier score: " << metrics.brier_score << '\n'
              << "Mean warning lead: " << metrics.mean_warning_lead_frames
              << " frames\n"
              << "Model file: " << model_path << '\n'
              << "Prediction file: " << output_path << std::endl;
    return 0;
}
