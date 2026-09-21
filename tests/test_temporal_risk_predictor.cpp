#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "temporal_risk_predictor.h"

namespace {

adaptive_fusion_slam::FailurePredictionSample makeSample(
    double severity,
    bool label,
    std::size_t lead) {
    adaptive_fusion_slam::FailurePredictionSample sample;
    sample.future_failure = label;
    sample.frames_until_failure = lead;
    for (std::size_t step = 0; step < 5; ++step) {
        const double temporal = severity * static_cast<double>(step + 1) / 5.0;
        sample.history.push_back({
            0.95 - 0.55 * temporal,
            0.25 + 2.5 * temporal,
            0.10 + 0.8 * temporal,
            0.90 - 0.45 * temporal,
            5.0 - 2.5 * temporal,
            0.95 - 0.45 * temporal,
            temporal > 0.7 ? 1.0 : 0.0,
        });
    }
    return sample;
}

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << std::endl;
        return false;
    }
    return true;
}

}  // namespace

int main() {
    std::vector<adaptive_fusion_slam::FailurePredictionSample> training;
    for (std::size_t index = 0; index < 20; ++index) {
        training.push_back(makeSample(0.05 + 0.02 * index, false, 4));
        training.push_back(makeSample(0.72 + 0.012 * index, true, 1 + index % 3));
    }
    std::vector<adaptive_fusion_slam::FailurePredictionSample> testing = {
        makeSample(0.12, false, 4), makeSample(0.25, false, 4),
        makeSample(0.38, false, 4), makeSample(0.76, true, 3),
        makeSample(0.84, true, 2), makeSample(0.94, true, 1),
    };

    adaptive_fusion_slam::TemporalRiskPredictor predictor;
    predictor.train(training);
    const auto probabilities = predictor.predictProbabilities(testing);
    const auto metrics = adaptive_fusion_slam::evaluateRiskPredictions(
        testing, probabilities, predictor.decisionThreshold());

    bool passed = true;
    passed &= check(predictor.isTrained() && predictor.weights().size() == 21,
                    "seven features should produce latest/mean/slope encoding");
    passed &= check(probabilities.front() < 0.2 && probabilities.back() > 0.8,
                    "risk should separate healthy and severe windows");
    passed &= check(metrics.accuracy > 0.99 && metrics.auroc > 0.99 &&
                        metrics.auprc > 0.99 && metrics.brier_score < 0.05,
                    "held-out synthetic metrics should be strong");
    passed &= check(std::abs(metrics.mean_warning_lead_frames - 2.0) < 1e-12,
                    "warning lead time should average true positive offsets");

    if (!passed) {
        return 1;
    }
    std::cout << "Temporal risk test passed: accuracy " << metrics.accuracy
              << ", AUROC " << metrics.auroc << ", AUPRC " << metrics.auprc
              << ", Brier " << metrics.brier_score
              << ", mean lead " << metrics.mean_warning_lead_frames
              << " frames." << std::endl;
    return 0;
}
