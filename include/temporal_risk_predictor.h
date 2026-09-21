#pragma once

#include <cstddef>
#include <vector>

#include "failure_prediction_dataset.h"

namespace adaptive_fusion_slam {

struct TemporalRiskPredictorConfig {
    int training_iterations = 1500;
    double learning_rate = 0.05;
    double l2_regularization = 1e-3;
    double decision_threshold = 0.5;
};

struct RiskPredictionMetrics {
    std::size_t sample_count = 0;
    std::size_t positive_count = 0;
    double accuracy = 0.0;
    double auroc = 0.0;
    double auprc = 0.0;
    double brier_score = 0.0;
    double mean_warning_lead_frames = 0.0;
};

class TemporalRiskPredictor {
public:
    explicit TemporalRiskPredictor(TemporalRiskPredictorConfig config = {});

    void train(const std::vector<FailurePredictionSample>& samples);
    double predictProbability(const FailurePredictionSample& sample) const;
    std::vector<double> predictProbabilities(
        const std::vector<FailurePredictionSample>& samples) const;
    bool isTrained() const;
    const std::vector<double>& weights() const;
    double decisionThreshold() const;

private:
    std::vector<double> encode(
        const FailurePredictionSample& sample) const;
    std::vector<double> normalize(const std::vector<double>& features) const;

    TemporalRiskPredictorConfig config_;
    bool trained_ = false;
    std::vector<double> means_;
    std::vector<double> standard_deviations_;
    std::vector<double> weights_;
    double bias_ = 0.0;
};

RiskPredictionMetrics evaluateRiskPredictions(
    const std::vector<FailurePredictionSample>& samples,
    const std::vector<double>& probabilities,
    double decision_threshold = 0.5);

}  // namespace adaptive_fusion_slam
