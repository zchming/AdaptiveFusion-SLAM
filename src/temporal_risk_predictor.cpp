#include "temporal_risk_predictor.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace adaptive_fusion_slam {
namespace {

double sigmoid(double value) {
    if (value >= 0.0) {
        const double exponential = std::exp(-value);
        return 1.0 / (1.0 + exponential);
    }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

}  // namespace

TemporalRiskPredictor::TemporalRiskPredictor(
    TemporalRiskPredictorConfig config)
    : config_(config) {
    if (config_.training_iterations <= 0 || config_.learning_rate <= 0.0 ||
        config_.l2_regularization < 0.0 ||
        config_.decision_threshold <= 0.0 ||
        config_.decision_threshold >= 1.0) {
        throw std::invalid_argument("Temporal risk-predictor config is invalid.");
    }
}

std::vector<double> TemporalRiskPredictor::encode(
    const FailurePredictionSample& sample) const {
    if (sample.history.empty()) {
        throw std::invalid_argument("Risk sample history cannot be empty.");
    }
    const std::size_t length = sample.history.size();
    std::vector<double> encoded;
    encoded.reserve(3 * kHealthFeatureDimension);
    for (std::size_t feature = 0; feature < kHealthFeatureDimension; ++feature) {
        double sum = 0.0;
        for (const auto& step : sample.history) {
            sum += step[feature];
        }
        encoded.push_back(sample.history.back()[feature]);
        encoded.push_back(sum / static_cast<double>(length));

        double slope = 0.0;
        if (length > 1) {
            const double center = 0.5 * static_cast<double>(length - 1);
            double numerator = 0.0;
            double denominator = 0.0;
            for (std::size_t index = 0; index < length; ++index) {
                const double centered_index =
                    static_cast<double>(index) - center;
                numerator += centered_index * sample.history[index][feature];
                denominator += centered_index * centered_index;
            }
            slope = numerator / denominator;
        }
        encoded.push_back(slope);
    }
    return encoded;
}

std::vector<double> TemporalRiskPredictor::normalize(
    const std::vector<double>& features) const {
    if (features.size() != means_.size()) {
        throw std::invalid_argument("Risk feature dimension is inconsistent.");
    }
    std::vector<double> normalized(features.size());
    for (std::size_t index = 0; index < features.size(); ++index) {
        normalized[index] =
            (features[index] - means_[index]) /
            standard_deviations_[index];
    }
    return normalized;
}

void TemporalRiskPredictor::train(
    const std::vector<FailurePredictionSample>& samples) {
    if (samples.empty()) {
        throw std::invalid_argument("Risk predictor needs training samples.");
    }
    std::vector<std::vector<double>> encoded_samples;
    encoded_samples.reserve(samples.size());
    std::size_t positive_count = 0;
    for (const auto& sample : samples) {
        encoded_samples.push_back(encode(sample));
        positive_count += sample.future_failure ? 1U : 0U;
    }
    if (positive_count == 0 || positive_count == samples.size()) {
        throw std::invalid_argument(
            "Risk predictor training requires both classes.");
    }

    const std::size_t dimension = encoded_samples.front().size();
    means_.assign(dimension, 0.0);
    standard_deviations_.assign(dimension, 0.0);
    for (const auto& features : encoded_samples) {
        if (features.size() != dimension) {
            throw std::invalid_argument(
                "All temporal samples must have the same history length.");
        }
        for (std::size_t index = 0; index < dimension; ++index) {
            means_[index] += features[index];
        }
    }
    for (double& mean : means_) {
        mean /= static_cast<double>(samples.size());
    }
    for (const auto& features : encoded_samples) {
        for (std::size_t index = 0; index < dimension; ++index) {
            const double difference = features[index] - means_[index];
            standard_deviations_[index] += difference * difference;
        }
    }
    for (double& deviation : standard_deviations_) {
        deviation = std::sqrt(deviation / static_cast<double>(samples.size()));
        if (deviation < 1e-9) {
            deviation = 1.0;
        }
    }
    for (auto& features : encoded_samples) {
        features = normalize(features);
    }

    weights_.assign(dimension, 0.0);
    bias_ = 0.0;
    const double positive_weight =
        static_cast<double>(samples.size()) /
        (2.0 * static_cast<double>(positive_count));
    const double negative_weight =
        static_cast<double>(samples.size()) /
        (2.0 * static_cast<double>(samples.size() - positive_count));

    for (int iteration = 0; iteration < config_.training_iterations;
         ++iteration) {
        std::vector<double> weight_gradient(dimension, 0.0);
        double bias_gradient = 0.0;
        for (std::size_t sample_index = 0;
             sample_index < samples.size();
             ++sample_index) {
            const double label =
                samples[sample_index].future_failure ? 1.0 : 0.0;
            const double class_weight =
                label > 0.5 ? positive_weight : negative_weight;
            const double logit = bias_ + std::inner_product(
                weights_.begin(), weights_.end(),
                encoded_samples[sample_index].begin(), 0.0);
            const double error = class_weight * (sigmoid(logit) - label);
            bias_gradient += error;
            for (std::size_t index = 0; index < dimension; ++index) {
                weight_gradient[index] +=
                    error * encoded_samples[sample_index][index];
            }
        }
        const double inverse_count = 1.0 / static_cast<double>(samples.size());
        bias_ -= config_.learning_rate * bias_gradient * inverse_count;
        for (std::size_t index = 0; index < dimension; ++index) {
            const double gradient = weight_gradient[index] * inverse_count +
                config_.l2_regularization * weights_[index];
            weights_[index] -= config_.learning_rate * gradient;
        }
    }
    trained_ = true;
}

double TemporalRiskPredictor::predictProbability(
    const FailurePredictionSample& sample) const {
    if (!trained_) {
        throw std::logic_error("Risk predictor must be trained before use.");
    }
    const auto features = normalize(encode(sample));
    return sigmoid(bias_ + std::inner_product(
        weights_.begin(), weights_.end(), features.begin(), 0.0));
}

std::vector<double> TemporalRiskPredictor::predictProbabilities(
    const std::vector<FailurePredictionSample>& samples) const {
    std::vector<double> probabilities;
    probabilities.reserve(samples.size());
    for (const auto& sample : samples) {
        probabilities.push_back(predictProbability(sample));
    }
    return probabilities;
}

bool TemporalRiskPredictor::isTrained() const {
    return trained_;
}

const std::vector<double>& TemporalRiskPredictor::weights() const {
    return weights_;
}

double TemporalRiskPredictor::decisionThreshold() const {
    return config_.decision_threshold;
}

RiskPredictionMetrics evaluateRiskPredictions(
    const std::vector<FailurePredictionSample>& samples,
    const std::vector<double>& probabilities,
    double decision_threshold) {
    if (samples.empty() || samples.size() != probabilities.size() ||
        decision_threshold <= 0.0 || decision_threshold >= 1.0) {
        throw std::invalid_argument("Risk-evaluation inputs are invalid.");
    }

    RiskPredictionMetrics metrics;
    metrics.sample_count = samples.size();
    std::size_t correct = 0;
    double squared_error = 0.0;
    double lead_sum = 0.0;
    std::size_t warned_positive_count = 0;
    std::vector<std::pair<double, bool>> ranked;
    ranked.reserve(samples.size());
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const bool label = samples[index].future_failure;
        const bool prediction = probabilities[index] >= decision_threshold;
        metrics.positive_count += label ? 1U : 0U;
        correct += prediction == label ? 1U : 0U;
        const double difference = probabilities[index] - (label ? 1.0 : 0.0);
        squared_error += difference * difference;
        if (prediction && label) {
            lead_sum += static_cast<double>(samples[index].frames_until_failure);
            ++warned_positive_count;
        }
        ranked.emplace_back(probabilities[index], label);
    }
    const std::size_t negative_count = samples.size() - metrics.positive_count;
    if (metrics.positive_count == 0 || negative_count == 0) {
        throw std::invalid_argument("Risk evaluation requires both classes.");
    }
    metrics.accuracy = static_cast<double>(correct) / samples.size();
    metrics.brier_score = squared_error / samples.size();
    metrics.mean_warning_lead_frames = warned_positive_count == 0
        ? 0.0
        : lead_sum / static_cast<double>(warned_positive_count);

    std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
        return left.first > right.first;
    });
    double true_positives = 0.0;
    double false_positives = 0.0;
    double previous_tpr = 0.0;
    double previous_fpr = 0.0;
    double precision_sum = 0.0;
    for (const auto& item : ranked) {
        if (item.second) {
            true_positives += 1.0;
            precision_sum += true_positives / (true_positives + false_positives);
        } else {
            false_positives += 1.0;
        }
        const double tpr = true_positives / metrics.positive_count;
        const double fpr = false_positives / negative_count;
        metrics.auroc +=
            (fpr - previous_fpr) * (tpr + previous_tpr) * 0.5;
        previous_tpr = tpr;
        previous_fpr = fpr;
    }
    metrics.auprc = precision_sum / metrics.positive_count;
    return metrics;
}

}  // namespace adaptive_fusion_slam
