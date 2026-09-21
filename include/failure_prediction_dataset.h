#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>
#include <vector>

#include "geometric_health.h"

namespace adaptive_fusion_slam {

constexpr std::size_t kHealthFeatureDimension = 7;
using HealthFeatureVector = std::array<double, kHealthFeatureDimension>;

struct FailurePredictionDatasetConfig {
    std::size_t history_length = 5;
    std::size_t prediction_horizon = 3;
    bool require_successful_history = true;
};

struct FailurePredictionSample {
    std::size_t anchor_frame_id = 0;
    double anchor_timestamp = 0.0;
    std::vector<HealthFeatureVector> history;
    bool future_failure = false;
    std::size_t frames_until_failure = 0;
};

class FailurePredictionDatasetBuilder {
public:
    explicit FailurePredictionDatasetBuilder(
        FailurePredictionDatasetConfig config = {});

    std::vector<FailurePredictionSample> build(
        const std::vector<GeometricHealth>& sequence) const;
    const FailurePredictionDatasetConfig& config() const;

private:
    FailurePredictionDatasetConfig config_;
};

HealthFeatureVector makeHealthFeatureVector(const GeometricHealth& health);
void writeFailurePredictionCsv(
    std::ostream& output,
    const std::vector<FailurePredictionSample>& samples,
    std::size_t history_length);

}  // namespace adaptive_fusion_slam
