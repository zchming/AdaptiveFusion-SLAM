#include "failure_prediction_dataset.h"

#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace adaptive_fusion_slam {

FailurePredictionDatasetBuilder::FailurePredictionDatasetBuilder(
    FailurePredictionDatasetConfig config)
    : config_(config) {
    if (config_.history_length == 0 || config_.prediction_horizon == 0) {
        throw std::invalid_argument(
            "History length and prediction horizon must be positive.");
    }
}

std::vector<FailurePredictionSample> FailurePredictionDatasetBuilder::build(
    const std::vector<GeometricHealth>& sequence) const {
    std::vector<FailurePredictionSample> samples;
    if (sequence.size() <
        config_.history_length + config_.prediction_horizon) {
        return samples;
    }

    const std::size_t first_anchor = config_.history_length - 1;
    const std::size_t last_anchor =
        sequence.size() - config_.prediction_horizon - 1;
    for (std::size_t anchor = first_anchor; anchor <= last_anchor; ++anchor) {
        bool history_is_usable = true;
        const std::size_t history_begin =
            anchor + 1 - config_.history_length;
        for (std::size_t index = history_begin; index <= anchor; ++index) {
            if (!sequence[index].has_tracking_measurement ||
                (config_.require_successful_history &&
                 !sequence[index].tracking_success)) {
                history_is_usable = false;
                break;
            }
        }
        if (!history_is_usable) {
            continue;
        }

        FailurePredictionSample sample;
        sample.anchor_frame_id = sequence[anchor].frame_id;
        sample.anchor_timestamp = sequence[anchor].timestamp;
        sample.frames_until_failure = config_.prediction_horizon + 1;
        sample.history.reserve(config_.history_length);
        for (std::size_t index = history_begin; index <= anchor; ++index) {
            sample.history.push_back(makeHealthFeatureVector(sequence[index]));
        }
        for (std::size_t offset = 1;
             offset <= config_.prediction_horizon;
             ++offset) {
            if (!sequence[anchor + offset].tracking_success) {
                sample.future_failure = true;
                sample.frames_until_failure = offset;
                break;
            }
        }
        samples.push_back(std::move(sample));
    }
    return samples;
}

const FailurePredictionDatasetConfig&
FailurePredictionDatasetBuilder::config() const {
    return config_;
}

HealthFeatureVector makeHealthFeatureVector(const GeometricHealth& health) {
    return {
        health.inlier_ratio,
        health.mean_reprojection_error_pixels,
        health.mean_forward_backward_error_pixels,
        health.spatial_coverage,
        health.median_parallax_pixels,
        health.valid_depth_ratio,
        health.used_orb_fallback ? 1.0 : 0.0,
    };
}

void writeFailurePredictionCsv(
    std::ostream& output,
    const std::vector<FailurePredictionSample>& samples,
    std::size_t history_length) {
    output << "anchor_frame_id,anchor_timestamp,future_failure,"
              "frames_until_failure";
    {
        for (std::size_t step = 0; step < history_length; ++step) {
            for (const char* name : {
                     "inlier_ratio", "reprojection_error", "fb_error",
                     "spatial_coverage", "parallax", "valid_depth_ratio",
                     "orb_fallback"}) {
                output << ",h" << step << '_' << name;
            }
        }
    }
    output << '\n';

    output << std::setprecision(17);
    for (const auto& sample : samples) {
        output << sample.anchor_frame_id << ',' << sample.anchor_timestamp << ','
               << static_cast<int>(sample.future_failure) << ','
               << sample.frames_until_failure;
        for (const auto& feature_vector : sample.history) {
            for (double value : feature_vector) {
                output << ',' << value;
            }
        }
        output << '\n';
    }
}

}  // namespace adaptive_fusion_slam
