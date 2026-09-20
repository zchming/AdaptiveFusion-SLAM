#pragma once

#include <cstddef>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>

namespace adaptive_fusion_slam {

struct LkOpticalFlowConfig {
    int window_size = 21;
    int max_pyramid_level = 3;
    int termination_count = 30;
    double termination_epsilon = 0.01;
    double min_eigenvalue_threshold = 1e-4;
    float max_forward_backward_error = 1.0F;
};

struct TrackedFeature {
    std::size_t source_index;
    cv::Point2f previous_point;
    cv::Point2f current_point;
    float forward_backward_error;
};

class LkOpticalFlowTracker {
public:
    explicit LkOpticalFlowTracker(LkOpticalFlowConfig config = {});

    std::vector<TrackedFeature> track(
        const cv::Mat& previous_image,
        const cv::Mat& current_image,
        const std::vector<cv::KeyPoint>& previous_keypoints) const;

    const LkOpticalFlowConfig& config() const;

private:
    LkOpticalFlowConfig config_;
};

}  // namespace adaptive_fusion_slam
