#include "lk_optical_flow_tracker.h"

#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace adaptive_fusion_slam {
namespace {

cv::Mat toGrayscale(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("LK optical flow input image is empty.");
    }
    if (image.depth() != CV_8U) {
        throw std::invalid_argument("LK optical flow input must use 8-bit pixels.");
    }

    cv::Mat grayscale;
    if (image.channels() == 1) {
        grayscale = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, grayscale, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::invalid_argument(
            "LK optical flow input must have 1, 3, or 4 channels.");
    }
    return grayscale;
}

bool isInsideImage(const cv::Point2f& point, const cv::Size& image_size) {
    return point.x >= 0.0F && point.y >= 0.0F &&
           point.x < static_cast<float>(image_size.width) &&
           point.y < static_cast<float>(image_size.height);
}

}  // namespace

LkOpticalFlowTracker::LkOpticalFlowTracker(LkOpticalFlowConfig config)
    : config_(config) {
    if (config_.window_size <= 0 || config_.window_size % 2 == 0) {
        throw std::invalid_argument("LK window size must be a positive odd number.");
    }
    if (config_.max_pyramid_level < 0) {
        throw std::invalid_argument("LK maximum pyramid level cannot be negative.");
    }
    if (config_.termination_count <= 0 || config_.termination_epsilon <= 0.0) {
        throw std::invalid_argument("LK termination criteria must be positive.");
    }
    if (config_.min_eigenvalue_threshold < 0.0 ||
        config_.max_forward_backward_error < 0.0F) {
        throw std::invalid_argument("LK quality thresholds cannot be negative.");
    }
}

std::vector<TrackedFeature> LkOpticalFlowTracker::track(
    const cv::Mat& previous_image,
    const cv::Mat& current_image,
    const std::vector<cv::KeyPoint>& previous_keypoints) const {
    const cv::Mat previous_gray = toGrayscale(previous_image);
    const cv::Mat current_gray = toGrayscale(current_image);
    if (previous_gray.size() != current_gray.size()) {
        throw std::invalid_argument(
            "LK optical flow images must have equal dimensions.");
    }
    if (previous_keypoints.empty()) {
        return {};
    }

    std::vector<cv::Point2f> previous_points;
    previous_points.reserve(previous_keypoints.size());
    for (const auto& keypoint : previous_keypoints) {
        previous_points.push_back(keypoint.pt);
    }

    const cv::Size window(config_.window_size, config_.window_size);
    const cv::TermCriteria termination(
        cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
        config_.termination_count,
        config_.termination_epsilon);

    std::vector<cv::Point2f> forward_points;
    std::vector<unsigned char> forward_status;
    std::vector<float> forward_error;
    cv::calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        forward_points,
        forward_status,
        forward_error,
        window,
        config_.max_pyramid_level,
        termination,
        0,
        config_.min_eigenvalue_threshold);

    std::vector<cv::Point2f> valid_forward_points;
    std::vector<std::size_t> valid_source_indices;
    valid_forward_points.reserve(forward_points.size());
    valid_source_indices.reserve(forward_points.size());
    for (std::size_t index = 0; index < forward_points.size(); ++index) {
        if (forward_status[index] != 0 &&
            isInsideImage(forward_points[index], current_gray.size())) {
            valid_forward_points.push_back(forward_points[index]);
            valid_source_indices.push_back(index);
        }
    }
    if (valid_forward_points.empty()) {
        return {};
    }

    std::vector<cv::Point2f> backward_points;
    std::vector<unsigned char> backward_status;
    std::vector<float> backward_error;
    cv::calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        valid_forward_points,
        backward_points,
        backward_status,
        backward_error,
        window,
        config_.max_pyramid_level,
        termination,
        0,
        config_.min_eigenvalue_threshold);

    std::vector<TrackedFeature> tracks;
    tracks.reserve(valid_forward_points.size());
    for (std::size_t index = 0; index < valid_forward_points.size(); ++index) {
        if (backward_status[index] == 0 ||
            !isInsideImage(backward_points[index], previous_gray.size())) {
            continue;
        }

        const std::size_t source_index = valid_source_indices[index];
        const float forward_backward_error = static_cast<float>(cv::norm(
            backward_points[index] - previous_points[source_index]));
        if (forward_backward_error > config_.max_forward_backward_error) {
            continue;
        }

        tracks.push_back({
            source_index,
            previous_points[source_index],
            valid_forward_points[index],
            forward_backward_error,
        });
    }

    return tracks;
}

const LkOpticalFlowConfig& LkOpticalFlowTracker::config() const {
    return config_;
}

}  // namespace adaptive_fusion_slam
