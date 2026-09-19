#include "orb_feature_extractor.h"

#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace adaptive_fusion_slam {

OrbFeatureExtractor::OrbFeatureExtractor(OrbFeatureConfig config)
    : config_(config) {
    if (config_.max_features <= 0) {
        throw std::invalid_argument("ORB maximum feature count must be positive.");
    }
    if (config_.scale_factor <= 1.0F) {
        throw std::invalid_argument("ORB scale factor must be greater than one.");
    }
    if (config_.pyramid_levels <= 0) {
        throw std::invalid_argument("ORB pyramid level count must be positive.");
    }
    if (config_.fast_threshold < 0) {
        throw std::invalid_argument("ORB FAST threshold cannot be negative.");
    }

    orb_ = cv::ORB::create(
        config_.max_features,
        config_.scale_factor,
        config_.pyramid_levels,
        31,
        0,
        2,
        cv::ORB::HARRIS_SCORE,
        31,
        config_.fast_threshold);
}

FeatureSet OrbFeatureExtractor::extract(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Cannot extract ORB features from an empty image.");
    }
    if (image.depth() != CV_8U) {
        throw std::invalid_argument("ORB input image must use 8-bit pixels.");
    }

    cv::Mat grayscale;
    if (image.channels() == 1) {
        grayscale = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, grayscale, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::invalid_argument("ORB input image must have 1, 3, or 4 channels.");
    }

    FeatureSet features;
    orb_->detectAndCompute(
        grayscale,
        cv::noArray(),
        features.keypoints,
        features.descriptors);
    return features;
}

const OrbFeatureConfig& OrbFeatureExtractor::config() const {
    return config_;
}

}  // namespace adaptive_fusion_slam
