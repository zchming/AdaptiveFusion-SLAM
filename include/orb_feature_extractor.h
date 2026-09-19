#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>

namespace adaptive_fusion_slam {

struct OrbFeatureConfig {
    int max_features = 1000;
    float scale_factor = 1.2F;
    int pyramid_levels = 8;
    int fast_threshold = 20;
};

struct FeatureSet {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

class OrbFeatureExtractor {
public:
    explicit OrbFeatureExtractor(OrbFeatureConfig config = {});

    FeatureSet extract(const cv::Mat& image);
    const OrbFeatureConfig& config() const;

private:
    OrbFeatureConfig config_;
    cv::Ptr<cv::ORB> orb_;
};

}  // namespace adaptive_fusion_slam
