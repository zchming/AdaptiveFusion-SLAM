#pragma once

#include <cstddef>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace adaptive_fusion_slam {

struct OrbMatcherConfig {
    float ratio_threshold = 0.75F;
    int max_hamming_distance = 64;
    bool require_mutual_consistency = true;
};

struct FeatureMatch {
    std::size_t query_index;
    std::size_t train_index;
    float distance;
};

class OrbFeatureMatcher {
public:
    explicit OrbFeatureMatcher(OrbMatcherConfig config = {});

    std::vector<FeatureMatch> match(
        const cv::Mat& query_descriptors,
        const cv::Mat& train_descriptors) const;

    const OrbMatcherConfig& config() const;

private:
    OrbMatcherConfig config_;
};

}  // namespace adaptive_fusion_slam
