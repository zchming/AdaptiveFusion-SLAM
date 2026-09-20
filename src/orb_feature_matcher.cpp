#include "orb_feature_matcher.h"

#include <stdexcept>
#include <vector>

#include <opencv2/features2d.hpp>

namespace adaptive_fusion_slam {

OrbFeatureMatcher::OrbFeatureMatcher(OrbMatcherConfig config)
    : config_(config) {
    if (config_.ratio_threshold <= 0.0F || config_.ratio_threshold >= 1.0F) {
        throw std::invalid_argument(
            "ORB match ratio threshold must be between zero and one.");
    }
    if (config_.max_hamming_distance < 0 ||
        config_.max_hamming_distance > 256) {
        throw std::invalid_argument(
            "ORB maximum Hamming distance must be between zero and 256.");
    }
}

std::vector<FeatureMatch> OrbFeatureMatcher::match(
    const cv::Mat& query_descriptors,
    const cv::Mat& train_descriptors) const {
    if (query_descriptors.empty() || train_descriptors.empty()) {
        return {};
    }
    if (query_descriptors.type() != CV_8UC1 ||
        train_descriptors.type() != CV_8UC1) {
        throw std::invalid_argument(
            "ORB descriptor matrices must contain unsigned 8-bit values.");
    }
    if (query_descriptors.cols != 32 || train_descriptors.cols != 32) {
        throw std::invalid_argument(
            "ORB descriptor rows must contain exactly 32 bytes.");
    }

    const cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> forward_neighbors;
    matcher.knnMatch(
        query_descriptors,
        train_descriptors,
        forward_neighbors,
        2);

    std::vector<int> reverse_query_to_train(
        static_cast<std::size_t>(train_descriptors.rows),
        -1);
    if (config_.require_mutual_consistency) {
        std::vector<cv::DMatch> reverse_best_matches;
        matcher.match(
            train_descriptors,
            query_descriptors,
            reverse_best_matches);
        for (const auto& reverse_match : reverse_best_matches) {
            reverse_query_to_train[
                static_cast<std::size_t>(reverse_match.queryIdx)] =
                reverse_match.trainIdx;
        }
    }

    std::vector<FeatureMatch> accepted_matches;
    accepted_matches.reserve(forward_neighbors.size());

    for (const auto& neighbors : forward_neighbors) {
        if (neighbors.size() < 2) {
            continue;
        }

        const cv::DMatch& best = neighbors[0];
        const cv::DMatch& second_best = neighbors[1];
        if (best.distance > config_.max_hamming_distance) {
            continue;
        }
        if (best.distance >=
            config_.ratio_threshold * second_best.distance) {
            continue;
        }

        if (config_.require_mutual_consistency) {
            const auto train_index = static_cast<std::size_t>(best.trainIdx);
            if (train_index >= reverse_query_to_train.size() ||
                reverse_query_to_train[train_index] != best.queryIdx) {
                continue;
            }
        }

        accepted_matches.push_back({
            static_cast<std::size_t>(best.queryIdx),
            static_cast<std::size_t>(best.trainIdx),
            best.distance,
        });
    }

    return accepted_matches;
}

const OrbMatcherConfig& OrbFeatureMatcher::config() const {
    return config_;
}

}  // namespace adaptive_fusion_slam
