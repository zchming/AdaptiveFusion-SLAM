#include <cmath>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "orb_feature_extractor.h"
#include "orb_feature_matcher.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << std::endl;
        return false;
    }
    return true;
}

cv::Mat makeDeterministicTexture() {
    cv::Mat image(480, 640, CV_8UC1);
    cv::RNG random_generator(12345);
    random_generator.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.8);
    return image;
}

}  // namespace

int main() {
    constexpr float expected_horizontal_shift = 6.0F;
    constexpr float expected_vertical_shift = 4.0F;

    const cv::Mat first_image = makeDeterministicTexture();
    cv::Mat second_image;
    const cv::Mat translation =
        (cv::Mat_<double>(2, 3) <<
            1.0, 0.0, expected_horizontal_shift,
            0.0, 1.0, expected_vertical_shift);
    cv::warpAffine(
        first_image,
        second_image,
        translation,
        first_image.size(),
        cv::INTER_LINEAR,
        cv::BORDER_REFLECT_101);

    adaptive_fusion_slam::OrbFeatureConfig extractor_config;
    extractor_config.max_features = 800;
    adaptive_fusion_slam::OrbFeatureExtractor extractor(extractor_config);
    const auto first_features = extractor.extract(first_image);
    const auto second_features = extractor.extract(second_image);

    adaptive_fusion_slam::OrbFeatureMatcher matcher;
    const auto matches = matcher.match(
        first_features.descriptors,
        second_features.descriptors);

    std::size_t translation_consistent_matches = 0;
    double distance_sum = 0.0;
    for (const auto& match : matches) {
        const cv::Point2f displacement =
            second_features.keypoints[match.train_index].pt -
            first_features.keypoints[match.query_index].pt;
        const cv::Point2f displacement_error(
            displacement.x - expected_horizontal_shift,
            displacement.y - expected_vertical_shift);
        if (cv::norm(displacement_error) < 2.5) {
            ++translation_consistent_matches;
        }
        distance_sum += match.distance;
    }

    const double consistent_ratio = matches.empty()
        ? 0.0
        : static_cast<double>(translation_consistent_matches) /
              static_cast<double>(matches.size());
    const double mean_distance = matches.empty()
        ? 0.0
        : distance_sum / static_cast<double>(matches.size());

    bool passed = true;
    passed &= check(matches.size() >= 100,
                    "translated texture should produce at least 100 matches");
    passed &= check(consistent_ratio >= 0.90,
                    "at least 90 percent of matches should follow translation");
    passed &= check(
        matcher.match(cv::Mat(), second_features.descriptors).empty(),
        "empty descriptors should produce no matches");

    if (!passed) {
        return 1;
    }

    std::cout << "ORB matcher test passed with " << matches.size()
              << " matches, " << translation_consistent_matches
              << " translation-consistent matches, and mean Hamming distance "
              << mean_distance << "." << std::endl;
    return 0;
}
