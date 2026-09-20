#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "lk_optical_flow_tracker.h"
#include "orb_feature_extractor.h"

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
    cv::RNG random_generator(24680);
    random_generator.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(5, 5), 1.0);
    return image;
}

}  // namespace

int main() {
    constexpr float expected_horizontal_shift = 5.0F;
    constexpr float expected_vertical_shift = 3.0F;

    const cv::Mat previous_image = makeDeterministicTexture();
    cv::Mat current_image;
    const cv::Mat translation =
        (cv::Mat_<double>(2, 3) <<
            1.0, 0.0, expected_horizontal_shift,
            0.0, 1.0, expected_vertical_shift);
    cv::warpAffine(
        previous_image,
        current_image,
        translation,
        previous_image.size(),
        cv::INTER_LINEAR,
        cv::BORDER_REFLECT_101);

    adaptive_fusion_slam::OrbFeatureConfig extractor_config;
    extractor_config.max_features = 800;
    adaptive_fusion_slam::OrbFeatureExtractor extractor(extractor_config);
    const auto previous_features = extractor.extract(previous_image);

    adaptive_fusion_slam::LkOpticalFlowTracker tracker;
    const auto tracks = tracker.track(
        previous_image,
        current_image,
        previous_features.keypoints);

    std::size_t translation_consistent_tracks = 0;
    double forward_backward_error_sum = 0.0;
    for (const auto& track : tracks) {
        const cv::Point2f displacement =
            track.current_point - track.previous_point;
        const cv::Point2f displacement_error(
            displacement.x - expected_horizontal_shift,
            displacement.y - expected_vertical_shift);
        if (cv::norm(displacement_error) < 1.0) {
            ++translation_consistent_tracks;
        }
        forward_backward_error_sum += track.forward_backward_error;
    }

    const double consistent_ratio = tracks.empty()
        ? 0.0
        : static_cast<double>(translation_consistent_tracks) /
              static_cast<double>(tracks.size());
    const double mean_forward_backward_error = tracks.empty()
        ? 0.0
        : forward_backward_error_sum / static_cast<double>(tracks.size());

    bool passed = true;
    passed &= check(tracks.size() >= 500,
                    "translated texture should retain at least 500 tracks");
    passed &= check(consistent_ratio >= 0.98,
                    "at least 98 percent of tracks should follow translation");
    passed &= check(mean_forward_backward_error < 0.1,
                    "mean forward-backward error should be below 0.1 pixels");
    passed &= check(
        tracker.track(previous_image, current_image, {}).empty(),
        "empty keypoint input should produce no tracks");

    if (!passed) {
        return 1;
    }

    std::cout << "LK optical-flow test passed with " << tracks.size()
              << " tracks, " << translation_consistent_tracks
              << " translation-consistent tracks, and mean forward-backward "
                 "error "
              << mean_forward_backward_error << " pixels." << std::endl;
    return 0;
}
