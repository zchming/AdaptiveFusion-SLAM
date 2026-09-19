#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "orb_feature_extractor.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << std::endl;
        return false;
    }
    return true;
}

cv::Mat makeCheckerboard() {
    constexpr int image_width = 640;
    constexpr int image_height = 480;
    constexpr int cell_size = 40;

    cv::Mat image(image_height, image_width, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < image_height; y += cell_size) {
        for (int x = 0; x < image_width; x += cell_size) {
            if ((x / cell_size + y / cell_size) % 2 == 0) {
                cv::rectangle(
                    image,
                    cv::Rect(x, y, cell_size, cell_size),
                    cv::Scalar(255),
                    cv::FILLED);
            }
        }
    }
    return image;
}

}  // namespace

int main() {
    adaptive_fusion_slam::OrbFeatureConfig config;
    config.max_features = 500;
    adaptive_fusion_slam::OrbFeatureExtractor extractor(config);

    cv::Mat color_image;
    cv::cvtColor(makeCheckerboard(), color_image, cv::COLOR_GRAY2BGR);
    const auto features = extractor.extract(color_image);

    bool passed = true;
    passed &= check(!features.keypoints.empty(),
                    "checkerboard should produce keypoints");
    passed &= check(features.keypoints.size() <= 500,
                    "keypoint count exceeds configured maximum");
    passed &= check(
        features.descriptors.rows ==
            static_cast<int>(features.keypoints.size()),
        "each keypoint must have one descriptor row");
    passed &= check(features.descriptors.cols == 32,
                    "ORB descriptor must contain 32 bytes");
    passed &= check(features.descriptors.type() == CV_8UC1,
                    "ORB descriptors must be unsigned 8-bit values");

    const cv::Mat blank_image(480, 640, CV_8UC1, cv::Scalar(127));
    const auto blank_features = extractor.extract(blank_image);
    passed &= check(blank_features.keypoints.empty(),
                    "uniform image should not produce keypoints");
    passed &= check(blank_features.descriptors.empty(),
                    "uniform image should not produce descriptors");

    if (!passed) {
        return 1;
    }

    std::cout << "ORB feature test passed with "
              << features.keypoints.size()
              << " checkerboard keypoints and 32-byte descriptors."
              << std::endl;
    return 0;
}
