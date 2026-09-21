#include <iostream>

#include <opencv2/core.hpp>

#include "image_degradation.h"

namespace {
bool check(bool condition, const char* message) {
    if (!condition) std::cerr << "Test failed: " << message << std::endl;
    return condition;
}
}  // namespace

int main() {
    adaptive_fusion_slam::RgbdFrame frame;
    frame.rgb_image = cv::Mat(120, 160, CV_8UC3);
    cv::randu(frame.rgb_image, 0, 256);
    frame.depth_image = cv::Mat(120, 160, CV_16UC1, cv::Scalar(5000));
    bool passed = true;
    for (const auto& name : {"blur", "dark", "occlusion", "noise", "drop"}) {
        adaptive_fusion_slam::ImageDegradationConfig config;
        config.type = adaptive_fusion_slam::parseImageDegradationType(name);
        config.drop_interval = 1;
        const adaptive_fusion_slam::ImageDegrader degrader(config);
        const auto degraded = degrader.apply(frame, 0);
        passed &= check(degraded.rgb_image.size() == frame.rgb_image.size(),
                        "degradation must preserve image dimensions");
        passed &= check(cv::norm(degraded.depth_image, frame.depth_image) == 0.0,
                        "visual degradation must preserve depth data");
        passed &= check(cv::norm(degraded.rgb_image, frame.rgb_image) > 0.0,
                        "selected degradation should change RGB pixels");
    }
    if (!passed) return 1;
    std::cout << "Image degradation test passed for blur, dark, occlusion, "
                 "noise, and dropped-frame modes."
              << std::endl;
    return 0;
}
