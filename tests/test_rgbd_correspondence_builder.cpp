#include <cmath>
#include <cstdint>
#include <limits>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

#include "camera.h"
#include "rgbd_correspondence_builder.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << std::endl;
        return false;
    }
    return true;
}

bool near(double actual, double expected, double tolerance = 1e-9) {
    return std::abs(actual - expected) <= tolerance;
}

}  // namespace

int main() {
    constexpr double fx = 517.3;
    constexpr double fy = 516.5;
    constexpr double cx = 318.6;
    constexpr double cy = 255.3;

    cv::Mat depth_image(480, 640, CV_16UC1, cv::Scalar(0));
    depth_image.at<std::uint16_t>(240, 320) = 10000;
    depth_image.at<std::uint16_t>(300, 400) = 5000;
    depth_image.at<std::uint16_t>(200, 200) = 45000;
    depth_image.at<std::uint16_t>(350, 300) = 7500;

    const std::vector<adaptive_fusion_slam::PixelCorrespondence>
        pixel_correspondences = {
            {0, {320.2F, 239.7F}, {325.2F, 242.7F}},
            {1, {400.0F, 300.0F}, {405.0F, 303.0F}},
            {2, {100.0F, 100.0F}, {105.0F, 103.0F}},
            {3, {-1.0F, 50.0F}, {4.0F, 53.0F}},
            {4, {200.0F, 200.0F}, {205.0F, 203.0F}},
            {5,
             {std::numeric_limits<float>::quiet_NaN(), 10.0F},
             {15.0F, 13.0F}},
            {6, {300.0F, 350.0F}, {700.0F, 350.0F}},
        };

    const adaptive_fusion_slam::Camera camera(fx, fy, cx, cy);
    const adaptive_fusion_slam::RgbdCorrespondenceBuilder builder(camera);
    const auto correspondences = builder.build(
        depth_image,
        pixel_correspondences);

    bool passed = true;
    passed &= check(correspondences.size() == 2,
                    "only two correspondences should have valid depth");
    if (!passed) {
        return 1;
    }

    const auto& first = correspondences[0];
    const double expected_x = (320.2 - cx) * 2.0 / fx;
    const double expected_y = (239.7 - cy) * 2.0 / fy;
    passed &= check(first.source_index == 0,
                    "source index should be preserved");
    passed &= check(near(first.depth_meters, 2.0),
                    "raw depth 10000 should convert to 2 meters");
    passed &= check(near(first.point_previous_camera.x(), expected_x, 1e-7),
                    "unexpected camera-frame X coordinate");
    passed &= check(near(first.point_previous_camera.y(), expected_y, 1e-7),
                    "unexpected camera-frame Y coordinate");
    passed &= check(near(first.point_previous_camera.z(), 2.0),
                    "unexpected camera-frame Z coordinate");
    passed &= check(
        near(first.current_pixel.x(), static_cast<double>(325.2F)) &&
            near(first.current_pixel.y(), static_cast<double>(242.7F)),
                    "current-frame pixel should be preserved");

    if (!passed) {
        return 1;
    }

    std::cout << "RGB-D correspondence test passed with "
              << correspondences.size()
              << " valid correspondences from "
              << pixel_correspondences.size() << " candidates. First depth: "
              << first.depth_meters << " meters. First 3D point: "
              << first.point_previous_camera.transpose() << std::endl;
    return 0;
}
