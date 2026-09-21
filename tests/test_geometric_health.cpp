#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include "geometric_health.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << std::endl;
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const std::vector<adaptive_fusion_slam::PixelCorrespondence>
        correspondences = {
            {0, {37.0F, 36.0F}, {40.0F, 40.0F}},
            {1, {197.0F, 36.0F}, {200.0F, 40.0F}},
            {2, {357.0F, 276.0F}, {360.0F, 280.0F}},
            {3, {517.0F, 436.0F}, {520.0F, 440.0F}},
        };
    const std::vector<adaptive_fusion_slam::TrackedFeature> tracks = {
        {0, {37.0F, 36.0F}, {40.0F, 40.0F}, 0.2F},
        {1, {197.0F, 36.0F}, {200.0F, 40.0F}, 0.4F},
    };
    adaptive_fusion_slam::PoseEstimate pose;
    pose.success = true;
    pose.inlier_indices = {0, 1, 2};
    pose.inlier_ratio = 0.75;
    pose.mean_reprojection_error_pixels = 0.8;

    const adaptive_fusion_slam::GeometricHealthMonitor monitor;
    const auto health = monitor.compute(
        cv::Size(640, 480), correspondences, 2, tracks, pose);

    bool passed = true;
    passed &= check(health.has_tracking_measurement && health.tracking_success,
                    "successful pose should produce valid health state");
    passed &= check(health.correspondence_count == 4 &&
                        health.pnp_inlier_count == 3,
                    "correspondence and inlier counts should be retained");
    passed &= check(std::abs(health.inlier_ratio - 0.75) < 1e-12,
                    "inlier ratio should be retained");
    passed &= check(
        std::abs(health.mean_forward_backward_error_pixels - 0.3) < 1e-6,
        "mean forward-backward error should be computed");
    passed &= check(std::abs(health.spatial_coverage - 4.0 / 12.0) < 1e-12,
                    "coverage should count occupied grid cells");
    passed &= check(std::abs(health.median_parallax_pixels - 5.0) < 1e-12,
                    "median parallax should be robustly computed");
    passed &= check(std::abs(health.valid_depth_ratio - 0.5) < 1e-12,
                    "valid-depth ratio should be computed");

    std::ostringstream csv;
    adaptive_fusion_slam::writeGeometricHealthCsvHeader(csv);
    adaptive_fusion_slam::writeGeometricHealthCsvRow(csv, health);
    passed &= check(csv.str().find("median_parallax_pixels") != std::string::npos &&
                        csv.str().find(",0.5\n") != std::string::npos,
                    "health state should be serializable as CSV");

    if (!passed) {
        return 1;
    }
    std::cout << "Geometric health test passed: inlier ratio "
              << health.inlier_ratio << ", reprojection error "
              << health.mean_reprojection_error_pixels
              << " px, forward-backward error "
              << health.mean_forward_backward_error_pixels
              << " px, coverage " << health.spatial_coverage
              << ", parallax " << health.median_parallax_pixels
              << " px, valid-depth ratio " << health.valid_depth_ratio << '.'
              << std::endl;
    return 0;
}
