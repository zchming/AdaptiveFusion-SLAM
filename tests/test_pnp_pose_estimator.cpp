#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "camera.h"
#include "pnp_pose_estimator.h"

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
    constexpr double fx = 517.3;
    constexpr double fy = 516.5;
    constexpr double cx = 318.6;
    constexpr double cy = 255.3;
    const adaptive_fusion_slam::Camera camera(fx, fy, cx, cy);

    const Eigen::Matrix3d expected_rotation =
        (Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(-0.04, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    const Eigen::Vector3d expected_translation(0.10, -0.03, 0.05);

    std::vector<adaptive_fusion_slam::RgbdCorrespondence> correspondences;
    for (int row = 0; row < 6; ++row) {
        for (int column = 0; column < 10; ++column) {
            const Eigen::Vector3d point_previous(
                -0.9 + 0.2 * static_cast<double>(column),
                -0.5 + 0.2 * static_cast<double>(row),
                2.5 + 0.15 * static_cast<double>((row + column) % 5));
            const Eigen::Vector3d point_current =
                expected_rotation * point_previous + expected_translation;
            Eigen::Vector2d current_pixel = camera.cameraToPixel(point_current);

            const std::size_t index = correspondences.size();
            if (index >= 50) {
                current_pixel += Eigen::Vector2d(120.0, -80.0);
            }

            correspondences.push_back({
                index,
                point_previous,
                current_pixel,
                point_previous.z(),
            });
        }
    }

    cv::setRNGSeed(12345);
    const adaptive_fusion_slam::PnpPoseEstimator estimator(camera);
    const auto estimate = estimator.estimate(correspondences);

    const Eigen::Matrix3d rotation_error_matrix =
        estimate.rotation_current_from_previous * expected_rotation.transpose();
    const double rotation_error =
        Eigen::AngleAxisd(rotation_error_matrix).angle();
    const double translation_error =
        (estimate.translation_current_from_previous -
         expected_translation).norm();

    bool passed = true;
    passed &= check(estimate.success, "PnP estimation should succeed");
    passed &= check(estimate.inlier_indices.size() >= 49,
                    "RANSAC should retain at least 49 of 50 true inliers");
    passed &= check(rotation_error < 1e-5,
                    "rotation error should be below 1e-5 radians");
    passed &= check(translation_error < 1e-5,
                    "translation error should be below 1e-5 meters");
    passed &= check(estimate.mean_reprojection_error_pixels < 1e-4,
                    "mean reprojection error should be below 1e-4 pixels");
    passed &= check(!estimator.estimate({}).success,
                    "insufficient correspondences should fail cleanly");
    auto invalid_correspondences = correspondences;
    invalid_correspondences[0].point_previous_camera.x() =
        std::numeric_limits<double>::quiet_NaN();
    passed &= check(!estimator.estimate(invalid_correspondences).success,
                    "non-finite geometry should fail cleanly");

    if (!passed) {
        return 1;
    }

    std::cout << "PnP pose test passed with "
              << estimate.inlier_indices.size() << "/"
              << correspondences.size() << " inliers, rotation error "
              << rotation_error << " radians, translation error "
              << translation_error << " meters, and mean reprojection error "
              << estimate.mean_reprojection_error_pixels << " pixels."
              << std::endl;
    return 0;
}
