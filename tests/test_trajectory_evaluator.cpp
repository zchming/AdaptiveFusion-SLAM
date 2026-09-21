#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include <Eigen/Geometry>

#include "trajectory_evaluator.h"

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
    std::vector<adaptive_fusion_slam::TrajectoryPose> truth;
    std::vector<adaptive_fusion_slam::TrajectoryPose> estimate;
    Eigen::Isometry3d truth_from_estimate = Eigen::Isometry3d::Identity();
    truth_from_estimate.linear() =
        Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    truth_from_estimate.translation() = Eigen::Vector3d(1.0, -0.5, 0.2);

    for (std::size_t index = 0; index < 10; ++index) {
        Eigen::Isometry3d truth_pose = Eigen::Isometry3d::Identity();
        const double value = static_cast<double>(index);
        truth_pose.translation() = Eigen::Vector3d(
            0.12 * value, 0.008 * value * value, 0.03 * std::sin(value));
        truth_pose.linear() = Eigen::AngleAxisd(
            0.02 * value, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        truth.push_back({1.0 + 0.1 * value, truth_pose});
        estimate.push_back({
            1.005 + 0.1 * value,
            truth_from_estimate.inverse() * truth_pose,
        });
    }

    const adaptive_fusion_slam::TrajectoryEvaluator evaluator;
    const auto exact_result = evaluator.evaluate(truth, estimate);
    bool passed = true;
    passed &= check(exact_result.matched_pose_count == 10 &&
                        std::abs(exact_result.estimated_pose_match_ratio - 1.0) <
                            1e-12,
                    "all offset timestamps should associate one-to-one");
    passed &= check(exact_result.ate_translation_rmse_meters < 1e-12 &&
                        exact_result.rpe_translation_rmse_meters < 1e-12 &&
                        exact_result.rpe_rotation_rmse_radians < 1e-12,
                    "rigid alignment should remove only global frame offset");

    auto drifting_estimate = estimate;
    for (std::size_t index = 0; index < drifting_estimate.size(); ++index) {
        const double value = static_cast<double>(index);
        drifting_estimate[index].pose_world_from_camera.translation().z() +=
            0.002 * value * value;
        drifting_estimate[index].pose_world_from_camera.linear() *=
            Eigen::AngleAxisd(
                0.001 * value * value,
                Eigen::Vector3d::UnitY()).toRotationMatrix();
    }
    const auto drift_result = evaluator.evaluate(truth, drifting_estimate);
    passed &= check(drift_result.ate_translation_rmse_meters > 0.005 &&
                        drift_result.rpe_translation_rmse_meters > 0.001 &&
                        drift_result.rpe_rotation_rmse_radians > 0.001,
                    "ATE and RPE should expose non-rigid drift");

    std::istringstream tum_input(
        "# comment\n"
        "1.0 0 0 0 0 0 0 1\n"
        "1.1 1 2 3 0 0 0 2\n");
    const auto parsed = adaptive_fusion_slam::readTumTrajectory(tum_input);
    passed &= check(parsed.size() == 2 &&
                        parsed[1].pose_world_from_camera.rotation().isApprox(
                            Eigen::Matrix3d::Identity(), 1e-12),
                    "TUM parser should skip comments and normalize quaternions");

    if (!passed) {
        return 1;
    }
    std::cout << "Trajectory evaluator test passed: aligned ATE "
              << exact_result.ate_translation_rmse_meters
              << " m; drifted ATE "
              << drift_result.ate_translation_rmse_meters
              << " m, RPE translation "
              << drift_result.rpe_translation_rmse_meters
              << " m, RPE rotation "
              << drift_result.rpe_rotation_rmse_radians << " rad."
              << std::endl;
    return 0;
}
