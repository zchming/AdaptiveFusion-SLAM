#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "camera.h"
#include "frame.h"
#include "keyframe.h"
#include "local_bundle_adjuster.h"
#include "map_point.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << std::endl;
        return false;
    }
    return true;
}

adaptive_fusion_slam::Frame makeFrame(
    std::size_t id,
    const Eigen::Isometry3d& initial_pose,
    const Eigen::Isometry3d& true_pose,
    const std::vector<Eigen::Vector3d>& true_points,
    const adaptive_fusion_slam::Camera& camera) {
    adaptive_fusion_slam::Frame frame;
    frame.id = id;
    frame.association.rgb_timestamp = static_cast<double>(id);
    frame.pose_world_from_camera = initial_pose;
    frame.pose_valid = true;
    frame.rgb_image = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0));
    frame.depth_image = cv::Mat(480, 640, CV_16UC1, cv::Scalar(0));
    frame.features.descriptors = cv::Mat(
        static_cast<int>(true_points.size()), 32, CV_8UC1);

    const Eigen::Isometry3d true_camera_from_world = true_pose.inverse();
    for (std::size_t index = 0; index < true_points.size(); ++index) {
        const Eigen::Vector2d pixel = camera.cameraToPixel(
            true_camera_from_world * true_points[index]);
        frame.features.keypoints.emplace_back(
            static_cast<float>(pixel.x()),
            static_cast<float>(pixel.y()),
            31.0F);
        const Eigen::Vector3d point_camera =
            true_camera_from_world * true_points[index];
        frame.depth_image.at<std::uint16_t>(
            static_cast<int>(std::lround(pixel.y())),
            static_cast<int>(std::lround(pixel.x()))) =
            static_cast<std::uint16_t>(std::lround(point_camera.z() * 5000.0));
        frame.features.descriptors.row(static_cast<int>(index)).setTo(
            cv::Scalar(static_cast<int>(index + 1)));
    }
    return frame;
}

}  // namespace

int main() {
    const adaptive_fusion_slam::Camera camera(
        500.0, 500.0, 320.0, 240.0);
    const std::vector<Eigen::Vector3d> true_points = {
        {-0.8, -0.4, 4.0}, {-0.3, 0.5, 4.5}, {0.2, -0.6, 5.0},
        {0.7, 0.4, 5.5}, {-0.5, 0.1, 6.0}, {0.6, -0.2, 6.5},
    };

    const Eigen::Isometry3d first_pose = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d true_second_pose = Eigen::Isometry3d::Identity();
    true_second_pose.translation() = Eigen::Vector3d(0.30, 0.02, 0.0);
    Eigen::Isometry3d noisy_second_pose = true_second_pose;
    noisy_second_pose.translation() += Eigen::Vector3d(0.08, -0.04, 0.03);
    noisy_second_pose.linear() =
        Eigen::AngleAxisd(0.025, Eigen::Vector3d::UnitY()).toRotationMatrix();

    std::vector<adaptive_fusion_slam::Keyframe> keyframes;
    keyframes.emplace_back(
        0, makeFrame(0, first_pose, first_pose, true_points, camera));
    keyframes.emplace_back(
        1,
        makeFrame(
            1, noisy_second_pose, true_second_pose, true_points, camera));

    std::vector<adaptive_fusion_slam::MapPoint> map_points;
    for (std::size_t index = 0; index < true_points.size(); ++index) {
        const Eigen::Vector3d perturbation(
            0.03 * static_cast<double>((index % 3) + 1),
            -0.02 * static_cast<double>((index % 2) + 1),
            0.04 * static_cast<double>((index % 4) + 1));
        map_points.emplace_back(
            index,
            true_points[index] + perturbation,
            keyframes[0].descriptors().row(static_cast<int>(index)),
            adaptive_fusion_slam::MapObservation{0, index});
        map_points.back().addObservation({1, index});
        keyframes[0].associateMapPoint(index, index);
        keyframes[1].associateMapPoint(index, index);
    }

    adaptive_fusion_slam::LocalBundleAdjustmentConfig config;
    config.maximum_iterations = 50;
    adaptive_fusion_slam::LocalBundleAdjuster optimizer(camera, config);
    const auto result = optimizer.optimize(keyframes, map_points);

    bool passed = true;
    passed &= check(result.optimized, "optimization should produce a solution");
    passed &= check(result.keyframes_optimized == 2,
                    "both local keyframes should participate");
    passed &= check(result.map_points_optimized == true_points.size(),
                    "all shared map points should participate");
    passed &= check(result.observations_used == 2 * true_points.size(),
                    "both observations of every point should be used");
    passed &= check(result.depth_observations_used == 2 * true_points.size(),
                    "all synthetic RGB-D observations should constrain scale");
    passed &= check(result.final_reprojection_rmse <
                        0.05 * result.initial_reprojection_rmse,
                    "bundle adjustment should strongly reduce reprojection error");
    passed &= check(
        keyframes[0].poseWorldFromCamera().matrix().isApprox(
            first_pose.matrix(), 1e-12),
        "the anchor keyframe must remain fixed");
    const double translation_error =
        (keyframes[1].poseWorldFromCamera().translation() -
         true_second_pose.translation())
            .norm();
    const Eigen::Matrix3d rotation_error =
        true_second_pose.rotation().transpose() *
        keyframes[1].poseWorldFromCamera().rotation();
    const double rotation_error_radians =
        Eigen::AngleAxisd(rotation_error).angle();
    passed &= check(translation_error < 1e-4,
                    "RGB-D residuals should recover metric translation");
    passed &= check(rotation_error_radians < 1e-5,
                    "bundle adjustment should recover camera rotation");

    if (!passed) {
        return 1;
    }

    std::cout << "Local BA reduced reprojection RMSE from "
              << result.initial_reprojection_rmse << " to "
              << result.final_reprojection_rmse << " pixels using "
              << result.observations_used << " observations; translation error "
              << translation_error << " m." << std::endl;
    return 0;
}
