#include "pnp_pose_estimator.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace adaptive_fusion_slam {

PnpPoseEstimator::PnpPoseEstimator(Camera camera, PnpRansacConfig config)
    : camera_(std::move(camera)), config_(config) {
    if (config_.iterations <= 0) {
        throw std::invalid_argument("PnP RANSAC iterations must be positive.");
    }
    if (config_.reprojection_threshold_pixels <= 0.0) {
        throw std::invalid_argument(
            "PnP reprojection threshold must be positive.");
    }
    if (config_.confidence <= 0.0 || config_.confidence >= 1.0) {
        throw std::invalid_argument(
            "PnP RANSAC confidence must be between zero and one.");
    }
    if (config_.min_correspondences < 4 || config_.min_inliers < 4) {
        throw std::invalid_argument(
            "PnP correspondence and inlier limits must be at least four.");
    }
}

PoseEstimate PnpPoseEstimator::estimate(
    const std::vector<RgbdCorrespondence>& correspondences) const {
    PoseEstimate estimate;
    if (correspondences.size() < config_.min_correspondences ||
        correspondences.size() < config_.min_inliers) {
        return estimate;
    }

    std::vector<cv::Point3d> object_points;
    std::vector<cv::Point2d> image_points;
    object_points.reserve(correspondences.size());
    image_points.reserve(correspondences.size());
    for (const auto& correspondence : correspondences) {
        const auto& point = correspondence.point_previous_camera;
        if (!point.allFinite() || point.z() <= 0.0 ||
            !correspondence.current_pixel.allFinite()) {
            return estimate;
        }
        object_points.emplace_back(point.x(), point.y(), point.z());
        image_points.emplace_back(
            correspondence.current_pixel.x(),
            correspondence.current_pixel.y());
    }

    const cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) <<
            camera_.fx(), 0.0, camera_.cx(),
            0.0, camera_.fy(), camera_.cy(),
            0.0, 0.0, 1.0);
    const cv::Mat distortion = cv::Mat::zeros(4, 1, CV_64F);
    cv::Mat rotation_vector;
    cv::Mat translation_vector;
    cv::Mat inlier_matrix;

    const bool ransac_succeeded = cv::solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        rotation_vector,
        translation_vector,
        false,
        config_.iterations,
        static_cast<float>(config_.reprojection_threshold_pixels),
        config_.confidence,
        inlier_matrix,
        cv::SOLVEPNP_EPNP);

    if (!ransac_succeeded ||
        static_cast<std::size_t>(inlier_matrix.rows) < config_.min_inliers) {
        return estimate;
    }

    std::vector<cv::Point3d> inlier_object_points;
    std::vector<cv::Point2d> inlier_image_points;
    estimate.inlier_indices.reserve(static_cast<std::size_t>(inlier_matrix.rows));
    for (int row = 0; row < inlier_matrix.rows; ++row) {
        const int index = inlier_matrix.at<int>(row, 0);
        estimate.inlier_indices.push_back(static_cast<std::size_t>(index));
        inlier_object_points.push_back(object_points[static_cast<std::size_t>(index)]);
        inlier_image_points.push_back(image_points[static_cast<std::size_t>(index)]);
    }

    const cv::Mat ransac_rotation_vector = rotation_vector.clone();
    const cv::Mat ransac_translation_vector = translation_vector.clone();
    const bool refinement_succeeded = cv::solvePnP(
        inlier_object_points,
        inlier_image_points,
        camera_matrix,
        distortion,
        rotation_vector,
        translation_vector,
        true,
        cv::SOLVEPNP_ITERATIVE);
    if (!refinement_succeeded) {
        rotation_vector = ransac_rotation_vector;
        translation_vector = ransac_translation_vector;
    }

    cv::Mat rotation_matrix;
    cv::Rodrigues(rotation_vector, rotation_matrix);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            estimate.rotation_current_from_previous(row, column) =
                rotation_matrix.at<double>(row, column);
        }
        estimate.translation_current_from_previous(row) =
            translation_vector.at<double>(row, 0);
    }

    double reprojection_error_sum = 0.0;
    for (const std::size_t index : estimate.inlier_indices) {
        const Eigen::Vector3d point_current =
            estimate.rotation_current_from_previous *
                correspondences[index].point_previous_camera +
            estimate.translation_current_from_previous;
        if (point_current.z() <= 0.0) {
            return PoseEstimate{};
        }
        const Eigen::Vector2d projected_pixel =
            camera_.cameraToPixel(point_current);
        reprojection_error_sum +=
            (projected_pixel - correspondences[index].current_pixel).norm();
    }

    estimate.inlier_ratio =
        static_cast<double>(estimate.inlier_indices.size()) /
        static_cast<double>(correspondences.size());
    estimate.mean_reprojection_error_pixels =
        reprojection_error_sum /
        static_cast<double>(estimate.inlier_indices.size());
    estimate.success = true;
    return estimate;
}

const PnpRansacConfig& PnpPoseEstimator::config() const {
    return config_;
}

}  // namespace adaptive_fusion_slam
