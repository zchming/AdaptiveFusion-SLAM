#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "camera.h"
#include "rgbd_correspondence_builder.h"

namespace adaptive_fusion_slam {

struct PnpRansacConfig {
    int iterations = 100;
    double reprojection_threshold_pixels = 3.0;
    double confidence = 0.99;
    std::size_t min_correspondences = 6;
    std::size_t min_inliers = 6;
};

struct PoseEstimate {
    bool success = false;
    Eigen::Matrix3d rotation_current_from_previous =
        Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation_current_from_previous =
        Eigen::Vector3d::Zero();
    std::vector<std::size_t> inlier_indices;
    double inlier_ratio = 0.0;
    double mean_reprojection_error_pixels = 0.0;
};

class PnpPoseEstimator {
public:
    PnpPoseEstimator(Camera camera, PnpRansacConfig config = {});

    PoseEstimate estimate(
        const std::vector<RgbdCorrespondence>& correspondences) const;

    const PnpRansacConfig& config() const;

private:
    Camera camera_;
    PnpRansacConfig config_;
};

}  // namespace adaptive_fusion_slam
