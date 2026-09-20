#pragma once

#include <cstddef>

#include <Eigen/Geometry>
#include <opencv2/core/mat.hpp>

#include "orb_feature_extractor.h"
#include "tum_rgbd_dataset.h"

namespace adaptive_fusion_slam {

struct Frame {
    std::size_t id = 0;
    RgbdAssociation association;
    cv::Mat rgb_image;
    cv::Mat depth_image;
    FeatureSet features;
    Eigen::Isometry3d pose_world_from_camera = Eigen::Isometry3d::Identity();
    bool pose_valid = false;
};

}  // namespace adaptive_fusion_slam
