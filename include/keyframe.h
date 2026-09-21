#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>

#include "frame.h"

namespace adaptive_fusion_slam {

class Keyframe {
public:
    Keyframe(std::size_t id, const Frame& frame);

    std::size_t id() const;
    std::size_t sourceFrameId() const;
    double timestamp() const;
    const Eigen::Isometry3d& poseWorldFromCamera() const;
    const cv::Mat& rgbImage() const;
    const cv::Mat& depthImage() const;
    const std::vector<cv::KeyPoint>& keypoints() const;
    const cv::Mat& descriptors() const;
    const std::vector<std::optional<std::size_t>>& mapPointIds() const;
    void setPoseWorldFromCamera(const Eigen::Isometry3d& pose);
    void associateMapPoint(std::size_t feature_index, std::size_t map_point_id);

private:
    std::size_t id_;
    std::size_t source_frame_id_;
    double timestamp_;
    Eigen::Isometry3d pose_world_from_camera_;
    cv::Mat rgb_image_;
    cv::Mat depth_image_;
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
    std::vector<std::optional<std::size_t>> map_point_ids_;
};

}  // namespace adaptive_fusion_slam
