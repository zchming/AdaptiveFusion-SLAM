#include "keyframe.h"

#include <stdexcept>

namespace adaptive_fusion_slam {

Keyframe::Keyframe(std::size_t id, const Frame& frame)
    : id_(id),
      source_frame_id_(frame.id),
      timestamp_(frame.association.rgb_timestamp),
      pose_world_from_camera_(frame.pose_world_from_camera),
      rgb_image_(frame.rgb_image),
      depth_image_(frame.depth_image),
      keypoints_(frame.features.keypoints),
      descriptors_(frame.features.descriptors.clone()),
      map_point_ids_(frame.features.keypoints.size()) {
    if (!frame.pose_valid) {
        throw std::invalid_argument(
            "Cannot create a keyframe from an invalid frame pose.");
    }
    if (!pose_world_from_camera_.matrix().allFinite()) {
        throw std::invalid_argument("Keyframe pose must be finite.");
    }
    if (descriptors_.rows != static_cast<int>(keypoints_.size()) ||
        (!descriptors_.empty() &&
         (descriptors_.cols != 32 || descriptors_.type() != CV_8UC1))) {
        throw std::invalid_argument(
            "Keyframe ORB keypoints and descriptors are inconsistent.");
    }
}

std::size_t Keyframe::id() const {
    return id_;
}

std::size_t Keyframe::sourceFrameId() const {
    return source_frame_id_;
}

double Keyframe::timestamp() const {
    return timestamp_;
}

const Eigen::Isometry3d& Keyframe::poseWorldFromCamera() const {
    return pose_world_from_camera_;
}

const cv::Mat& Keyframe::rgbImage() const {
    return rgb_image_;
}

const cv::Mat& Keyframe::depthImage() const {
    return depth_image_;
}

const std::vector<cv::KeyPoint>& Keyframe::keypoints() const {
    return keypoints_;
}

const cv::Mat& Keyframe::descriptors() const {
    return descriptors_;
}

const std::vector<std::optional<std::size_t>>& Keyframe::mapPointIds() const {
    return map_point_ids_;
}

void Keyframe::setPoseWorldFromCamera(const Eigen::Isometry3d& pose) {
    if (!pose.matrix().allFinite()) {
        throw std::invalid_argument("Keyframe pose must be finite.");
    }
    pose_world_from_camera_ = pose;
}

void Keyframe::associateMapPoint(
    std::size_t feature_index,
    std::size_t map_point_id) {
    if (feature_index >= map_point_ids_.size()) {
        throw std::out_of_range("Keyframe feature index is out of range.");
    }
    const auto existing = map_point_ids_[feature_index];
    if (existing && *existing != map_point_id) {
        throw std::logic_error(
            "Keyframe feature is already associated with another map point.");
    }
    map_point_ids_[feature_index] = map_point_id;
}

}  // namespace adaptive_fusion_slam
