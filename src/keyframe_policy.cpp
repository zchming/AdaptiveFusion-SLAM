#include "keyframe_policy.h"

#include <algorithm>
#include <stdexcept>

#include <Eigen/Geometry>

namespace adaptive_fusion_slam {

KeyframePolicy::KeyframePolicy(KeyframePolicyConfig config)
    : config_(config) {
    if (config_.min_frame_gap == 0 ||
        config_.max_frame_gap < config_.min_frame_gap) {
        throw std::invalid_argument("Keyframe frame-gap limits are invalid.");
    }
    if (config_.min_translation_meters < 0.0 ||
        config_.min_rotation_radians < 0.0) {
        throw std::invalid_argument(
            "Keyframe motion thresholds cannot be negative.");
    }
}

bool KeyframePolicy::shouldInsert(
    const Frame& frame,
    const Keyframe* last_keyframe) const {
    if (!frame.pose_valid) {
        return false;
    }
    if (last_keyframe == nullptr) {
        return true;
    }
    if (frame.id <= last_keyframe->sourceFrameId()) {
        return false;
    }

    const std::size_t frame_gap =
        frame.id - last_keyframe->sourceFrameId();
    if (frame_gap < config_.min_frame_gap) {
        return false;
    }
    if (frame_gap >= config_.max_frame_gap) {
        return true;
    }

    const Eigen::Isometry3d relative_world_motion =
        last_keyframe->poseWorldFromCamera().inverse() *
        frame.pose_world_from_camera;
    const double translation = relative_world_motion.translation().norm();
    const double rotation =
        Eigen::AngleAxisd(relative_world_motion.linear()).angle();
    return translation >= config_.min_translation_meters ||
           rotation >= config_.min_rotation_radians;
}

bool KeyframePolicy::shouldInsert(
    const Frame& frame,
    const Keyframe* last_keyframe,
    const RiskAdaptiveDecision& decision) const {
    if (!decision.allow_keyframe_insertion) {
        return false;
    }
    if (!decision.request_early_keyframe) {
        return shouldInsert(frame, last_keyframe);
    }
    if (!frame.pose_valid || last_keyframe == nullptr) {
        return frame.pose_valid && last_keyframe == nullptr;
    }
    if (frame.id <= last_keyframe->sourceFrameId()) {
        return false;
    }
    const std::size_t frame_gap = frame.id - last_keyframe->sourceFrameId();
    const std::size_t early_gap = std::max<std::size_t>(
        1, config_.min_frame_gap / 2);
    if (frame_gap < early_gap) {
        return false;
    }
    if (decision.level == RiskLevel::High ||
        frame_gap >= config_.max_frame_gap) {
        return true;
    }

    const Eigen::Isometry3d relative_world_motion =
        last_keyframe->poseWorldFromCamera().inverse() *
        frame.pose_world_from_camera;
    const double translation = relative_world_motion.translation().norm();
    const double rotation =
        Eigen::AngleAxisd(relative_world_motion.linear()).angle();
    return translation >= 0.5 * config_.min_translation_meters ||
           rotation >= 0.5 * config_.min_rotation_radians;
}

const KeyframePolicyConfig& KeyframePolicy::config() const {
    return config_;
}

}  // namespace adaptive_fusion_slam
