#include "trajectory.h"

#include <iomanip>
#include <ostream>
#include <stdexcept>

namespace adaptive_fusion_slam {

bool Trajectory::addFrame(const Frame& frame) {
    if (!frame.pose_valid) {
        return false;
    }
    if (!poses_.empty() &&
        frame.association.rgb_timestamp <= poses_.back().timestamp) {
        throw std::invalid_argument(
            "Trajectory timestamps must be strictly increasing.");
    }
    if (!frame.pose_world_from_camera.matrix().allFinite()) {
        throw std::invalid_argument("Trajectory pose must be finite.");
    }

    poses_.push_back({
        frame.association.rgb_timestamp,
        frame.pose_world_from_camera,
    });
    return true;
}

const std::vector<TrajectoryPose>& Trajectory::poses() const {
    return poses_;
}

void Trajectory::writeTum(std::ostream& output) const {
    output << std::fixed << std::setprecision(9);
    for (const auto& pose : poses_) {
        Eigen::Quaterniond quaternion(pose.pose_world_from_camera.linear());
        quaternion.normalize();
        const Eigen::Vector3d translation =
            pose.pose_world_from_camera.translation();
        output << pose.timestamp << ' '
               << translation.x() << ' '
               << translation.y() << ' '
               << translation.z() << ' '
               << quaternion.x() << ' '
               << quaternion.y() << ' '
               << quaternion.z() << ' '
               << quaternion.w() << '\n';
    }
}

}  // namespace adaptive_fusion_slam
