#pragma once

#include <iosfwd>
#include <vector>

#include <Eigen/Geometry>

#include "frame.h"

namespace adaptive_fusion_slam {

struct TrajectoryPose {
    double timestamp;
    Eigen::Isometry3d pose_world_from_camera;
};

class Trajectory {
public:
    bool addFrame(const Frame& frame);
    const std::vector<TrajectoryPose>& poses() const;
    void writeTum(std::ostream& output) const;

private:
    std::vector<TrajectoryPose> poses_;
};

}  // namespace adaptive_fusion_slam
