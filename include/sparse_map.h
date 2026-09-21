#pragma once

#include <cstddef>
#include <vector>

#include "camera.h"
#include "frame.h"
#include "keyframe.h"
#include "map_point.h"
#include "rgbd_correspondence_builder.h"

namespace adaptive_fusion_slam {

struct KeyframeInsertionResult {
    std::size_t keyframe_id;
    std::size_t map_points_created;
};

class SparseMap {
public:
    SparseMap(Camera camera, DepthConversionConfig depth_config = {});

    KeyframeInsertionResult insertKeyframe(const Frame& frame);
    const std::vector<Keyframe>& keyframes() const;
    const std::vector<MapPoint>& mapPoints() const;
    const Keyframe* lastKeyframe() const;

private:
    RgbdCorrespondenceBuilder correspondence_builder_;
    std::vector<Keyframe> keyframes_;
    std::vector<MapPoint> map_points_;
};

}  // namespace adaptive_fusion_slam
