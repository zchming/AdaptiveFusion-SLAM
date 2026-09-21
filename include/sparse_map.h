#pragma once

#include <cstddef>
#include <vector>

#include "camera.h"
#include "frame.h"
#include "keyframe.h"
#include "map_point.h"
#include "orb_feature_matcher.h"
#include "rgbd_correspondence_builder.h"

namespace adaptive_fusion_slam {

struct KeyframeInsertionResult {
    std::size_t keyframe_id;
    std::size_t map_points_created;
    std::size_t existing_map_points_observed;
};

struct MapAssociationConfig {
    double max_reprojection_error_pixels = 3.0;
    double max_depth_difference_meters = 0.15;
    double max_relative_depth_difference = 0.10;
};

class SparseMap {
public:
    SparseMap(
        Camera camera,
        DepthConversionConfig depth_config = {},
        OrbMatcherConfig matcher_config = {},
        MapAssociationConfig association_config = {});

    KeyframeInsertionResult insertKeyframe(const Frame& frame);
    const std::vector<Keyframe>& keyframes() const;
    const std::vector<MapPoint>& mapPoints() const;
    const Keyframe* lastKeyframe() const;

private:
    Camera camera_;
    RgbdCorrespondenceBuilder correspondence_builder_;
    OrbFeatureMatcher feature_matcher_;
    MapAssociationConfig association_config_;
    std::vector<Keyframe> keyframes_;
    std::vector<MapPoint> map_points_;
};

}  // namespace adaptive_fusion_slam
