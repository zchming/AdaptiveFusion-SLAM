#include "sparse_map.h"

#include <utility>
#include <vector>

namespace adaptive_fusion_slam {

SparseMap::SparseMap(Camera camera, DepthConversionConfig depth_config)
    : correspondence_builder_(std::move(camera), depth_config) {}

KeyframeInsertionResult SparseMap::insertKeyframe(const Frame& frame) {
    const std::size_t keyframe_id = keyframes_.size();
    Keyframe keyframe(keyframe_id, frame);

    std::vector<PixelCorrespondence> feature_pixels;
    feature_pixels.reserve(keyframe.keypoints().size());
    for (std::size_t index = 0; index < keyframe.keypoints().size(); ++index) {
        feature_pixels.push_back({
            index,
            keyframe.keypoints()[index].pt,
            keyframe.keypoints()[index].pt,
        });
    }

    const auto rgbd_correspondences = correspondence_builder_.build(
        keyframe.depthImage(), feature_pixels);
    std::vector<MapPoint> new_map_points;
    new_map_points.reserve(rgbd_correspondences.size());
    for (const auto& correspondence : rgbd_correspondences) {
        const std::size_t map_point_id =
            map_points_.size() + new_map_points.size();
        const Eigen::Vector3d position_world =
            keyframe.poseWorldFromCamera() *
            correspondence.point_previous_camera;
        new_map_points.emplace_back(
            map_point_id,
            position_world,
            keyframe.descriptors().row(
                static_cast<int>(correspondence.source_index)),
            MapObservation{keyframe_id, correspondence.source_index});
        keyframe.associateMapPoint(
            correspondence.source_index,
            map_point_id);
    }

    keyframes_.push_back(std::move(keyframe));
    for (auto& map_point : new_map_points) {
        map_points_.push_back(std::move(map_point));
    }

    return {keyframe_id, rgbd_correspondences.size()};
}

const std::vector<Keyframe>& SparseMap::keyframes() const {
    return keyframes_;
}

const std::vector<MapPoint>& SparseMap::mapPoints() const {
    return map_points_;
}

const Keyframe* SparseMap::lastKeyframe() const {
    return keyframes_.empty() ? nullptr : &keyframes_.back();
}

}  // namespace adaptive_fusion_slam
