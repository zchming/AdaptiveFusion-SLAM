#include "sparse_map.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace adaptive_fusion_slam {

SparseMap::SparseMap(
    Camera camera,
    DepthConversionConfig depth_config,
    OrbMatcherConfig matcher_config,
    MapAssociationConfig association_config)
    : camera_(std::move(camera)),
      correspondence_builder_(camera_, depth_config),
      feature_matcher_(matcher_config),
      association_config_(association_config) {
    if (association_config_.max_reprojection_error_pixels <= 0.0 ||
        association_config_.max_depth_difference_meters < 0.0 ||
        association_config_.max_relative_depth_difference < 0.0) {
        throw std::invalid_argument(
            "Map-association thresholds must be non-negative and the "
            "reprojection threshold must be positive.");
    }
}

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

    std::vector<std::optional<RgbdCorrespondence>> depth_by_feature(
        keyframe.keypoints().size());
    for (const auto& correspondence : rgbd_correspondences) {
        depth_by_feature[correspondence.source_index] = correspondence;
    }

    struct DeferredObservation {
        std::size_t map_point_id;
        std::size_t feature_index;
    };
    std::vector<DeferredObservation> deferred_observations;

    if (!keyframes_.empty()) {
        const Keyframe& reference = keyframes_.back();
        const auto matches = feature_matcher_.match(
            reference.descriptors(), keyframe.descriptors());
        const Eigen::Isometry3d pose_camera_from_world =
            keyframe.poseWorldFromCamera().inverse();

        for (const auto& match : matches) {
            if (match.query_index >= reference.mapPointIds().size() ||
                match.train_index >= keyframe.keypoints().size()) {
                continue;
            }
            const auto map_point_id =
                reference.mapPointIds()[match.query_index];
            if (!map_point_id || *map_point_id >= map_points_.size()) {
                continue;
            }

            const Eigen::Vector3d predicted_camera_point =
                pose_camera_from_world *
                map_points_[*map_point_id].positionWorld();
            if (predicted_camera_point.z() <= 0.0 ||
                !predicted_camera_point.allFinite()) {
                continue;
            }
            const Eigen::Vector2d predicted_pixel =
                camera_.cameraToPixel(predicted_camera_point);
            const cv::Point2f& measured_pixel =
                keyframe.keypoints()[match.train_index].pt;
            const double reprojection_error =
                (predicted_pixel -
                 Eigen::Vector2d(measured_pixel.x, measured_pixel.y))
                    .norm();
            if (reprojection_error >
                association_config_.max_reprojection_error_pixels) {
                continue;
            }

            const auto& measured_depth = depth_by_feature[match.train_index];
            if (measured_depth) {
                const double allowed_depth_difference = std::max(
                    association_config_.max_depth_difference_meters,
                    association_config_.max_relative_depth_difference *
                        predicted_camera_point.z());
                if (std::abs(measured_depth->depth_meters -
                             predicted_camera_point.z()) >
                    allowed_depth_difference) {
                    continue;
                }
            }

            keyframe.associateMapPoint(match.train_index, *map_point_id);
            deferred_observations.push_back(
                {*map_point_id, match.train_index});
        }
    }

    std::vector<MapPoint> new_map_points;
    new_map_points.reserve(rgbd_correspondences.size());
    for (const auto& correspondence : rgbd_correspondences) {
        if (keyframe.mapPointIds()[correspondence.source_index]) {
            continue;
        }
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

    // Work on copies so a rejected observation or allocation failure cannot
    // leave only half of the keyframe insertion in the live map.
    std::vector<Keyframe> updated_keyframes = keyframes_;
    std::vector<MapPoint> updated_map_points = map_points_;
    for (const auto& observation : deferred_observations) {
        updated_map_points[observation.map_point_id].addObservation(
            {keyframe_id, observation.feature_index});
    }
    for (auto& map_point : new_map_points) {
        updated_map_points.push_back(std::move(map_point));
    }
    updated_keyframes.push_back(std::move(keyframe));
    keyframes_.swap(updated_keyframes);
    map_points_.swap(updated_map_points);

    return {
        keyframe_id,
        new_map_points.size(),
        deferred_observations.size(),
    };
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
