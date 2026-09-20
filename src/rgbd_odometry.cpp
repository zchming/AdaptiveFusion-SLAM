#include "rgbd_odometry.h"

#include <utility>
#include <vector>

namespace adaptive_fusion_slam {
namespace {

std::vector<PixelCorrespondence> makeLkCorrespondences(
    const std::vector<TrackedFeature>& tracks) {
    std::vector<PixelCorrespondence> correspondences;
    correspondences.reserve(tracks.size());
    for (const auto& track : tracks) {
        correspondences.push_back({
            track.source_index,
            track.previous_point,
            track.current_point,
        });
    }
    return correspondences;
}

std::vector<PixelCorrespondence> makeOrbCorrespondences(
    const Frame& previous_frame,
    const Frame& current_frame,
    const std::vector<FeatureMatch>& matches) {
    std::vector<PixelCorrespondence> correspondences;
    correspondences.reserve(matches.size());
    for (const auto& match : matches) {
        correspondences.push_back({
            match.query_index,
            previous_frame.features.keypoints[match.query_index].pt,
            current_frame.features.keypoints[match.train_index].pt,
        });
    }
    return correspondences;
}

Eigen::Isometry3d relativeTransform(const PoseEstimate& pose) {
    Eigen::Isometry3d current_from_previous = Eigen::Isometry3d::Identity();
    current_from_previous.linear() = pose.rotation_current_from_previous;
    current_from_previous.translation() =
        pose.translation_current_from_previous;
    return current_from_previous;
}

}  // namespace

RgbdOdometry::RgbdOdometry(Camera camera, RgbdOdometryConfig config)
    : camera_(std::move(camera)),
      feature_extractor_(config.orb_features),
      feature_matcher_(config.orb_matching),
      optical_flow_tracker_(config.lk_tracking),
      correspondence_builder_(camera_, config.depth_conversion),
      pose_estimator_(camera_, config.pnp) {}

OdometryResult RgbdOdometry::process(const RgbdFrame& rgbd_frame) {
    OdometryResult result;
    result.frame.id = next_frame_id_++;
    result.frame.association = rgbd_frame.association;
    result.frame.rgb_image = rgbd_frame.rgb_image;
    result.frame.depth_image = rgbd_frame.depth_image;
    result.frame.features = feature_extractor_.extract(rgbd_frame.rgb_image);

    if (!reference_frame_) {
        result.frame.pose_world_from_camera = Eigen::Isometry3d::Identity();
        result.frame.pose_valid = true;
        result.status = TrackingStatus::Initialized;
        reference_frame_ = result.frame;
        return result;
    }

    const auto tracks = optical_flow_tracker_.track(
        reference_frame_->rgb_image,
        result.frame.rgb_image,
        reference_frame_->features.keypoints);
    result.lk_tracks = tracks.size();

    auto pixel_correspondences = makeLkCorrespondences(tracks);
    auto rgbd_correspondences = correspondence_builder_.build(
        reference_frame_->depth_image,
        pixel_correspondences);
    result.rgbd_correspondences = rgbd_correspondences.size();
    result.relative_pose = pose_estimator_.estimate(rgbd_correspondences);
    if (result.relative_pose.success) {
        result.method = TrackingMethod::LkOpticalFlow;
    } else {
        const auto matches = feature_matcher_.match(
            reference_frame_->features.descriptors,
            result.frame.features.descriptors);
        result.orb_matches = matches.size();
        pixel_correspondences = makeOrbCorrespondences(
            *reference_frame_, result.frame, matches);
        rgbd_correspondences = correspondence_builder_.build(
            reference_frame_->depth_image,
            pixel_correspondences);
        result.rgbd_correspondences = rgbd_correspondences.size();
        result.relative_pose = pose_estimator_.estimate(rgbd_correspondences);
        if (result.relative_pose.success) {
            result.method = TrackingMethod::OrbMatching;
        }
    }

    if (!result.relative_pose.success) {
        result.status = TrackingStatus::Lost;
        return result;
    }

    const Eigen::Isometry3d current_from_previous =
        relativeTransform(result.relative_pose);
    result.frame.pose_world_from_camera =
        reference_frame_->pose_world_from_camera *
        current_from_previous.inverse();
    result.frame.pose_valid = true;
    result.status = TrackingStatus::Tracked;
    reference_frame_ = result.frame;
    return result;
}

const Frame* RgbdOdometry::referenceFrame() const {
    return reference_frame_ ? &(*reference_frame_) : nullptr;
}

}  // namespace adaptive_fusion_slam
