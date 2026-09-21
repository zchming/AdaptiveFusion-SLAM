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
      pose_estimator_(camera_, config.pnp),
      health_monitor_(config.geometric_health) {}

OdometryResult RgbdOdometry::process(const RgbdFrame& rgbd_frame) {
    return process(rgbd_frame, RiskAdaptiveDecision{});
}

OdometryResult RgbdOdometry::process(
    const RgbdFrame& rgbd_frame,
    const RiskAdaptiveDecision& decision) {
    OdometryResult result;
    result.frame.id = next_frame_id_++;
    result.frame.association = rgbd_frame.association;
    result.frame.rgb_image = rgbd_frame.rgb_image;
    result.frame.depth_image = rgbd_frame.depth_image;
    result.frame.features = feature_extractor_.extract(rgbd_frame.rgb_image);
    result.health.frame_id = result.frame.id;
    result.health.timestamp = result.frame.association.rgb_timestamp;
    result.health.feature_count = result.frame.features.keypoints.size();

    if (!reference_frame_) {
        result.frame.pose_world_from_camera = Eigen::Isometry3d::Identity();
        result.frame.pose_valid = true;
        result.status = TrackingStatus::Initialized;
        result.health.tracking_success = true;
        reference_frame_ = result.frame;
        return result;
    }

    const auto tracks = optical_flow_tracker_.track(
        reference_frame_->rgb_image,
        result.frame.rgb_image,
        reference_frame_->features.keypoints);
    result.lk_tracks = tracks.size();

    const auto lk_pixel_correspondences = makeLkCorrespondences(tracks);
    const auto lk_rgbd_correspondences = correspondence_builder_.build(
        reference_frame_->depth_image,
        lk_pixel_correspondences);
    auto pixel_correspondences = lk_pixel_correspondences;
    auto rgbd_correspondences = lk_rgbd_correspondences;
    result.rgbd_correspondences = rgbd_correspondences.size();
    if (!decision.force_orb_redetection) {
        result.relative_pose = pose_estimator_.estimate(rgbd_correspondences);
    }
    bool orb_fallback_attempted = false;
    if (result.relative_pose.success && !decision.enable_orb_verification) {
        result.method = TrackingMethod::LkOpticalFlow;
    } else {
        orb_fallback_attempted = true;
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
        const PoseEstimate orb_pose =
            pose_estimator_.estimate(rgbd_correspondences);
        const bool prefer_orb = orb_pose.success &&
            (!result.relative_pose.success ||
             orb_pose.inlier_ratio > result.relative_pose.inlier_ratio ||
             (orb_pose.inlier_ratio == result.relative_pose.inlier_ratio &&
              orb_pose.mean_reprojection_error_pixels <
                  result.relative_pose.mean_reprojection_error_pixels));
        if (prefer_orb) {
            result.relative_pose = orb_pose;
            result.method = TrackingMethod::OrbMatching;
        } else if (result.relative_pose.success) {
            result.method = TrackingMethod::LkOpticalFlow;
            pixel_correspondences = lk_pixel_correspondences;
            rgbd_correspondences = lk_rgbd_correspondences;
            result.rgbd_correspondences = rgbd_correspondences.size();
        }
    }

    result.health = health_monitor_.compute(
        result.frame.rgb_image.size(),
        pixel_correspondences,
        rgbd_correspondences.size(),
        tracks,
        result.relative_pose);
    result.health.frame_id = result.frame.id;
    result.health.timestamp = result.frame.association.rgb_timestamp;
    result.health.feature_count = result.frame.features.keypoints.size();
    result.health.used_orb_fallback = orb_fallback_attempted;

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
    if (!decision.preserve_trusted_reference) {
        reference_frame_ = result.frame;
    }
    return result;
}

const Frame* RgbdOdometry::referenceFrame() const {
    return reference_frame_ ? &(*reference_frame_) : nullptr;
}

}  // namespace adaptive_fusion_slam
