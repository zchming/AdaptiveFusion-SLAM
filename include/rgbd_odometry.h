#pragma once

#include <cstddef>
#include <optional>

#include "camera.h"
#include "frame.h"
#include "geometric_health.h"
#include "lk_optical_flow_tracker.h"
#include "orb_feature_extractor.h"
#include "orb_feature_matcher.h"
#include "pnp_pose_estimator.h"
#include "rgbd_correspondence_builder.h"
#include "tum_rgbd_dataset.h"

namespace adaptive_fusion_slam {

enum class TrackingStatus {
    Initialized,
    Tracked,
    Lost,
};

enum class TrackingMethod {
    None,
    LkOpticalFlow,
    OrbMatching,
};

struct RgbdOdometryConfig {
    OrbFeatureConfig orb_features;
    OrbMatcherConfig orb_matching;
    LkOpticalFlowConfig lk_tracking;
    GeometricHealthConfig geometric_health;
    DepthConversionConfig depth_conversion;
    PnpRansacConfig pnp;
};

struct OdometryResult {
    Frame frame;
    TrackingStatus status = TrackingStatus::Lost;
    TrackingMethod method = TrackingMethod::None;
    std::size_t lk_tracks = 0;
    std::size_t orb_matches = 0;
    std::size_t rgbd_correspondences = 0;
    PoseEstimate relative_pose;
    GeometricHealth health;
};

class RgbdOdometry {
public:
    RgbdOdometry(Camera camera, RgbdOdometryConfig config = {});

    OdometryResult process(const RgbdFrame& rgbd_frame);
    const Frame* referenceFrame() const;

private:
    Camera camera_;
    OrbFeatureExtractor feature_extractor_;
    OrbFeatureMatcher feature_matcher_;
    LkOpticalFlowTracker optical_flow_tracker_;
    RgbdCorrespondenceBuilder correspondence_builder_;
    PnpPoseEstimator pose_estimator_;
    GeometricHealthMonitor health_monitor_;
    std::optional<Frame> reference_frame_;
    std::size_t next_frame_id_ = 0;
};

}  // namespace adaptive_fusion_slam
