#pragma once

#include <cstddef>
#include <iosfwd>
#include <vector>

#include <opencv2/core/types.hpp>

#include "lk_optical_flow_tracker.h"
#include "pnp_pose_estimator.h"
#include "rgbd_correspondence_builder.h"

namespace adaptive_fusion_slam {

struct GeometricHealth {
    std::size_t frame_id = 0;
    double timestamp = 0.0;
    bool has_tracking_measurement = false;
    bool tracking_success = false;
    bool used_orb_fallback = false;
    std::size_t feature_count = 0;
    std::size_t correspondence_count = 0;
    std::size_t pnp_inlier_count = 0;
    double inlier_ratio = 0.0;
    double mean_reprojection_error_pixels = 0.0;
    double mean_forward_backward_error_pixels = 0.0;
    double spatial_coverage = 0.0;
    double median_parallax_pixels = 0.0;
    double valid_depth_ratio = 0.0;
};

struct GeometricHealthConfig {
    int coverage_grid_columns = 4;
    int coverage_grid_rows = 3;
};

class GeometricHealthMonitor {
public:
    explicit GeometricHealthMonitor(GeometricHealthConfig config = {});

    GeometricHealth compute(
        const cv::Size& image_size,
        const std::vector<PixelCorrespondence>& correspondences,
        std::size_t valid_depth_correspondences,
        const std::vector<TrackedFeature>& lk_tracks,
        const PoseEstimate& pose) const;

private:
    GeometricHealthConfig config_;
};

void writeGeometricHealthCsvHeader(std::ostream& output);
void writeGeometricHealthCsvRow(
    std::ostream& output,
    const GeometricHealth& health);

}  // namespace adaptive_fusion_slam
