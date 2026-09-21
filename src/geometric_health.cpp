#include "geometric_health.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>

namespace adaptive_fusion_slam {

GeometricHealthMonitor::GeometricHealthMonitor(GeometricHealthConfig config)
    : config_(config) {
    if (config_.coverage_grid_columns <= 0 ||
        config_.coverage_grid_rows <= 0) {
        throw std::invalid_argument(
            "Geometric-health coverage grid dimensions must be positive.");
    }
}

GeometricHealth GeometricHealthMonitor::compute(
    const cv::Size& image_size,
    const std::vector<PixelCorrespondence>& correspondences,
    std::size_t valid_depth_correspondences,
    const std::vector<TrackedFeature>& lk_tracks,
    const PoseEstimate& pose) const {
    if (image_size.width <= 0 || image_size.height <= 0) {
        throw std::invalid_argument(
            "Geometric-health image dimensions must be positive.");
    }

    GeometricHealth health;
    health.has_tracking_measurement = true;
    health.tracking_success = pose.success;
    health.correspondence_count = correspondences.size();
    health.pnp_inlier_count = pose.inlier_indices.size();
    health.inlier_ratio = pose.inlier_ratio;
    health.mean_reprojection_error_pixels =
        pose.mean_reprojection_error_pixels;
    if (!correspondences.empty()) {
        health.valid_depth_ratio =
            static_cast<double>(valid_depth_correspondences) /
            static_cast<double>(correspondences.size());
    }

    if (!lk_tracks.empty()) {
        double error_sum = 0.0;
        for (const auto& track : lk_tracks) {
            error_sum += track.forward_backward_error;
        }
        health.mean_forward_backward_error_pixels =
            error_sum / static_cast<double>(lk_tracks.size());
    }

    if (!correspondences.empty()) {
        std::vector<double> parallaxes;
        parallaxes.reserve(correspondences.size());
        const std::size_t cell_count = static_cast<std::size_t>(
            config_.coverage_grid_columns * config_.coverage_grid_rows);
        std::vector<bool> occupied_cells(cell_count, false);
        for (const auto& correspondence : correspondences) {
            parallaxes.push_back(cv::norm(
                correspondence.current_pixel -
                correspondence.previous_pixel));
            const int column = std::clamp(
                static_cast<int>(correspondence.current_pixel.x /
                                 static_cast<float>(image_size.width) *
                                 config_.coverage_grid_columns),
                0,
                config_.coverage_grid_columns - 1);
            const int row = std::clamp(
                static_cast<int>(correspondence.current_pixel.y /
                                 static_cast<float>(image_size.height) *
                                 config_.coverage_grid_rows),
                0,
                config_.coverage_grid_rows - 1);
            occupied_cells[static_cast<std::size_t>(
                row * config_.coverage_grid_columns + column)] = true;
        }
        const auto middle = parallaxes.begin() +
            static_cast<std::ptrdiff_t>(parallaxes.size() / 2);
        std::nth_element(parallaxes.begin(), middle, parallaxes.end());
        health.median_parallax_pixels = *middle;
        if (parallaxes.size() % 2 == 0) {
            const auto lower = std::max_element(parallaxes.begin(), middle);
            health.median_parallax_pixels =
                (*lower + *middle) * 0.5;
        }
        health.spatial_coverage =
            static_cast<double>(std::count(
                occupied_cells.begin(), occupied_cells.end(), true)) /
            static_cast<double>(cell_count);
    }
    return health;
}

void writeGeometricHealthCsvHeader(std::ostream& output) {
    output << "frame_id,timestamp,has_tracking_measurement,tracking_success,"
              "used_orb_fallback,feature_count,correspondence_count,"
              "pnp_inlier_count,inlier_ratio,mean_reprojection_error_pixels,"
              "mean_forward_backward_error_pixels,spatial_coverage,"
              "median_parallax_pixels,valid_depth_ratio\n";
}

void writeGeometricHealthCsvRow(
    std::ostream& output,
    const GeometricHealth& health) {
    output << std::setprecision(17)
           << health.frame_id << ',' << health.timestamp << ','
           << static_cast<int>(health.has_tracking_measurement) << ','
           << static_cast<int>(health.tracking_success) << ','
           << static_cast<int>(health.used_orb_fallback) << ','
           << health.feature_count << ',' << health.correspondence_count << ','
           << health.pnp_inlier_count << ',' << health.inlier_ratio << ','
           << health.mean_reprojection_error_pixels << ','
           << health.mean_forward_backward_error_pixels << ','
           << health.spatial_coverage << ','
           << health.median_parallax_pixels << ','
           << health.valid_depth_ratio << '\n';
}

}  // namespace adaptive_fusion_slam
