#pragma once

#include <cstddef>
#include <vector>

#include "camera.h"
#include "keyframe.h"
#include "map_point.h"

namespace adaptive_fusion_slam {

struct LocalBundleAdjustmentConfig {
    std::size_t window_size = 5;
    std::size_t minimum_point_observations = 2;
    int maximum_iterations = 20;
    double huber_delta_pixels = 2.0;
    double depth_scale = 5000.0;
    double depth_residual_weight = 100.0;
};

struct LocalBundleAdjustmentResult {
    bool optimized = false;
    std::size_t keyframes_optimized = 0;
    std::size_t map_points_optimized = 0;
    std::size_t observations_used = 0;
    std::size_t depth_observations_used = 0;
    double initial_reprojection_rmse = 0.0;
    double final_reprojection_rmse = 0.0;
};

class LocalBundleAdjuster {
public:
    LocalBundleAdjuster(
        Camera camera,
        LocalBundleAdjustmentConfig config = {});

    LocalBundleAdjustmentResult optimize(
        std::vector<Keyframe>& keyframes,
        std::vector<MapPoint>& map_points) const;

private:
    Camera camera_;
    LocalBundleAdjustmentConfig config_;
};

}  // namespace adaptive_fusion_slam
