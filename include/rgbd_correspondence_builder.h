#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "camera.h"

namespace adaptive_fusion_slam {

struct DepthConversionConfig {
    double depth_scale = 5000.0;
    double min_depth_meters = 0.1;
    double max_depth_meters = 8.0;
};

struct PixelCorrespondence {
    std::size_t source_index;
    cv::Point2f previous_pixel;
    cv::Point2f current_pixel;
};

struct RgbdCorrespondence {
    std::size_t source_index;
    Eigen::Vector3d point_previous_camera;
    Eigen::Vector2d current_pixel;
    double depth_meters;
};

class RgbdCorrespondenceBuilder {
public:
    RgbdCorrespondenceBuilder(
        Camera camera,
        DepthConversionConfig config = {});

    std::vector<RgbdCorrespondence> build(
        const cv::Mat& previous_depth_image,
        const std::vector<PixelCorrespondence>& pixel_correspondences) const;

    const DepthConversionConfig& config() const;

private:
    Camera camera_;
    DepthConversionConfig config_;
};

}  // namespace adaptive_fusion_slam
