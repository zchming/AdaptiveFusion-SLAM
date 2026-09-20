#include "rgbd_correspondence_builder.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace adaptive_fusion_slam {

RgbdCorrespondenceBuilder::RgbdCorrespondenceBuilder(
    Camera camera,
    DepthConversionConfig config)
    : camera_(std::move(camera)), config_(config) {
    if (config_.depth_scale <= 0.0) {
        throw std::invalid_argument("Depth scale must be positive.");
    }
    if (config_.min_depth_meters <= 0.0 ||
        config_.max_depth_meters <= config_.min_depth_meters) {
        throw std::invalid_argument("Depth limits must define a positive range.");
    }
}

std::vector<RgbdCorrespondence> RgbdCorrespondenceBuilder::build(
    const cv::Mat& previous_depth_image,
    const std::vector<PixelCorrespondence>& pixel_correspondences) const {
    if (previous_depth_image.empty()) {
        throw std::invalid_argument("Depth image is empty.");
    }
    if (previous_depth_image.type() != CV_16UC1) {
        throw std::invalid_argument(
            "TUM depth image must be single-channel unsigned 16-bit data.");
    }

    std::vector<RgbdCorrespondence> correspondences;
    correspondences.reserve(pixel_correspondences.size());

    for (const auto& pixel_correspondence : pixel_correspondences) {
        const cv::Point2f& previous_pixel =
            pixel_correspondence.previous_pixel;
        const cv::Point2f& current_pixel =
            pixel_correspondence.current_pixel;
        if (!std::isfinite(previous_pixel.x) ||
            !std::isfinite(previous_pixel.y) ||
            !std::isfinite(current_pixel.x) ||
            !std::isfinite(current_pixel.y)) {
            continue;
        }
        if (current_pixel.x < 0.0F || current_pixel.y < 0.0F ||
            current_pixel.x >= static_cast<float>(previous_depth_image.cols) ||
            current_pixel.y >= static_cast<float>(previous_depth_image.rows)) {
            continue;
        }

        const int column = cvRound(previous_pixel.x);
        const int row = cvRound(previous_pixel.y);
        if (column < 0 || column >= previous_depth_image.cols ||
            row < 0 || row >= previous_depth_image.rows) {
            continue;
        }

        const std::uint16_t raw_depth =
            previous_depth_image.at<std::uint16_t>(row, column);
        if (raw_depth == 0) {
            continue;
        }

        const double depth_meters =
            static_cast<double>(raw_depth) / config_.depth_scale;
        if (depth_meters < config_.min_depth_meters ||
            depth_meters > config_.max_depth_meters) {
            continue;
        }

        const Eigen::Vector2d previous_pixel_eigen(
            static_cast<double>(previous_pixel.x),
            static_cast<double>(previous_pixel.y));
        const Eigen::Vector3d point_previous_camera =
            camera_.pixelToCamera(previous_pixel_eigen, depth_meters);

        correspondences.push_back({
            pixel_correspondence.source_index,
            point_previous_camera,
            Eigen::Vector2d(
                static_cast<double>(current_pixel.x),
                static_cast<double>(current_pixel.y)),
            depth_meters,
        });
    }

    return correspondences;
}

const DepthConversionConfig& RgbdCorrespondenceBuilder::config() const {
    return config_;
}

}  // namespace adaptive_fusion_slam
