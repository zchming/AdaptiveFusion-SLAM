#include "image_degradation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace adaptive_fusion_slam {

ImageDegrader::ImageDegrader(ImageDegradationConfig config) : config_(config) {
    if (config_.severity < 0.0 || config_.severity > 1.0 ||
        config_.drop_interval == 0) {
        throw std::invalid_argument("Image degradation config is invalid.");
    }
}

RgbdFrame ImageDegrader::apply(
    const RgbdFrame& frame,
    std::size_t frame_index) const {
    RgbdFrame degraded = frame;
    degraded.rgb_image = frame.rgb_image.clone();
    switch (config_.type) {
        case ImageDegradationType::None:
            break;
        case ImageDegradationType::MotionBlur: {
            int kernel = 3 + 2 * static_cast<int>(10.0 * config_.severity);
            kernel = std::max(3, kernel | 1);
            cv::Mat motion_kernel = cv::Mat::zeros(kernel, kernel, CV_32F);
            motion_kernel.row(kernel / 2).setTo(1.0F / kernel);
            cv::filter2D(
                degraded.rgb_image, degraded.rgb_image, -1, motion_kernel);
            break;
        }
        case ImageDegradationType::LowLight:
            degraded.rgb_image.convertTo(
                degraded.rgb_image,
                -1,
                std::max(0.05, 1.0 - 0.9 * config_.severity));
            break;
        case ImageDegradationType::Occlusion: {
            const int width = static_cast<int>(
                degraded.rgb_image.cols * 0.75 * config_.severity);
            const int height = static_cast<int>(
                degraded.rgb_image.rows * 0.75 * config_.severity);
            const cv::Rect region(
                (degraded.rgb_image.cols - width) / 2,
                (degraded.rgb_image.rows - height) / 2,
                width,
                height);
            degraded.rgb_image(region).setTo(cv::Scalar::all(0));
            break;
        }
        case ImageDegradationType::GaussianNoise: {
            cv::Mat noise(
                degraded.rgb_image.size(), CV_16SC3);
            cv::RNG random(static_cast<std::uint64_t>(frame_index + 1729));
            random.fill(
                noise,
                cv::RNG::NORMAL,
                0.0,
                60.0 * config_.severity);
            cv::Mat signed_image;
            degraded.rgb_image.convertTo(signed_image, CV_16SC3);
            signed_image += noise;
            signed_image.convertTo(degraded.rgb_image, CV_8UC3);
            break;
        }
        case ImageDegradationType::DroppedFrame:
            if ((frame_index + 1) % config_.drop_interval == 0) {
                degraded.rgb_image.setTo(cv::Scalar::all(127));
            }
            break;
    }
    return degraded;
}

const ImageDegradationConfig& ImageDegrader::config() const {
    return config_;
}

ImageDegradationType parseImageDegradationType(const std::string& name) {
    if (name == "none") return ImageDegradationType::None;
    if (name == "blur") return ImageDegradationType::MotionBlur;
    if (name == "dark") return ImageDegradationType::LowLight;
    if (name == "occlusion") return ImageDegradationType::Occlusion;
    if (name == "noise") return ImageDegradationType::GaussianNoise;
    if (name == "drop") return ImageDegradationType::DroppedFrame;
    throw std::invalid_argument("Unknown image degradation mode: " + name);
}

const char* imageDegradationName(ImageDegradationType type) {
    switch (type) {
        case ImageDegradationType::None: return "none";
        case ImageDegradationType::MotionBlur: return "blur";
        case ImageDegradationType::LowLight: return "dark";
        case ImageDegradationType::Occlusion: return "occlusion";
        case ImageDegradationType::GaussianNoise: return "noise";
        case ImageDegradationType::DroppedFrame: return "drop";
    }
    return "unknown";
}

}  // namespace adaptive_fusion_slam
