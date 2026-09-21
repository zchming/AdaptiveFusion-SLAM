#pragma once

#include <cstddef>
#include <string>

#include "tum_rgbd_dataset.h"

namespace adaptive_fusion_slam {

enum class ImageDegradationType {
    None,
    MotionBlur,
    LowLight,
    Occlusion,
    GaussianNoise,
    DroppedFrame,
};

struct ImageDegradationConfig {
    ImageDegradationType type = ImageDegradationType::None;
    double severity = 0.7;
    std::size_t drop_interval = 10;
};

class ImageDegrader {
public:
    explicit ImageDegrader(ImageDegradationConfig config = {});
    RgbdFrame apply(const RgbdFrame& frame, std::size_t frame_index) const;
    const ImageDegradationConfig& config() const;

private:
    ImageDegradationConfig config_;
};

ImageDegradationType parseImageDegradationType(const std::string& name);
const char* imageDegradationName(ImageDegradationType type);

}  // namespace adaptive_fusion_slam
