#pragma once

#include <cstddef>

#include "frame.h"
#include "keyframe.h"

namespace adaptive_fusion_slam {

struct KeyframePolicyConfig {
    std::size_t min_frame_gap = 5;
    std::size_t max_frame_gap = 20;
    double min_translation_meters = 0.15;
    double min_rotation_radians = 0.15;
};

class KeyframePolicy {
public:
    explicit KeyframePolicy(KeyframePolicyConfig config = {});

    bool shouldInsert(const Frame& frame, const Keyframe* last_keyframe) const;
    const KeyframePolicyConfig& config() const;

private:
    KeyframePolicyConfig config_;
};

}  // namespace adaptive_fusion_slam
