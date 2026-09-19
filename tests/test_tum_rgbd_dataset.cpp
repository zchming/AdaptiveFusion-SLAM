#include <cmath>
#include <filesystem>
#include <iostream>

#include <opencv2/core.hpp>

#include "tum_rgbd_dataset.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << std::endl;
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const std::filesystem::path dataset_root = TEST_DATA_DIR;
    adaptive_fusion_slam::TumRgbdDataset dataset(dataset_root, 0.02);
    dataset.loadAssociations();

    bool passed = true;
    passed &= check(dataset.size() == 3, "expected three synchronized pairs");
    if (!passed) {
        return 1;
    }

    const auto& associations = dataset.associations();
    passed &= check(
        std::abs(associations[0].rgb_timestamp - 1.000) < 1e-12,
        "unexpected first RGB timestamp");
    passed &= check(
        std::abs(associations[0].depth_timestamp - 0.999) < 1e-12,
        "unexpected first depth timestamp");
    passed &= check(
        associations[2].timeDifference() <= 0.02,
        "pair exceeds synchronization threshold");

    const auto frame = dataset.loadFrame(0);
    passed &= check(frame.rgb_image.cols == 2 && frame.rgb_image.rows == 2,
                    "unexpected RGB image size");
    passed &= check(frame.rgb_image.channels() == 3,
                    "RGB image must have three channels");
    passed &= check(frame.depth_image.type() == CV_16UC1,
                    "depth image must preserve 16-bit values");

    if (!passed) {
        return 1;
    }

    std::cout << "TUM RGB-D synchronization test passed with "
              << dataset.size() << " pairs." << std::endl;
    return 0;
}
