#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace adaptive_fusion_slam {

struct RgbdAssociation {
    double rgb_timestamp;
    std::filesystem::path rgb_path;
    double depth_timestamp;
    std::filesystem::path depth_path;

    double timeDifference() const;
};

struct RgbdFrame {
    RgbdAssociation association;
    cv::Mat rgb_image;
    cv::Mat depth_image;
};

class TumRgbdDataset {
public:
    explicit TumRgbdDataset(
        std::filesystem::path dataset_root,
        double max_time_difference = 0.02);

    void loadAssociations();

    std::size_t size() const;
    const std::vector<RgbdAssociation>& associations() const;
    RgbdFrame loadFrame(std::size_t index) const;

private:
    std::filesystem::path dataset_root_;
    double max_time_difference_;
    std::vector<RgbdAssociation> associations_;
};

}  // namespace adaptive_fusion_slam
