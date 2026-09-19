#include "tum_rgbd_dataset.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <opencv2/imgcodecs.hpp>

namespace adaptive_fusion_slam {
namespace {

struct TimestampedPath {
    double timestamp;
    std::filesystem::path path;
};

struct CandidateAssociation {
    std::size_t rgb_index;
    std::size_t depth_index;
    double time_difference;
};

std::vector<TimestampedPath> readIndexFile(
    const std::filesystem::path& index_path) {
    std::ifstream input(index_path);
    if (!input) {
        throw std::runtime_error("Cannot open index file: " + index_path.string());
    }

    std::vector<TimestampedPath> entries;
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        const std::size_t first_character = line.find_first_not_of(" \t\r");
        if (first_character == std::string::npos || line[first_character] == '#') {
            continue;
        }

        std::istringstream line_stream(line);
        TimestampedPath entry;
        if (!(line_stream >> entry.timestamp >> entry.path)) {
            throw std::runtime_error(
                "Invalid entry in " + index_path.string() + " at line " +
                std::to_string(line_number));
        }
        entries.push_back(std::move(entry));
    }

    std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) {
        return left.timestamp < right.timestamp;
    });
    return entries;
}

}  // namespace

double RgbdAssociation::timeDifference() const {
    return std::abs(rgb_timestamp - depth_timestamp);
}

TumRgbdDataset::TumRgbdDataset(
    std::filesystem::path dataset_root,
    double max_time_difference)
    : dataset_root_(std::move(dataset_root)),
      max_time_difference_(max_time_difference) {
    if (max_time_difference_ < 0.0) {
        throw std::invalid_argument("Maximum time difference cannot be negative.");
    }
}

void TumRgbdDataset::loadAssociations() {
    const auto rgb_entries = readIndexFile(dataset_root_ / "rgb.txt");
    const auto depth_entries = readIndexFile(dataset_root_ / "depth.txt");

    std::vector<CandidateAssociation> candidates;
    for (std::size_t rgb_index = 0; rgb_index < rgb_entries.size(); ++rgb_index) {
        const double earliest_depth =
            rgb_entries[rgb_index].timestamp - max_time_difference_;
        auto depth_iterator = std::lower_bound(
            depth_entries.begin(),
            depth_entries.end(),
            earliest_depth,
            [](const TimestampedPath& entry, double timestamp) {
                return entry.timestamp < timestamp;
            });

        for (; depth_iterator != depth_entries.end(); ++depth_iterator) {
            if (depth_iterator->timestamp >
                rgb_entries[rgb_index].timestamp + max_time_difference_) {
                break;
            }

            const std::size_t depth_index = static_cast<std::size_t>(
                std::distance(depth_entries.begin(), depth_iterator));
            const double difference = std::abs(
                rgb_entries[rgb_index].timestamp -
                depth_iterator->timestamp);
            candidates.push_back({rgb_index, depth_index, difference});
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return left.time_difference < right.time_difference;
    });

    std::vector<bool> rgb_used(rgb_entries.size(), false);
    std::vector<bool> depth_used(depth_entries.size(), false);
    associations_.clear();

    for (const auto& candidate : candidates) {
        if (rgb_used[candidate.rgb_index] || depth_used[candidate.depth_index]) {
            continue;
        }

        const auto& rgb = rgb_entries[candidate.rgb_index];
        const auto& depth = depth_entries[candidate.depth_index];
        associations_.push_back({
            rgb.timestamp,
            dataset_root_ / rgb.path,
            depth.timestamp,
            dataset_root_ / depth.path,
        });
        rgb_used[candidate.rgb_index] = true;
        depth_used[candidate.depth_index] = true;
    }

    std::sort(
        associations_.begin(),
        associations_.end(),
        [](const auto& left, const auto& right) {
            return left.rgb_timestamp < right.rgb_timestamp;
        });
}

std::size_t TumRgbdDataset::size() const {
    return associations_.size();
}

const std::vector<RgbdAssociation>& TumRgbdDataset::associations() const {
    return associations_;
}

RgbdFrame TumRgbdDataset::loadFrame(std::size_t index) const {
    if (index >= associations_.size()) {
        throw std::out_of_range("RGB-D frame index is out of range.");
    }

    const auto& association = associations_[index];
    cv::Mat rgb_image = cv::imread(association.rgb_path.string(), cv::IMREAD_COLOR);
    cv::Mat depth_image = cv::imread(
        association.depth_path.string(), cv::IMREAD_UNCHANGED);

    if (rgb_image.empty()) {
        throw std::runtime_error(
            "Cannot read RGB image: " + association.rgb_path.string());
    }
    if (depth_image.empty()) {
        throw std::runtime_error(
            "Cannot read depth image: " + association.depth_path.string());
    }
    if (rgb_image.size() != depth_image.size()) {
        throw std::runtime_error("RGB and depth image sizes do not match.");
    }

    return {association, std::move(rgb_image), std::move(depth_image)};
}

}  // namespace adaptive_fusion_slam
