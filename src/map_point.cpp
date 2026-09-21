#include "map_point.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace adaptive_fusion_slam {

MapPoint::MapPoint(
    std::size_t id,
    Eigen::Vector3d position_world,
    const cv::Mat& descriptor,
    MapObservation first_observation)
    : id_(id),
      position_world_(std::move(position_world)),
      descriptor_(descriptor.clone()),
      observations_{first_observation} {
    if (!position_world_.allFinite()) {
        throw std::invalid_argument("Map-point position must be finite.");
    }
    if (descriptor_.rows != 1 || descriptor_.cols != 32 ||
        descriptor_.type() != CV_8UC1) {
        throw std::invalid_argument(
            "Map-point descriptor must be one 32-byte ORB row.");
    }
}

std::size_t MapPoint::id() const {
    return id_;
}

const Eigen::Vector3d& MapPoint::positionWorld() const {
    return position_world_;
}

const cv::Mat& MapPoint::descriptor() const {
    return descriptor_;
}

const std::vector<MapObservation>& MapPoint::observations() const {
    return observations_;
}

bool MapPoint::addObservation(MapObservation observation) {
    const auto duplicate = std::find_if(
        observations_.begin(),
        observations_.end(),
        [&observation](const MapObservation& existing) {
            return existing.keyframe_id == observation.keyframe_id &&
                   existing.feature_index == observation.feature_index;
        });
    if (duplicate != observations_.end()) {
        return false;
    }
    observations_.push_back(observation);
    return true;
}

}  // namespace adaptive_fusion_slam
