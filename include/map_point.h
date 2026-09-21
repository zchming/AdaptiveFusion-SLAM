#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>

namespace adaptive_fusion_slam {

struct MapObservation {
    std::size_t keyframe_id;
    std::size_t feature_index;
};

class MapPoint {
public:
    MapPoint(
        std::size_t id,
        Eigen::Vector3d position_world,
        const cv::Mat& descriptor,
        MapObservation first_observation);

    std::size_t id() const;
    const Eigen::Vector3d& positionWorld() const;
    void setPositionWorld(const Eigen::Vector3d& position_world);
    const cv::Mat& descriptor() const;
    const std::vector<MapObservation>& observations() const;
    bool addObservation(MapObservation observation);

private:
    std::size_t id_;
    Eigen::Vector3d position_world_;
    cv::Mat descriptor_;
    std::vector<MapObservation> observations_;
};

}  // namespace adaptive_fusion_slam
