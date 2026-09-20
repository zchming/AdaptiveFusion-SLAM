#pragma once

#include <Eigen/Core>

namespace adaptive_fusion_slam {

class Camera {
public:
    Camera(double fx, double fy, double cx, double cy);

    Eigen::Vector3d pixelToCamera(
        const Eigen::Vector2d& pixel,
        double depth) const;

    Eigen::Vector2d cameraToPixel(
        const Eigen::Vector3d& point_camera) const;

    double fx() const;
    double fy() const;
    double cx() const;
    double cy() const;

private:
    double fx_;
    double fy_;
    double cx_;
    double cy_;
};

}  // namespace adaptive_fusion_slam
