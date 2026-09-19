#include "camera.h"

#include <stdexcept>

namespace adaptive_fusion_slam {

Camera::Camera(double fx, double fy, double cx, double cy)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy) {
    if (fx_ <= 0.0 || fy_ <= 0.0) {
        throw std::invalid_argument("Camera focal length must be positive.");
    }
}

Eigen::Vector3d Camera::pixelToCamera(
    const Eigen::Vector2d& pixel,
    double depth) const {

    if (depth <= 0.0) {
        throw std::invalid_argument("Depth must be positive.");
    }

    const double u = pixel.x();
    const double v = pixel.y();

    const double x = (u - cx_) * depth / fx_;
    const double y = (v - cy_) * depth / fy_;
    const double z = depth;

    return Eigen::Vector3d(x, y, z);
}

Eigen::Vector2d Camera::cameraToPixel(
    const Eigen::Vector3d& point_camera) const {

    const double x = point_camera.x();
    const double y = point_camera.y();
    const double z = point_camera.z();

    if (z <= 0.0) {
        throw std::invalid_argument(
            "The 3D point must be in front of the camera.");
    }

    const double u = fx_ * x / z + cx_;
    const double v = fy_ * y / z + cy_;

    return Eigen::Vector2d(u, v);
}

}  // namespace adaptive_fusion_slam
