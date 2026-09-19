#include <iostream>

#include <Eigen/Core>

#include "camera.h"

int main() {
    using adaptive_fusion_slam::Camera;

    const double fx = 517.3;
    const double fy = 516.5;
    const double cx = 318.6;
    const double cy = 255.3;

    const Camera camera(fx, fy, cx, cy);

    const Eigen::Vector2d original_pixel(320.0, 240.0);
    const double depth = 2.0;

    const Eigen::Vector3d point_camera =
        camera.pixelToCamera(original_pixel, depth);

    const Eigen::Vector2d projected_pixel =
        camera.cameraToPixel(point_camera);

    const double round_trip_error =
        (original_pixel - projected_pixel).norm();

    std::cout << "Original pixel: "
              << original_pixel.transpose() << std::endl;

    std::cout << "3D point in camera coordinates: "
              << point_camera.transpose() << std::endl;

    std::cout << "Projected pixel: "
              << projected_pixel.transpose() << std::endl;

    std::cout << "Round-trip error: "
              << round_trip_error << std::endl;

    if (round_trip_error > 1e-9) {
        std::cerr << "Camera model test failed." << std::endl;
        return 1;
    }

    std::cout << "Camera model test passed." << std::endl;
    return 0;
}
