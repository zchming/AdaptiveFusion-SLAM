#include <iostream>
#include <sstream>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "camera.h"
#include "rgbd_odometry.h"
#include "trajectory.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << std::endl;
        return false;
    }
    return true;
}

cv::Mat makeTexture() {
    cv::Mat image(480, 640, CV_8UC1);
    cv::RNG random_generator(13579);
    random_generator.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(5, 5), 1.0);
    cv::Mat color;
    cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
    return color;
}

adaptive_fusion_slam::RgbdFrame makeRgbdFrame(
    double timestamp,
    const cv::Mat& rgb_image) {
    adaptive_fusion_slam::RgbdFrame frame;
    frame.association.rgb_timestamp = timestamp;
    frame.association.depth_timestamp = timestamp;
    frame.rgb_image = rgb_image;
    frame.depth_image = cv::Mat(
        rgb_image.rows,
        rgb_image.cols,
        CV_16UC1,
        cv::Scalar(10000));
    return frame;
}

}  // namespace

int main() {
    const adaptive_fusion_slam::Camera camera(
        517.3, 516.5, 318.6, 255.3);
    adaptive_fusion_slam::RgbdOdometry odometry(camera);
    adaptive_fusion_slam::Trajectory trajectory;

    const cv::Mat first_image = makeTexture();
    cv::Mat second_image;
    const cv::Mat translation =
        (cv::Mat_<double>(2, 3) <<
            1.0, 0.0, 5.0,
            0.0, 1.0, 3.0);
    cv::warpAffine(
        first_image,
        second_image,
        translation,
        first_image.size(),
        cv::INTER_LINEAR,
        cv::BORDER_REFLECT_101);

    const auto first_result = odometry.process(makeRgbdFrame(1.0, first_image));
    const auto second_result = odometry.process(makeRgbdFrame(1.1, second_image));
    trajectory.addFrame(first_result.frame);
    trajectory.addFrame(second_result.frame);

    bool passed = true;
    passed &= check(
        first_result.status == adaptive_fusion_slam::TrackingStatus::Initialized,
        "first frame should initialize odometry");
    passed &= check(
        second_result.status == adaptive_fusion_slam::TrackingStatus::Tracked,
        "translated second frame should be tracked");
    passed &= check(
        second_result.method ==
            adaptive_fusion_slam::TrackingMethod::LkOpticalFlow,
        "normal tracking should use LK optical flow");
    passed &= check(second_result.health.has_tracking_measurement &&
                        second_result.health.tracking_success &&
                        second_result.health.inlier_ratio > 0.9 &&
                        second_result.health.valid_depth_ratio > 0.9,
                    "tracked frame should expose strong geometric health");
    passed &= check(trajectory.poses().size() == 2,
                    "trajectory should contain two valid poses");
    const Eigen::Vector3d expected_world_translation(
        -5.0 * 2.0 / 517.3,
        -3.0 * 2.0 / 516.5,
        0.0);
    passed &= check(
        (second_result.frame.pose_world_from_camera.translation() -
         expected_world_translation).norm() < 1e-3,
        "accumulated world-camera translation is incorrect");
    passed &= check(
        Eigen::AngleAxisd(
            second_result.frame.pose_world_from_camera.linear()).angle() < 1e-3,
        "pure image translation should produce negligible rotation");

    std::ostringstream tum_output;
    trajectory.writeTum(tum_output);
    passed &= check(tum_output.str().find("1.000000000") != std::string::npos &&
                        tum_output.str().find("1.100000000") != std::string::npos,
                    "TUM output should contain both timestamps");

    const cv::Mat blank_image(
        first_image.rows,
        first_image.cols,
        CV_8UC3,
        cv::Scalar(127, 127, 127));
    const auto lost_result = odometry.process(makeRgbdFrame(1.2, blank_image));
    trajectory.addFrame(lost_result.frame);
    passed &= check(
        lost_result.status == adaptive_fusion_slam::TrackingStatus::Lost,
        "blank frame should be reported as lost");
    passed &= check(lost_result.health.has_tracking_measurement &&
                        !lost_result.health.tracking_success &&
                        lost_result.health.used_orb_fallback,
                    "lost frame should retain failed health evidence");
    passed &= check(trajectory.poses().size() == 2,
                    "lost frame must not enter the trajectory");
    passed &= check(odometry.referenceFrame() != nullptr &&
                        odometry.referenceFrame()->id == second_result.frame.id,
                    "lost frame must not replace the trusted reference frame");

    adaptive_fusion_slam::RgbdOdometryConfig fallback_config;
    fallback_config.lk_tracking.min_eigenvalue_threshold = 10.0;
    adaptive_fusion_slam::RgbdOdometry fallback_odometry(camera, fallback_config);
    fallback_odometry.process(makeRgbdFrame(2.0, first_image));
    const auto fallback_result =
        fallback_odometry.process(makeRgbdFrame(2.1, second_image));
    passed &= check(
        fallback_result.status == adaptive_fusion_slam::TrackingStatus::Tracked,
        "ORB fallback should recover tracking when LK has no valid tracks");
    passed &= check(
        fallback_result.method ==
            adaptive_fusion_slam::TrackingMethod::OrbMatching,
        "failed LK tracking should select ORB matching fallback");

    if (!passed) {
        return 1;
    }

    std::cout << "RGB-D odometry test passed: initialized frame 0, tracked frame 1 "
                 "with "
              << second_result.relative_pose.inlier_indices.size()
              << " LK/PnP inliers, recovered with ORB fallback using "
              << fallback_result.relative_pose.inlier_indices.size()
              << " inliers, rejected lost frame 2, accumulated translation "
              << second_result.frame.pose_world_from_camera.translation().transpose()
              << ", and wrote "
              << trajectory.poses().size() << " TUM poses." << std::endl;
    return 0;
}
