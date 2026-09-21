#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include <opencv2/core.hpp>

#include "camera.h"
#include "frame.h"
#include "keyframe_policy.h"
#include "sparse_map.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << std::endl;
        return false;
    }
    return true;
}

adaptive_fusion_slam::Frame makeFrame() {
    adaptive_fusion_slam::Frame frame;
    frame.id = 0;
    frame.association.rgb_timestamp = 1.0;
    frame.rgb_image = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    frame.depth_image = cv::Mat(480, 640, CV_16UC1, cv::Scalar(0));
    frame.depth_image.at<std::uint16_t>(240, 320) = 10000;
    frame.depth_image.at<std::uint16_t>(300, 400) = 5000;
    frame.features.keypoints = {
        cv::KeyPoint(320.0F, 240.0F, 31.0F),
        cv::KeyPoint(400.0F, 300.0F, 31.0F),
        cv::KeyPoint(100.0F, 100.0F, 31.0F),
    };
    frame.features.descriptors = cv::Mat(3, 32, CV_8UC1);
    frame.features.descriptors.row(0).setTo(cv::Scalar(10));
    frame.features.descriptors.row(1).setTo(cv::Scalar(20));
    frame.features.descriptors.row(2).setTo(cv::Scalar(30));
    frame.pose_world_from_camera = Eigen::Isometry3d::Identity();
    frame.pose_world_from_camera.translation() = Eigen::Vector3d(1.0, 2.0, 0.0);
    frame.pose_valid = true;
    return frame;
}

}  // namespace

int main() {
    const adaptive_fusion_slam::Camera camera(
        517.3, 516.5, 318.6, 255.3);
    adaptive_fusion_slam::SparseMap sparse_map(camera);
    const auto frame = makeFrame();
    const auto insertion = sparse_map.insertKeyframe(frame);

    bool passed = true;
    passed &= check(insertion.keyframe_id == 0,
                    "first keyframe id should be zero");
    passed &= check(insertion.map_points_created == 2,
                    "only two features should have valid depth");
    passed &= check(insertion.existing_map_points_observed == 0,
                    "first keyframe cannot observe an existing point");
    passed &= check(sparse_map.keyframes().size() == 1,
                    "map should contain one keyframe");
    passed &= check(sparse_map.mapPoints().size() == 2,
                    "map should contain two points");

    const auto& first_point = sparse_map.mapPoints()[0];
    const Eigen::Vector3d expected_camera_point(
        (320.0 - 318.6) * 2.0 / 517.3,
        (240.0 - 255.3) * 2.0 / 516.5,
        2.0);
    const Eigen::Vector3d expected_world_point =
        frame.pose_world_from_camera * expected_camera_point;
    passed &= check(
        (first_point.positionWorld() - expected_world_point).norm() < 1e-9,
        "map point should be transformed into world coordinates");
    passed &= check(first_point.observations().size() == 1,
                    "new map point should have one observation");
    passed &= check(
        sparse_map.keyframes()[0].mapPointIds()[0].has_value() &&
            *sparse_map.keyframes()[0].mapPointIds()[0] == first_point.id(),
        "keyframe feature should reference its map point");
    passed &= check(first_point.descriptor().at<unsigned char>(0, 0) == 10,
                    "map point should retain its ORB descriptor");

    adaptive_fusion_slam::KeyframePolicy policy;
    adaptive_fusion_slam::Frame nearby_frame = frame;
    nearby_frame.id = 1;
    nearby_frame.pose_world_from_camera.translation().x() += 0.01;
    passed &= check(!policy.shouldInsert(
                        nearby_frame, sparse_map.lastKeyframe()),
                    "nearby frame should not become a keyframe");

    adaptive_fusion_slam::Frame moved_frame = frame;
    moved_frame.id = 5;
    moved_frame.pose_world_from_camera.translation().x() += 0.20;
    passed &= check(policy.shouldInsert(
                        moved_frame, sparse_map.lastKeyframe()),
                    "sufficient motion should request a keyframe");

    adaptive_fusion_slam::Frame invalid_frame = moved_frame;
    invalid_frame.pose_valid = false;
    passed &= check(!policy.shouldInsert(
                        invalid_frame, sparse_map.lastKeyframe()),
                    "invalid frame must not become a keyframe");
    bool rejected_invalid_insertion = false;
    try {
        sparse_map.insertKeyframe(invalid_frame);
    } catch (const std::invalid_argument&) {
        rejected_invalid_insertion = true;
    }
    passed &= check(rejected_invalid_insertion,
                    "map must reject invalid keyframe insertion");

    adaptive_fusion_slam::Frame malformed_depth_frame = moved_frame;
    malformed_depth_frame.depth_image = cv::Mat(
        480, 640, CV_8UC1, cv::Scalar(1));
    bool rejected_malformed_depth = false;
    try {
        sparse_map.insertKeyframe(malformed_depth_frame);
    } catch (const std::invalid_argument&) {
        rejected_malformed_depth = true;
    }
    passed &= check(rejected_malformed_depth,
                    "map must reject malformed depth data");
    passed &= check(sparse_map.keyframes().size() == 1 &&
                        sparse_map.mapPoints().size() == 2,
                    "failed insertion must not partially modify the map");

    adaptive_fusion_slam::Frame second_frame = frame;
    second_frame.id = 10;
    second_frame.association.rgb_timestamp = 1.1;
    second_frame.features.keypoints.push_back(
        cv::KeyPoint(500.0F, 350.0F, 31.0F));
    cv::Mat second_descriptors(4, 32, CV_8UC1);
    frame.features.descriptors.copyTo(second_descriptors.rowRange(0, 3));
    second_descriptors.row(3).setTo(cv::Scalar(40));
    second_frame.features.descriptors = second_descriptors;
    second_frame.depth_image.at<std::uint16_t>(350, 500) = 7500;

    const auto second_insertion = sparse_map.insertKeyframe(second_frame);
    passed &= check(second_insertion.keyframe_id == 1,
                    "second keyframe id should be one");
    passed &= check(second_insertion.existing_map_points_observed == 2,
                    "two geometrically consistent points should be reused");
    passed &= check(second_insertion.map_points_created == 1,
                    "only the unmatched valid-depth feature should create a point");
    passed &= check(sparse_map.keyframes().size() == 2 &&
                        sparse_map.mapPoints().size() == 3,
                    "association should prevent duplicate map points");
    passed &= check(sparse_map.mapPoints()[0].observations().size() == 2 &&
                        sparse_map.mapPoints()[1].observations().size() == 2,
                    "reused map points should record both observations");
    const auto& second_ids = sparse_map.keyframes()[1].mapPointIds();
    passed &= check(second_ids[0] && *second_ids[0] == 0 &&
                        second_ids[1] && *second_ids[1] == 1 &&
                        second_ids[3] && *second_ids[3] == 2,
                    "second keyframe should link reused and new map points");

    adaptive_fusion_slam::SparseMap geometry_checked_map(camera);
    geometry_checked_map.insertKeyframe(frame);
    adaptive_fusion_slam::Frame inconsistent_frame = frame;
    inconsistent_frame.id = 11;
    inconsistent_frame.features.keypoints[0].pt.x += 50.0F;
    inconsistent_frame.depth_image.at<std::uint16_t>(240, 370) = 10000;
    const auto inconsistent_insertion =
        geometry_checked_map.insertKeyframe(inconsistent_frame);
    passed &= check(inconsistent_insertion.existing_map_points_observed == 1 &&
                        inconsistent_insertion.map_points_created == 1,
                    "descriptor agreement must not override a bad reprojection");
    passed &= check(
        geometry_checked_map.keyframes()[1].mapPointIds()[0] &&
            *geometry_checked_map.keyframes()[1].mapPointIds()[0] == 2,
        "geometrically inconsistent feature should create a separate point");

    if (!passed) {
        return 1;
    }

    std::cout << "Sparse-map test passed with "
              << sparse_map.keyframes().size() << " keyframes and "
              << sparse_map.mapPoints().size()
              << " metric map points; cross-keyframe observations were "
                 "associated and failed insertions were isolated."
              << std::endl;
    return 0;
}
