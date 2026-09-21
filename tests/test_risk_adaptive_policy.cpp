#include <cstdint>
#include <iostream>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "camera.h"
#include "frame.h"
#include "keyframe_policy.h"
#include "risk_adaptive_policy.h"
#include "sparse_map.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << std::endl;
        return false;
    }
    return true;
}

adaptive_fusion_slam::Frame makeFrame(std::size_t id) {
    adaptive_fusion_slam::Frame frame;
    frame.id = id;
    frame.association.rgb_timestamp = 0.1 * static_cast<double>(id);
    frame.pose_valid = true;
    frame.pose_world_from_camera = Eigen::Isometry3d::Identity();
    frame.rgb_image = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0));
    frame.depth_image = cv::Mat(480, 640, CV_16UC1, cv::Scalar(0));
    frame.features.keypoints = {
        cv::KeyPoint(320.0F, 240.0F, 31.0F),
        cv::KeyPoint(400.0F, 300.0F, 31.0F),
    };
    frame.features.descriptors = cv::Mat(2, 32, CV_8UC1);
    frame.features.descriptors.row(0).setTo(cv::Scalar(10));
    frame.features.descriptors.row(1).setTo(cv::Scalar(20));
    frame.depth_image.at<std::uint16_t>(240, 320) = 10000;
    frame.depth_image.at<std::uint16_t>(300, 400) = 7500;
    return frame;
}

}  // namespace

int main() {
    const adaptive_fusion_slam::RiskAdaptivePolicy risk_policy;
    const auto low = risk_policy.decide(0.10);
    const auto medium = risk_policy.decide(0.40);
    const auto high = risk_policy.decide(0.70);
    const auto critical = risk_policy.decide(0.90);

    bool passed = true;
    passed &= check(low.level == adaptive_fusion_slam::RiskLevel::Low &&
                        low.allow_new_map_points &&
                        !low.enable_orb_verification,
                    "low risk should retain the efficient normal path");
    passed &= check(medium.level == adaptive_fusion_slam::RiskLevel::Medium &&
                        medium.enable_orb_verification &&
                        medium.request_early_keyframe &&
                        medium.allow_new_map_points,
                    "medium risk should verify and request an early keyframe");
    passed &= check(high.level == adaptive_fusion_slam::RiskLevel::High &&
                        high.force_orb_redetection &&
                        !high.allow_new_map_points &&
                        high.allow_map_observations,
                    "high risk should redetect but suppress new landmarks");
    passed &= check(critical.level ==
                            adaptive_fusion_slam::RiskLevel::Critical &&
                        !critical.allow_keyframe_insertion &&
                        !critical.allow_map_observations &&
                        critical.preserve_trusted_reference,
                    "critical risk should freeze persistent state");

    adaptive_fusion_slam::HystereticRiskAdaptivePolicy hysteretic_policy;
    passed &= check(
        hysteretic_policy.update(0.61).level ==
            adaptive_fusion_slam::RiskLevel::High &&
        hysteretic_policy.update(0.58).level ==
            adaptive_fusion_slam::RiskLevel::High &&
        hysteretic_policy.update(0.54).level ==
            adaptive_fusion_slam::RiskLevel::Medium,
        "hysteresis should prevent threshold chatter before a real decrease");

    const adaptive_fusion_slam::Camera camera(
        517.3, 516.5, 318.6, 255.3);
    adaptive_fusion_slam::SparseMap map(camera);
    auto first = makeFrame(0);
    map.insertKeyframe(first);
    auto second = makeFrame(2);
    second.pose_world_from_camera.translation().x() = 0.08;

    const adaptive_fusion_slam::KeyframePolicy keyframe_policy;
    passed &= check(!keyframe_policy.shouldInsert(second, map.lastKeyframe()),
                    "normal policy should respect the five-frame gap");
    passed &= check(
        keyframe_policy.shouldInsert(second, map.lastKeyframe(), medium),
        "medium risk should permit an earlier informative keyframe");
    passed &= check(
        !keyframe_policy.shouldInsert(second, map.lastKeyframe(), critical),
        "critical risk should block keyframe insertion");

    second.pose_world_from_camera = Eigen::Isometry3d::Identity();
    second.features.keypoints.push_back(
        cv::KeyPoint(500.0F, 350.0F, 31.0F));
    cv::Mat descriptors(3, 32, CV_8UC1);
    second.features.descriptors.copyTo(descriptors.rowRange(0, 2));
    descriptors.row(2).setTo(cv::Scalar(30));
    second.features.descriptors = descriptors;
    second.depth_image.at<std::uint16_t>(350, 500) = 6000;
    const auto insertion = map.insertKeyframe(
        second,
        {high.allow_map_observations, high.allow_new_map_points});
    passed &= check(insertion.existing_map_points_observed == 2 &&
                        insertion.map_points_created == 0 &&
                        insertion.map_points_suppressed == 1,
                    "high-risk map update should reuse trusted points only");
    passed &= check(map.mapPoints().size() == 2,
                    "suppressed high-risk landmark must not contaminate map");

    if (!passed) {
        return 1;
    }
    std::cout << "Risk-adaptive policy test passed: early keyframe enabled, "
                 "one high-risk landmark suppressed, critical map frozen."
              << std::endl;
    return 0;
}
