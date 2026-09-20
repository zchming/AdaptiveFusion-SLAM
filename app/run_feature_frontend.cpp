#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

#include "lk_optical_flow_tracker.h"
#include "orb_feature_extractor.h"
#include "orb_feature_matcher.h"
#include "pnp_pose_estimator.h"
#include "rgbd_correspondence_builder.h"
#include "tum_rgbd_dataset.h"

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: run_feature_frontend <dataset_root> "
                     "[first_frame_index]"
                  << std::endl;
        return 1;
    }

    try {
        const std::size_t first_frame_index =
            argc == 3 ? std::stoull(argv[2]) : 0;

        adaptive_fusion_slam::TumRgbdDataset dataset(argv[1]);
        dataset.loadAssociations();
        const auto first_frame = dataset.loadFrame(first_frame_index);
        const auto second_frame = dataset.loadFrame(first_frame_index + 1);

        adaptive_fusion_slam::OrbFeatureExtractor extractor;
        const auto first_features = extractor.extract(first_frame.rgb_image);
        const auto second_features = extractor.extract(second_frame.rgb_image);

        const adaptive_fusion_slam::OrbFeatureMatcher matcher;
        const auto matches = matcher.match(
            first_features.descriptors,
            second_features.descriptors);

        const adaptive_fusion_slam::LkOpticalFlowTracker optical_flow_tracker;
        const auto tracks = optical_flow_tracker.track(
            first_frame.rgb_image,
            second_frame.rgb_image,
            first_features.keypoints);

        std::vector<adaptive_fusion_slam::PixelCorrespondence>
            pixel_correspondences;
        pixel_correspondences.reserve(tracks.size());
        for (const auto& track : tracks) {
            pixel_correspondences.push_back({
                track.source_index,
                track.previous_point,
                track.current_point,
            });
        }

        const adaptive_fusion_slam::Camera camera(
            517.3, 516.5, 318.6, 255.3);
        const adaptive_fusion_slam::RgbdCorrespondenceBuilder
            correspondence_builder(camera);
        const auto rgbd_correspondences = correspondence_builder.build(
            first_frame.depth_image,
            pixel_correspondences);

        const adaptive_fusion_slam::PnpPoseEstimator pose_estimator(camera);
        const auto pose = pose_estimator.estimate(rgbd_correspondences);

        double mean_hamming_distance = 0.0;
        for (const auto& match : matches) {
            mean_hamming_distance += match.distance;
        }
        if (!matches.empty()) {
            mean_hamming_distance /= static_cast<double>(matches.size());
        }

        double mean_forward_backward_error = 0.0;
        for (const auto& track : tracks) {
            mean_forward_backward_error += track.forward_backward_error;
        }
        if (!tracks.empty()) {
            mean_forward_backward_error /= static_cast<double>(tracks.size());
        }

        const double valid_depth_ratio = tracks.empty()
            ? 0.0
            : static_cast<double>(rgbd_correspondences.size()) /
                  static_cast<double>(tracks.size());

        std::cout << std::fixed << std::setprecision(6)
                  << "First frame index: " << first_frame_index << '\n'
                  << "First RGB timestamp: "
                  << first_frame.association.rgb_timestamp << '\n'
                  << "Second RGB timestamp: "
                  << second_frame.association.rgb_timestamp << '\n'
                  << "First-frame keypoints: "
                  << first_features.keypoints.size() << '\n'
                  << "Second-frame keypoints: "
                  << second_features.keypoints.size() << '\n'
                  << "Accepted matches: " << matches.size() << '\n'
                  << "Mean Hamming distance: " << mean_hamming_distance << '\n'
                  << "Accepted LK tracks: " << tracks.size() << '\n'
                  << "Mean forward-backward error: "
                  << mean_forward_backward_error << " pixels\n"
                  << "Valid RGB-D correspondences: "
                  << rgbd_correspondences.size() << '\n'
                  << "Valid-depth ratio: " << valid_depth_ratio << '\n'
                  << "Pose estimation succeeded: "
                  << std::boolalpha << pose.success << '\n'
                  << "PnP inliers: " << pose.inlier_indices.size() << '\n'
                  << "PnP inlier ratio: " << pose.inlier_ratio << '\n'
                  << "Mean reprojection error: "
                  << pose.mean_reprojection_error_pixels << " pixels\n"
                  << "Translation current-from-previous: "
                  << pose.translation_current_from_previous.transpose()
                  << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Feature frontend failed: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
