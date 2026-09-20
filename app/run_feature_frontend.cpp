#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

#include "orb_feature_extractor.h"
#include "orb_feature_matcher.h"
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

        double mean_hamming_distance = 0.0;
        for (const auto& match : matches) {
            mean_hamming_distance += match.distance;
        }
        if (!matches.empty()) {
            mean_hamming_distance /= static_cast<double>(matches.size());
        }

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
                  << "Mean Hamming distance: " << mean_hamming_distance
                  << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Feature frontend failed: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
