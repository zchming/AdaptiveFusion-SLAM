#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

#include "orb_feature_extractor.h"
#include "tum_rgbd_dataset.h"

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: run_feature_frontend <dataset_root> [frame_index]"
                  << std::endl;
        return 1;
    }

    try {
        const std::size_t frame_index =
            argc == 3 ? std::stoull(argv[2]) : 0;

        adaptive_fusion_slam::TumRgbdDataset dataset(argv[1]);
        dataset.loadAssociations();
        const auto frame = dataset.loadFrame(frame_index);

        adaptive_fusion_slam::OrbFeatureExtractor extractor;
        const auto features = extractor.extract(frame.rgb_image);

        std::cout << std::fixed << std::setprecision(6)
                  << "Frame index: " << frame_index << '\n'
                  << "RGB timestamp: " << frame.association.rgb_timestamp << '\n'
                  << "Depth timestamp: " << frame.association.depth_timestamp << '\n'
                  << "Keypoints: " << features.keypoints.size() << '\n'
                  << "Descriptor rows: " << features.descriptors.rows << '\n'
                  << "Descriptor columns: " << features.descriptors.cols
                  << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Feature frontend failed: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
