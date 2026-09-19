#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>

#include "tum_rgbd_dataset.h"

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: inspect_tum_dataset <dataset_root> "
                     "[max_time_difference_seconds]"
                  << std::endl;
        return 1;
    }

    try {
        const double max_time_difference =
            argc == 3 ? std::stod(argv[2]) : 0.02;
        adaptive_fusion_slam::TumRgbdDataset dataset(
            argv[1], max_time_difference);
        dataset.loadAssociations();

        std::cout << "Synchronized RGB-D pairs: " << dataset.size() << std::endl;
        if (dataset.size() == 0) {
            return 0;
        }

        const auto frame = dataset.loadFrame(0);
        std::cout << std::fixed << std::setprecision(6)
                  << "First RGB timestamp: "
                  << frame.association.rgb_timestamp << '\n'
                  << "First depth timestamp: "
                  << frame.association.depth_timestamp << '\n'
                  << "Time difference: "
                  << frame.association.timeDifference() << " s\n"
                  << "Image size: " << frame.rgb_image.cols << " x "
                  << frame.rgb_image.rows << '\n'
                  << "RGB channels: " << frame.rgb_image.channels() << '\n'
                  << "Depth bit depth: "
                  << frame.depth_image.elemSize1() * 8 << " bits"
                  << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Dataset inspection failed: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
