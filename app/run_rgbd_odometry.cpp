#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>

#include "camera.h"
#include "rgbd_odometry.h"
#include "trajectory.h"
#include "tum_rgbd_dataset.h"

namespace {

const char* statusName(adaptive_fusion_slam::TrackingStatus status) {
    using adaptive_fusion_slam::TrackingStatus;
    switch (status) {
        case TrackingStatus::Initialized:
            return "initialized";
        case TrackingStatus::Tracked:
            return "tracked";
        case TrackingStatus::Lost:
            return "lost";
    }
    return "unknown";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: run_rgbd_odometry <dataset_root> "
                     "<trajectory.txt> [max_frames]"
                  << std::endl;
        return 1;
    }

    try {
        adaptive_fusion_slam::TumRgbdDataset dataset(argv[1]);
        dataset.loadAssociations();
        const std::size_t requested_frames =
            argc == 4 ? std::stoull(argv[3]) : dataset.size();
        const std::size_t frame_count =
            std::min(requested_frames, dataset.size());

        const adaptive_fusion_slam::Camera camera(
            517.3, 516.5, 318.6, 255.3);
        adaptive_fusion_slam::RgbdOdometry odometry(camera);
        adaptive_fusion_slam::Trajectory trajectory;

        for (std::size_t index = 0; index < frame_count; ++index) {
            const auto result = odometry.process(dataset.loadFrame(index));
            trajectory.addFrame(result.frame);
            std::cerr << "Frame " << index << ": "
                      << statusName(result.status)
                      << ", RGB-D correspondences "
                      << result.rgbd_correspondences
                      << ", PnP inliers "
                      << result.relative_pose.inlier_indices.size()
                      << std::endl;
        }

        std::ofstream trajectory_output(argv[2]);
        if (!trajectory_output) {
            throw std::runtime_error("Cannot open trajectory output file.");
        }
        trajectory.writeTum(trajectory_output);
        std::cout << "Processed frames: " << frame_count << '\n'
                  << "Valid trajectory poses: " << trajectory.poses().size()
                  << '\n'
                  << "Trajectory file: " << argv[2]
                  << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "RGB-D odometry failed: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
