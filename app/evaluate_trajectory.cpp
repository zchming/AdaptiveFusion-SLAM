#include <fstream>
#include <iostream>
#include <string>

#include "trajectory_evaluator.h"

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: evaluate_trajectory <groundtruth.txt> "
                     "<estimate.txt> [max_time_diff] [report.csv]"
                  << std::endl;
        return 1;
    }
    try {
        std::ifstream ground_truth_input(argv[1]);
        std::ifstream estimated_input(argv[2]);
        if (!ground_truth_input || !estimated_input) {
            throw std::runtime_error("Cannot open trajectory input file.");
        }
        adaptive_fusion_slam::TrajectoryEvaluationConfig config;
        if (argc >= 4) {
            config.max_timestamp_difference = std::stod(argv[3]);
        }
        const auto ground_truth =
            adaptive_fusion_slam::readTumTrajectory(ground_truth_input);
        const auto estimated =
            adaptive_fusion_slam::readTumTrajectory(estimated_input);
        const adaptive_fusion_slam::TrajectoryEvaluator evaluator(config);
        const auto result = evaluator.evaluate(ground_truth, estimated);

        std::cout << "Ground-truth poses: " << result.ground_truth_pose_count
                  << '\n' << "Estimated poses: " << result.estimated_pose_count
                  << '\n' << "Matched poses: " << result.matched_pose_count
                  << '\n' << "Estimated match ratio: "
                  << result.estimated_pose_match_ratio
                  << '\n' << "Duration coverage ratio: "
                  << result.duration_coverage_ratio
                  << '\n' << "ATE translation RMSE: "
                  << result.ate_translation_rmse_meters << " m"
                  << '\n' << "RPE translation RMSE: "
                  << result.rpe_translation_rmse_meters << " m"
                  << '\n' << "RPE rotation RMSE: "
                  << result.rpe_rotation_rmse_radians << " rad" << std::endl;
        if (argc == 5) {
            std::ofstream report(argv[4]);
            if (!report) {
                throw std::runtime_error("Cannot open evaluation report file.");
            }
            adaptive_fusion_slam::writeTrajectoryEvaluationCsvHeader(report);
            adaptive_fusion_slam::writeTrajectoryEvaluationCsvRow(
                report, "estimate", result);
        }
    } catch (const std::exception& error) {
        std::cerr << "Trajectory evaluation failed: " << error.what()
                  << std::endl;
        return 1;
    }
    return 0;
}
