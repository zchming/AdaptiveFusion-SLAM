#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "trajectory_evaluator.h"

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: compare_slam_trajectories <groundtruth.txt> "
                     "<baseline.txt> <adaptive.txt> [report.csv]"
                  << std::endl;
        return 1;
    }
    try {
        std::ifstream truth_input(argv[1]);
        std::ifstream baseline_input(argv[2]);
        std::ifstream adaptive_input(argv[3]);
        if (!truth_input || !baseline_input || !adaptive_input) {
            throw std::runtime_error("Cannot open comparison trajectory file.");
        }
        const auto truth =
            adaptive_fusion_slam::readTumTrajectory(truth_input);
        const auto baseline =
            adaptive_fusion_slam::readTumTrajectory(baseline_input);
        const auto adaptive =
            adaptive_fusion_slam::readTumTrajectory(adaptive_input);
        const adaptive_fusion_slam::TrajectoryEvaluator evaluator;
        const auto baseline_result = evaluator.evaluate(truth, baseline);
        const auto adaptive_result = evaluator.evaluate(truth, adaptive);

        std::cout << std::left << std::setw(12) << "mode"
                  << std::right << std::setw(10) << "matched"
                  << std::setw(12) << "coverage"
                  << std::setw(14) << "ATE(m)"
                  << std::setw(14) << "RPE-t(m)"
                  << std::setw(14) << "RPE-r(rad)" << '\n';
        const auto print_result = [](const char* name, const auto& result) {
            std::cout << std::left << std::setw(12) << name << std::right
                      << std::setw(10) << result.matched_pose_count
                      << std::setw(12) << result.duration_coverage_ratio
                      << std::setw(14) << result.ate_translation_rmse_meters
                      << std::setw(14) << result.rpe_translation_rmse_meters
                      << std::setw(14) << result.rpe_rotation_rmse_radians
                      << '\n';
        };
        print_result("baseline", baseline_result);
        print_result("adaptive", adaptive_result);

        if (argc == 5) {
            std::ofstream report(argv[4]);
            if (!report) {
                throw std::runtime_error("Cannot open comparison report file.");
            }
            adaptive_fusion_slam::writeTrajectoryEvaluationCsvHeader(report);
            adaptive_fusion_slam::writeTrajectoryEvaluationCsvRow(
                report, "baseline", baseline_result);
            adaptive_fusion_slam::writeTrajectoryEvaluationCsvRow(
                report, "adaptive", adaptive_result);
        }
    } catch (const std::exception& error) {
        std::cerr << "Trajectory comparison failed: " << error.what()
                  << std::endl;
        return 1;
    }
    return 0;
}
