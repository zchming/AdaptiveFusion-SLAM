#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "geometry_snapshot.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: replay_local_ba <input.snapshot> <new_output_directory>\n";
        return 1;
    }
    try {
        auto snapshot = adaptive_fusion_slam::readGeometrySnapshot(argv[1]);
        const std::filesystem::path output(argv[2]);
        if (!std::filesystem::create_directory(output))
            throw std::runtime_error("Output directory already exists");
        std::ofstream report(output / "result.json");
        const auto start = std::chrono::steady_clock::now();
        adaptive_fusion_slam::LocalBundleAdjuster solver(snapshot.camera, snapshot.config);
        const auto result = solver.optimize(snapshot.keyframes, snapshot.points);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        adaptive_fusion_slam::writeGeometrySnapshot(
            output / "optimized.snapshot", snapshot.camera, snapshot.config,
            snapshot.keyframes, snapshot.points, snapshot.state_version, snapshot.source_revision);
        report << std::setprecision(17) << "{\"backend\":\"ceres\",\"optimized\":"
            << (result.optimized ? "true" : "false") << ",\"solve_ms\":" << ms
            << ",\"observations_used\":" << result.observations_used
            << ",\"termination_type\":" << result.termination_type
            << ",\"iterations_including_initial\":" << result.iterations
            << ",\"initial_cost\":" << result.initial_cost
            << ",\"final_cost\":" << result.final_cost << "}\n";
        if (!report) throw std::runtime_error("Cannot write replay report");
        return result.optimized ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "Ceres replay failed: " << e.what() << '\n';
        return 1;
    }
}
