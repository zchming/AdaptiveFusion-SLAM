// Strict, truth-free file adapter for the existing production PnP estimator.
#include "pnp_pose_estimator.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace {
void expect(std::istream& input, const std::string& word) {
    std::string value;
    if (!(input >> value) || value != word)
        throw std::runtime_error("Expected " + word);
}

std::size_t integer(std::istream& input, std::size_t maximum) {
    std::string token;
    if (!(input >> token) || token.empty() ||
        token.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid unsigned integer");
    std::size_t used = 0;
    const auto value = std::stoull(token, &used);
    if (used != token.size() || value > maximum)
        throw std::runtime_error("Integer exceeds input bound");
    return static_cast<std::size_t>(value);
}

double number(std::istream& input) {
    double value;
    if (!(input >> value) || !std::isfinite(value))
        throw std::runtime_error("Expected finite number");
    return value;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5 || std::string(argv[1]) != "--input" ||
            std::string(argv[3]) != "--output") {
            std::cerr << "Usage: native_pnp_pair --input FILE --output NEW_DIRECTORY\n";
            return 2;
        }
        const std::filesystem::path input_path(argv[2]), output_path(argv[4]);
        if (std::filesystem::exists(output_path))
            throw std::runtime_error("Output exists; refuse overwrite");
        if (std::filesystem::file_size(input_path) > 32 * 1024 * 1024)
            throw std::runtime_error("Input exceeds 32 MiB");
        std::ifstream input(input_path);
        expect(input, "AFS_PNP_PAIR");
        if (integer(input, 1) != 1) throw std::runtime_error("Unsupported schema");
        expect(input, "CAMERA");
        const double fx = number(input), fy = number(input);
        const double cx = number(input), cy = number(input);
        const adaptive_fusion_slam::Camera camera(fx, fy, cx, cy);
        expect(input, "SEED");
        const auto seed = integer(input, std::numeric_limits<int>::max());
        expect(input, "CORRESPONDENCES");
        const auto count = integer(input, 100000);
        std::set<std::size_t> ids;
        std::vector<adaptive_fusion_slam::RgbdCorrespondence> correspondences;
        correspondences.reserve(count);
        for (std::size_t row = 0; row < count; ++row) {
            expect(input, "C");
            const auto id = integer(input, std::numeric_limits<int>::max());
            if (!ids.insert(id).second) throw std::runtime_error("Duplicate source ID");
            const double x = number(input), y = number(input), z = number(input);
            const double u = number(input), v = number(input);
            if (z <= 0) throw std::runtime_error("Source depth must be positive");
            correspondences.push_back({id, Eigen::Vector3d(x, y, z),
                                       Eigen::Vector2d(u, v), z});
        }
        expect(input, "END");
        std::string extra;
        if (input >> extra) throw std::runtime_error("Trailing input data");
        if (!std::filesystem::create_directory(output_path))
            throw std::runtime_error("Cannot create fresh output directory");
        cv::setNumThreads(1);
        cv::setRNGSeed(static_cast<int>(seed));
        const adaptive_fusion_slam::PnpPoseEstimator estimator(camera);
        const auto begin = std::chrono::steady_clock::now();
        const auto estimate = estimator.estimate(correspondences);
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        if (estimate.success &&
            (!estimate.rotation_current_from_previous.allFinite() ||
             !estimate.translation_current_from_previous.allFinite() ||
             !std::isfinite(estimate.mean_reprojection_error_pixels)))
            throw std::runtime_error("Nonfinite native estimate");
        std::ofstream output(output_path / "report.json");
        output << std::setprecision(17)
               << "{\n\"schema\":1,\n\"opencv_version\":\"" << CV_VERSION
               << "\",\n\"seed\":" << seed
               << ",\n\"input_count\":" << count
               << ",\n\"applied\":" << (estimate.success ? "true" : "false")
               << ",\n\"reason\":\""
               << (estimate.success ? "native estimator success; not accuracy certificate" :
                   (count < 6 ? "insufficient fit support" : "native estimator rejected"))
               << "\",\n\"solver_wall_s\":" << wall
               << ",\n\"inlier_source_ids\":[";
        for (std::size_t i = 0; i < estimate.inlier_indices.size(); ++i) {
            const auto row = estimate.inlier_indices[i];
            if (row >= correspondences.size()) throw std::runtime_error("Invalid native inlier");
            output << (i ? "," : "") << correspondences[row].source_index;
        }
        output << "],\n\"mean_reprojection_error_pixels\":";
        if (estimate.success) output << estimate.mean_reprojection_error_pixels;
        else output << "null";
        output << ",\n\"t_target_source\":";
        if (!estimate.success) output << "null";
        else {
            output << '[';
            for (int row = 0; row < 4; ++row) {
                output << (row ? ",[" : "[");
                for (int col = 0; col < 4; ++col) {
                    const double value = row == 3 ? (col == 3 ? 1. : 0.) :
                        (col == 3 ? estimate.translation_current_from_previous(row) :
                         estimate.rotation_current_from_previous(row, col));
                    output << (col ? "," : "") << value;
                }
                output << ']';
            }
            output << ']';
        }
        output << "\n}\n";
        output.close();
        if (!output) throw std::runtime_error("Failed to write native report");
        return 0;  // A recorded native rejection is distinct from a process error.
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;  // Preserve any partially created output for diagnosis.
    }
}
