#include "trajectory_evaluator.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <istream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <Eigen/Geometry>

namespace adaptive_fusion_slam {

std::vector<TrajectoryPose> readTumTrajectory(std::istream& input) {
    std::vector<TrajectoryPose> poses;
    std::string line;
    while (std::getline(input, line)) {
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        std::istringstream parser(line);
        double timestamp = 0.0;
        double tx = 0.0;
        double ty = 0.0;
        double tz = 0.0;
        double qx = 0.0;
        double qy = 0.0;
        double qz = 0.0;
        double qw = 0.0;
        if (!(parser >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
            throw std::runtime_error("Malformed TUM trajectory row.");
        }
        if (!poses.empty() && timestamp <= poses.back().timestamp) {
            throw std::runtime_error(
                "TUM trajectory timestamps must be strictly increasing.");
        }
        Eigen::Quaterniond quaternion(qw, qx, qy, qz);
        if (!std::isfinite(timestamp) ||
            !Eigen::Vector3d(tx, ty, tz).allFinite() ||
            !quaternion.coeffs().allFinite() || quaternion.norm() < 1e-12) {
            throw std::runtime_error("TUM trajectory pose is invalid.");
        }
        quaternion.normalize();
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.linear() = quaternion.toRotationMatrix();
        pose.translation() = Eigen::Vector3d(tx, ty, tz);
        poses.push_back({timestamp, pose});
    }
    return poses;
}

TrajectoryEvaluator::TrajectoryEvaluator(TrajectoryEvaluationConfig config)
    : config_(config) {
    if (config_.max_timestamp_difference <= 0.0) {
        throw std::invalid_argument(
            "Trajectory timestamp tolerance must be positive.");
    }
}

std::vector<TrajectoryAssociation> TrajectoryEvaluator::associate(
    const std::vector<TrajectoryPose>& ground_truth,
    const std::vector<TrajectoryPose>& estimated) const {
    std::vector<TrajectoryAssociation> associations;
    std::size_t ground_truth_cursor = 0;
    for (std::size_t estimated_index = 0;
         estimated_index < estimated.size() &&
         ground_truth_cursor < ground_truth.size();
         ++estimated_index) {
        const double timestamp = estimated[estimated_index].timestamp;
        while (ground_truth_cursor + 1 < ground_truth.size() &&
               std::abs(ground_truth[ground_truth_cursor + 1].timestamp -
                        timestamp) <=
                   std::abs(ground_truth[ground_truth_cursor].timestamp -
                            timestamp)) {
            ++ground_truth_cursor;
        }
        const double difference = std::abs(
            ground_truth[ground_truth_cursor].timestamp - timestamp);
        if (difference <= config_.max_timestamp_difference) {
            associations.push_back({
                ground_truth_cursor,
                estimated_index,
                difference,
            });
            ++ground_truth_cursor;
        }
    }
    return associations;
}

TrajectoryEvaluationResult TrajectoryEvaluator::evaluate(
    const std::vector<TrajectoryPose>& ground_truth,
    const std::vector<TrajectoryPose>& estimated) const {
    TrajectoryEvaluationResult result;
    result.ground_truth_pose_count = ground_truth.size();
    result.estimated_pose_count = estimated.size();
    const auto associations = associate(ground_truth, estimated);
    result.matched_pose_count = associations.size();
    if (associations.size() < 3) {
        throw std::runtime_error(
            "Trajectory evaluation requires at least three matched poses.");
    }
    result.estimated_pose_match_ratio = estimated.empty()
        ? 0.0
        : static_cast<double>(associations.size()) /
              static_cast<double>(estimated.size());
    const double ground_truth_duration =
        ground_truth.back().timestamp - ground_truth.front().timestamp;
    const double matched_duration =
        ground_truth[associations.back().ground_truth_index].timestamp -
        ground_truth[associations.front().ground_truth_index].timestamp;
    if (ground_truth_duration > 0.0) {
        result.duration_coverage_ratio = std::clamp(
            matched_duration / ground_truth_duration, 0.0, 1.0);
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic> estimated_positions(
        3, associations.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> truth_positions(
        3, associations.size());
    for (std::size_t index = 0; index < associations.size(); ++index) {
        estimated_positions.col(static_cast<Eigen::Index>(index)) =
            estimated[associations[index].estimated_index]
                .pose_world_from_camera.translation();
        truth_positions.col(static_cast<Eigen::Index>(index)) =
            ground_truth[associations[index].ground_truth_index]
                .pose_world_from_camera.translation();
    }
    const Eigen::Matrix4d alignment_matrix =
        Eigen::umeyama(estimated_positions, truth_positions, false);
    if (!alignment_matrix.allFinite()) {
        throw std::runtime_error("Trajectory alignment failed.");
    }
    result.alignment_ground_truth_from_estimate.matrix() = alignment_matrix;

    double ate_squared_sum = 0.0;
    double rpe_translation_squared_sum = 0.0;
    double rpe_rotation_squared_sum = 0.0;
    std::vector<Eigen::Isometry3d> aligned_estimates;
    aligned_estimates.reserve(associations.size());
    for (const auto& association : associations) {
        const Eigen::Isometry3d aligned =
            result.alignment_ground_truth_from_estimate *
            estimated[association.estimated_index].pose_world_from_camera;
        aligned_estimates.push_back(aligned);
        const Eigen::Vector3d error =
            ground_truth[association.ground_truth_index]
                .pose_world_from_camera.translation() -
            aligned.translation();
        ate_squared_sum += error.squaredNorm();
    }
    for (std::size_t index = 1; index < associations.size(); ++index) {
        const Eigen::Isometry3d truth_relative =
            ground_truth[associations[index - 1].ground_truth_index]
                .pose_world_from_camera.inverse() *
            ground_truth[associations[index].ground_truth_index]
                .pose_world_from_camera;
        const Eigen::Isometry3d estimate_relative =
            aligned_estimates[index - 1].inverse() * aligned_estimates[index];
        const Eigen::Isometry3d relative_error =
            truth_relative.inverse() * estimate_relative;
        rpe_translation_squared_sum +=
            relative_error.translation().squaredNorm();
        const double angle = Eigen::AngleAxisd(relative_error.rotation()).angle();
        rpe_rotation_squared_sum += angle * angle;
    }

    result.ate_translation_rmse_meters = std::sqrt(
        ate_squared_sum / static_cast<double>(associations.size()));
    const double relative_count =
        static_cast<double>(associations.size() - 1);
    result.rpe_translation_rmse_meters =
        std::sqrt(rpe_translation_squared_sum / relative_count);
    result.rpe_rotation_rmse_radians =
        std::sqrt(rpe_rotation_squared_sum / relative_count);
    return result;
}

void writeTrajectoryEvaluationCsvHeader(std::ostream& output) {
    output << "name,ground_truth_poses,estimated_poses,matched_poses,"
              "estimated_match_ratio,duration_coverage_ratio,ate_rmse_m,"
              "rpe_translation_rmse_m,rpe_rotation_rmse_rad\n";
}

void writeTrajectoryEvaluationCsvRow(
    std::ostream& output,
    const char* name,
    const TrajectoryEvaluationResult& result) {
    output << std::setprecision(17) << name << ','
           << result.ground_truth_pose_count << ','
           << result.estimated_pose_count << ',' << result.matched_pose_count
           << ',' << result.estimated_pose_match_ratio << ','
           << result.duration_coverage_ratio << ','
           << result.ate_translation_rmse_meters << ','
           << result.rpe_translation_rmse_meters << ','
           << result.rpe_rotation_rmse_radians << '\n';
}

}  // namespace adaptive_fusion_slam
