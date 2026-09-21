#include "local_bundle_adjuster.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace adaptive_fusion_slam {
namespace {

struct ReprojectionResidual {
    ReprojectionResidual(
        double observed_x,
        double observed_y,
        const Camera& camera)
        : observed_x(observed_x),
          observed_y(observed_y),
          fx(camera.fx()),
          fy(camera.fy()),
          cx(camera.cx()),
          cy(camera.cy()) {}

    template <typename T>
    bool operator()(
        const T* const pose_camera_from_world,
        const T* const point_world,
        T* residuals) const {
        T point_camera[3];
        ceres::AngleAxisRotatePoint(
            pose_camera_from_world, point_world, point_camera);
        point_camera[0] += pose_camera_from_world[3];
        point_camera[1] += pose_camera_from_world[4];
        point_camera[2] += pose_camera_from_world[5];

        const T predicted_x =
            T(fx) * point_camera[0] / point_camera[2] + T(cx);
        const T predicted_y =
            T(fy) * point_camera[1] / point_camera[2] + T(cy);
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }

    double observed_x;
    double observed_y;
    double fx;
    double fy;
    double cx;
    double cy;
};

struct DepthResidual {
    DepthResidual(double measured_depth, double weight)
        : measured_depth(measured_depth), weight(weight) {}

    template <typename T>
    bool operator()(
        const T* const pose_camera_from_world,
        const T* const point_world,
        T* residual) const {
        T point_camera[3];
        ceres::AngleAxisRotatePoint(
            pose_camera_from_world, point_world, point_camera);
        point_camera[2] += pose_camera_from_world[5];
        residual[0] = T(weight) * (point_camera[2] - T(measured_depth));
        return true;
    }

    double measured_depth;
    double weight;
};

std::array<double, 6> toCameraFromWorldParameters(
    const Eigen::Isometry3d& pose_world_from_camera) {
    const Eigen::Isometry3d pose_camera_from_world =
        pose_world_from_camera.inverse();
    std::array<double, 6> parameters{};
    Eigen::Matrix3d rotation = pose_camera_from_world.rotation();
    ceres::RotationMatrixToAngleAxis(rotation.data(), parameters.data());
    const Eigen::Vector3d translation = pose_camera_from_world.translation();
    parameters[3] = translation.x();
    parameters[4] = translation.y();
    parameters[5] = translation.z();
    return parameters;
}

Eigen::Isometry3d toWorldFromCameraPose(
    const std::array<double, 6>& parameters) {
    double rotation_data[9];
    ceres::AngleAxisToRotationMatrix(parameters.data(), rotation_data);
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> rotation(
        rotation_data);
    Eigen::Isometry3d pose_camera_from_world = Eigen::Isometry3d::Identity();
    pose_camera_from_world.linear() = rotation;
    pose_camera_from_world.translation() = Eigen::Vector3d(
        parameters[3], parameters[4], parameters[5]);
    return pose_camera_from_world.inverse();
}

}  // namespace

LocalBundleAdjuster::LocalBundleAdjuster(
    Camera camera,
    LocalBundleAdjustmentConfig config)
    : camera_(std::move(camera)), config_(config) {
    if (config_.window_size < 2 ||
        config_.minimum_point_observations < 2 ||
        config_.maximum_iterations <= 0 ||
        config_.huber_delta_pixels <= 0.0 || config_.depth_scale <= 0.0 ||
        config_.depth_residual_weight <= 0.0) {
        throw std::invalid_argument(
            "Local bundle-adjustment configuration is invalid.");
    }
}

LocalBundleAdjustmentResult LocalBundleAdjuster::optimize(
    std::vector<Keyframe>& keyframes,
    std::vector<MapPoint>& map_points) const {
    LocalBundleAdjustmentResult result;
    if (keyframes.size() < 2 || map_points.empty()) {
        return result;
    }

    const std::size_t window_begin =
        keyframes.size() > config_.window_size
            ? keyframes.size() - config_.window_size
            : 0;
    const std::size_t window_size = keyframes.size() - window_begin;

    std::vector<std::array<double, 6>> pose_parameters;
    pose_parameters.reserve(window_size);
    for (std::size_t index = window_begin; index < keyframes.size(); ++index) {
        pose_parameters.push_back(
            toCameraFromWorldParameters(
                keyframes[index].poseWorldFromCamera()));
    }

    std::vector<std::array<double, 3>> point_parameters;
    std::vector<std::size_t> optimized_point_indices;
    std::unordered_map<std::size_t, std::size_t> point_parameter_indices;
    for (std::size_t map_index = 0; map_index < map_points.size(); ++map_index) {
        const auto& map_point = map_points[map_index];
        std::size_t observations_in_window = 0;
        for (const auto& observation : map_point.observations()) {
            if (observation.keyframe_id >= window_begin &&
                observation.keyframe_id < keyframes.size()) {
                ++observations_in_window;
            }
        }
        if (observations_in_window < config_.minimum_point_observations) {
            continue;
        }
        const auto& position = map_point.positionWorld();
        point_parameter_indices.emplace(
            map_point.id(), point_parameters.size());
        optimized_point_indices.push_back(map_index);
        point_parameters.push_back({position.x(), position.y(), position.z()});
    }

    if (point_parameters.empty()) {
        return result;
    }

    ceres::Problem problem;
    for (const auto& map_point : map_points) {
        const auto point_parameter =
            point_parameter_indices.find(map_point.id());
        if (point_parameter == point_parameter_indices.end()) {
            continue;
        }
        for (const auto& observation : map_point.observations()) {
            if (observation.keyframe_id < window_begin ||
                observation.keyframe_id >= keyframes.size()) {
                continue;
            }
            const Keyframe& keyframe = keyframes[observation.keyframe_id];
            if (observation.feature_index >= keyframe.keypoints().size()) {
                throw std::out_of_range(
                    "Map observation feature index is out of range.");
            }
            const auto& pixel =
                keyframe.keypoints()[observation.feature_index].pt;
            auto* cost =
                new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 6, 3>(
                    new ReprojectionResidual(pixel.x, pixel.y, camera_));
            problem.AddResidualBlock(
                cost,
                new ceres::HuberLoss(config_.huber_delta_pixels),
                pose_parameters[observation.keyframe_id - window_begin].data(),
                point_parameters[point_parameter->second].data());
            ++result.observations_used;

            const cv::Mat& depth_image = keyframe.depthImage();
            const int depth_x = static_cast<int>(std::lround(pixel.x));
            const int depth_y = static_cast<int>(std::lround(pixel.y));
            if (depth_image.type() == CV_16UC1 && depth_x >= 0 &&
                depth_y >= 0 && depth_x < depth_image.cols &&
                depth_y < depth_image.rows) {
                const std::uint16_t raw_depth =
                    depth_image.at<std::uint16_t>(depth_y, depth_x);
                if (raw_depth > 0) {
                    const double measured_depth =
                        static_cast<double>(raw_depth) / config_.depth_scale;
                    auto* depth_cost =
                        new ceres::AutoDiffCostFunction<DepthResidual, 1, 6, 3>(
                            new DepthResidual(
                                measured_depth,
                                config_.depth_residual_weight));
                    problem.AddResidualBlock(
                        depth_cost,
                        new ceres::HuberLoss(config_.huber_delta_pixels),
                        pose_parameters[
                            observation.keyframe_id - window_begin]
                            .data(),
                        point_parameters[point_parameter->second].data());
                    ++result.depth_observations_used;
                }
            }
        }
    }

    if (result.observations_used == 0) {
        return result;
    }

    const auto compute_reprojection_rmse = [&]() {
        double squared_error_sum = 0.0;
        std::size_t observation_count = 0;
        for (const auto& map_point : map_points) {
            const auto point_parameter =
                point_parameter_indices.find(map_point.id());
            if (point_parameter == point_parameter_indices.end()) {
                continue;
            }
            for (const auto& observation : map_point.observations()) {
                if (observation.keyframe_id < window_begin ||
                    observation.keyframe_id >= keyframes.size()) {
                    continue;
                }
                const auto& pose = pose_parameters[
                    observation.keyframe_id - window_begin];
                const auto& point =
                    point_parameters[point_parameter->second];
                double point_camera[3];
                ceres::AngleAxisRotatePoint(
                    pose.data(), point.data(), point_camera);
                point_camera[0] += pose[3];
                point_camera[1] += pose[4];
                point_camera[2] += pose[5];
                const Eigen::Vector2d predicted = camera_.cameraToPixel(
                    Eigen::Vector3d(
                        point_camera[0], point_camera[1], point_camera[2]));
                const auto& measured = keyframes[observation.keyframe_id]
                                           .keypoints()[observation.feature_index]
                                           .pt;
                squared_error_sum +=
                    (predicted - Eigen::Vector2d(measured.x, measured.y))
                        .squaredNorm();
                ++observation_count;
            }
        }
        return std::sqrt(
            squared_error_sum / static_cast<double>(observation_count));
    };

    result.initial_reprojection_rmse = compute_reprojection_rmse();

    problem.SetParameterBlockConstant(pose_parameters.front().data());
    ceres::Solver::Options options;
    options.max_num_iterations = config_.maximum_iterations;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    result.final_reprojection_rmse = compute_reprojection_rmse();
    if (!summary.IsSolutionUsable()) {
        return result;
    }

    std::vector<Keyframe> updated_keyframes = keyframes;
    std::vector<MapPoint> updated_map_points = map_points;
    for (std::size_t local_index = 0; local_index < window_size;
         ++local_index) {
        updated_keyframes[window_begin + local_index].setPoseWorldFromCamera(
            toWorldFromCameraPose(pose_parameters[local_index]));
    }
    for (std::size_t index = 0; index < optimized_point_indices.size(); ++index) {
        const auto& point = point_parameters[index];
        updated_map_points[optimized_point_indices[index]].setPositionWorld(
            Eigen::Vector3d(point[0], point[1], point[2]));
    }
    keyframes.swap(updated_keyframes);
    map_points.swap(updated_map_points);

    result.optimized = true;
    result.keyframes_optimized = window_size;
    result.map_points_optimized = point_parameters.size();
    return result;
}

}  // namespace adaptive_fusion_slam
