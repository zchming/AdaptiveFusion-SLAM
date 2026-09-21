#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "camera.h"
#include "failure_prediction_dataset.h"
#include "image_degradation.h"
#include "keyframe_policy.h"
#include "online_risk_controller.h"
#include "rgbd_odometry.h"
#include "sparse_map.h"
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
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: run_rgbd_odometry <dataset_root> "
                     "<trajectory.txt> [max_frames] [risk_model.txt|-] "
                     "[none|blur|dark|occlusion|noise|drop]"
                  << std::endl;
        return 1;
    }

    try {
        adaptive_fusion_slam::TumRgbdDataset dataset(argv[1]);
        dataset.loadAssociations();
        const std::size_t requested_frames =
            argc >= 4 ? std::stoull(argv[3]) : dataset.size();
        const std::size_t frame_count =
            std::min(requested_frames, dataset.size());

        const adaptive_fusion_slam::Camera camera(
            517.3, 516.5, 318.6, 255.3);
        adaptive_fusion_slam::RgbdOdometry odometry(camera);
        adaptive_fusion_slam::Trajectory trajectory;
        adaptive_fusion_slam::SparseMap sparse_map(camera);
        const adaptive_fusion_slam::KeyframePolicy keyframe_policy;
        std::unique_ptr<adaptive_fusion_slam::OnlineRiskController>
            risk_controller;
        if (argc >= 5 && std::string(argv[4]) != "-") {
            std::ifstream model_input(argv[4]);
            if (!model_input) {
                throw std::runtime_error("Cannot open risk model file.");
            }
            adaptive_fusion_slam::TemporalRiskPredictor predictor;
            predictor.load(model_input);
            risk_controller =
                std::make_unique<adaptive_fusion_slam::OnlineRiskController>(
                    std::move(predictor));
        }
        adaptive_fusion_slam::ImageDegradationConfig degradation_config;
        if (argc == 6) {
            degradation_config.type =
                adaptive_fusion_slam::parseImageDegradationType(argv[5]);
        }
        const adaptive_fusion_slam::ImageDegrader degrader(degradation_config);
        adaptive_fusion_slam::RiskAdaptiveDecision current_decision;
        std::size_t bundle_adjustment_runs = 0;
        double latest_bundle_adjustment_rmse = 0.0;
        std::vector<adaptive_fusion_slam::GeometricHealth> health_sequence;
        std::vector<double> frame_times_ms;
        std::size_t lost_frames = 0;
        std::size_t current_lost_streak = 0;
        std::size_t recovery_events = 0;
        std::size_t total_recovery_frames = 0;
        std::size_t maximum_recovery_frames = 0;
        const std::string health_path = std::string(argv[2]) + ".health.csv";
        std::ofstream health_output(health_path);
        if (!health_output) {
            throw std::runtime_error("Cannot open geometric-health CSV file.");
        }
        adaptive_fusion_slam::writeGeometricHealthCsvHeader(health_output);
        const std::string risk_path = std::string(argv[2]) + ".risk.csv";
        std::ofstream risk_output(risk_path);
        if (!risk_output) {
            throw std::runtime_error("Cannot open online-risk CSV file.");
        }
        risk_output << "frame_id,applied_risk,applied_level,"
                       "next_prediction_available,next_risk,next_level,"
                       "allow_new_map_points,map_frozen\n";

        for (std::size_t index = 0; index < frame_count; ++index) {
            const auto frame_start = std::chrono::steady_clock::now();
            const auto applied_decision = current_decision;
            const auto input_frame =
                degrader.apply(dataset.loadFrame(index), index);
            const auto result = risk_controller
                ? odometry.process(
                      input_frame, applied_decision)
                : odometry.process(input_frame);
            adaptive_fusion_slam::OnlineRiskResult next_risk;
            if (risk_controller) {
                next_risk = risk_controller->observe(result.health);
                current_decision = next_risk.decision;
            }
            risk_output << result.frame.id << ','
                        << applied_decision.failure_probability << ','
                        << adaptive_fusion_slam::riskLevelName(
                               applied_decision.level) << ','
                        << static_cast<int>(next_risk.prediction_available)
                        << ',' << next_risk.failure_probability << ','
                        << adaptive_fusion_slam::riskLevelName(
                               next_risk.decision.level) << ','
                        << static_cast<int>(
                               applied_decision.allow_new_map_points) << ','
                        << static_cast<int>(
                               !applied_decision.allow_map_observations) << '\n';
            adaptive_fusion_slam::writeGeometricHealthCsvRow(
                health_output, result.health);
            health_sequence.push_back(result.health);
            trajectory.addFrame(result.frame);
            const bool insert_keyframe = risk_controller
                ? keyframe_policy.shouldInsert(
                      result.frame,
                      sparse_map.lastKeyframe(),
                      applied_decision)
                : keyframe_policy.shouldInsert(
                      result.frame, sparse_map.lastKeyframe());
            if (insert_keyframe) {
                const auto insertion = sparse_map.insertKeyframe(
                    result.frame,
                    {applied_decision.allow_map_observations,
                     applied_decision.allow_new_map_points});
                if (insertion.existing_map_points_observed > 0) {
                    const auto optimization = sparse_map.optimizeLocalMap();
                    if (optimization.optimized) {
                        ++bundle_adjustment_runs;
                        latest_bundle_adjustment_rmse =
                            optimization.final_reprojection_rmse;
                    }
                }
            }
            if (result.status == adaptive_fusion_slam::TrackingStatus::Lost) {
                ++lost_frames;
                ++current_lost_streak;
            } else if (current_lost_streak > 0) {
                ++recovery_events;
                total_recovery_frames += current_lost_streak;
                maximum_recovery_frames = std::max(
                    maximum_recovery_frames, current_lost_streak);
                current_lost_streak = 0;
            }
            const auto frame_end = std::chrono::steady_clock::now();
            frame_times_ms.push_back(std::chrono::duration<double, std::milli>(
                frame_end - frame_start).count());
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
        maximum_recovery_frames = std::max(
            maximum_recovery_frames, current_lost_streak);
        const adaptive_fusion_slam::FailurePredictionDatasetBuilder
            dataset_builder;
        const auto prediction_samples =
            dataset_builder.build(health_sequence);
        const std::string prediction_dataset_path =
            std::string(argv[2]) + ".failure_dataset.csv";
        std::ofstream prediction_dataset_output(prediction_dataset_path);
        if (!prediction_dataset_output) {
            throw std::runtime_error(
                "Cannot open failure-prediction dataset file.");
        }
        adaptive_fusion_slam::writeFailurePredictionCsv(
            prediction_dataset_output,
            prediction_samples,
            dataset_builder.config().history_length);
        std::vector<double> sorted_frame_times = frame_times_ms;
        std::sort(sorted_frame_times.begin(), sorted_frame_times.end());
        const double total_runtime_seconds =
            std::accumulate(frame_times_ms.begin(), frame_times_ms.end(), 0.0) /
            1000.0;
        const double mean_frame_time_ms = frame_times_ms.empty()
            ? 0.0
            : total_runtime_seconds * 1000.0 / frame_times_ms.size();
        const auto percentile = [&sorted_frame_times](double fraction) {
            if (sorted_frame_times.empty()) return 0.0;
            const std::size_t index = static_cast<std::size_t>(
                fraction * static_cast<double>(sorted_frame_times.size() - 1));
            return sorted_frame_times[index];
        };
        const double tracking_success_rate = frame_count == 0
            ? 0.0
            : static_cast<double>(frame_count - lost_frames) /
                  static_cast<double>(frame_count);
        const std::string summary_path = std::string(argv[2]) + ".summary.csv";
        std::ofstream summary_output(summary_path);
        if (!summary_output) {
            throw std::runtime_error("Cannot open runtime summary file.");
        }
        summary_output << "mode,degradation,processed_frames,valid_poses,"
                          "lost_frames,tracking_success_rate,recovery_events,"
                          "mean_recovery_frames,max_recovery_frames,keyframes,"
                          "map_points,local_ba_runs,total_runtime_s,mean_frame_ms,"
                          "p50_frame_ms,p95_frame_ms,fps\n"
                       << (risk_controller ? "adaptive" : "baseline") << ','
                       << adaptive_fusion_slam::imageDegradationName(
                              degradation_config.type) << ','
                       << frame_count << ',' << trajectory.poses().size() << ','
                       << lost_frames << ',' << tracking_success_rate << ','
                       << recovery_events << ','
                       << (recovery_events == 0
                               ? 0.0
                               : static_cast<double>(total_recovery_frames) /
                                     recovery_events) << ','
                       << maximum_recovery_frames << ','
                       << sparse_map.keyframes().size() << ','
                       << sparse_map.mapPoints().size() << ','
                       << bundle_adjustment_runs << ',' << total_runtime_seconds
                       << ',' << mean_frame_time_ms << ',' << percentile(0.50)
                       << ',' << percentile(0.95) << ','
                       << (total_runtime_seconds > 0.0
                               ? frame_count / total_runtime_seconds
                               : 0.0) << '\n';
        std::cout << "Processed frames: " << frame_count << '\n'
                  << "Valid trajectory poses: " << trajectory.poses().size()
                  << '\n'
                  << "Keyframes: " << sparse_map.keyframes().size() << '\n'
                  << "Map points: " << sparse_map.mapPoints().size() << '\n'
                  << "Local BA runs: " << bundle_adjustment_runs << '\n'
                  << "Latest local BA RMSE: "
                  << latest_bundle_adjustment_rmse << " pixels\n"
                  << "Geometric health file: " << health_path << '\n'
                  << "Online risk mode: "
                  << (risk_controller ? "adaptive" : "baseline") << '\n'
                  << "Online risk file: " << risk_path << '\n'
                  << "Degradation: "
                  << adaptive_fusion_slam::imageDegradationName(
                         degradation_config.type) << '\n'
                  << "Lost frames: " << lost_frames << '\n'
                  << "Tracking success rate: " << tracking_success_rate << '\n'
                  << "Mean frame time: " << mean_frame_time_ms << " ms\n"
                  << "P95 frame time: " << percentile(0.95) << " ms\n"
                  << "FPS: "
                  << (total_runtime_seconds > 0.0
                          ? frame_count / total_runtime_seconds
                          : 0.0) << '\n'
                  << "Runtime summary file: " << summary_path << '\n'
                  << "Failure dataset samples: "
                  << prediction_samples.size() << '\n'
                  << "Failure dataset file: "
                  << prediction_dataset_path << '\n'
                  << "Trajectory file: " << argv[2]
                  << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "RGB-D odometry failed: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
