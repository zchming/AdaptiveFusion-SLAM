#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "camera.h"
#include "failure_prediction_dataset.h"
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
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: run_rgbd_odometry <dataset_root> "
                     "<trajectory.txt> [max_frames] [risk_model.txt]"
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
        if (argc == 5) {
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
        adaptive_fusion_slam::RiskAdaptiveDecision current_decision;
        std::size_t bundle_adjustment_runs = 0;
        double latest_bundle_adjustment_rmse = 0.0;
        std::vector<adaptive_fusion_slam::GeometricHealth> health_sequence;
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
            const auto applied_decision = current_decision;
            const auto result = risk_controller
                ? odometry.process(
                      dataset.loadFrame(index), applied_decision)
                : odometry.process(dataset.loadFrame(index));
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
