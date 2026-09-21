#pragma once

#include <cstddef>
#include <iosfwd>
#include <vector>

#include "trajectory.h"

namespace adaptive_fusion_slam {

struct TrajectoryAssociation {
    std::size_t ground_truth_index;
    std::size_t estimated_index;
    double timestamp_difference;
};

struct TrajectoryEvaluationConfig {
    double max_timestamp_difference = 0.02;
};

struct TrajectoryEvaluationResult {
    std::size_t ground_truth_pose_count = 0;
    std::size_t estimated_pose_count = 0;
    std::size_t matched_pose_count = 0;
    double estimated_pose_match_ratio = 0.0;
    double duration_coverage_ratio = 0.0;
    double ate_translation_rmse_meters = 0.0;
    double rpe_translation_rmse_meters = 0.0;
    double rpe_rotation_rmse_radians = 0.0;
    Eigen::Isometry3d alignment_ground_truth_from_estimate =
        Eigen::Isometry3d::Identity();
};

std::vector<TrajectoryPose> readTumTrajectory(std::istream& input);

class TrajectoryEvaluator {
public:
    explicit TrajectoryEvaluator(TrajectoryEvaluationConfig config = {});

    std::vector<TrajectoryAssociation> associate(
        const std::vector<TrajectoryPose>& ground_truth,
        const std::vector<TrajectoryPose>& estimated) const;
    TrajectoryEvaluationResult evaluate(
        const std::vector<TrajectoryPose>& ground_truth,
        const std::vector<TrajectoryPose>& estimated) const;

private:
    TrajectoryEvaluationConfig config_;
};

void writeTrajectoryEvaluationCsvHeader(std::ostream& output);
void writeTrajectoryEvaluationCsvRow(
    std::ostream& output,
    const char* name,
    const TrajectoryEvaluationResult& result);

}  // namespace adaptive_fusion_slam
