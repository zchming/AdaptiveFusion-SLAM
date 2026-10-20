#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "geometry_snapshot.h"

int main() {
    using namespace adaptive_fusion_slam;
    const auto unique = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto root = std::filesystem::temp_directory_path() / ("afs-snapshot-test-" + unique);
    try {
        if (!std::filesystem::create_directory(root)) throw std::runtime_error("Temporary path collision");
        const Camera camera(500, 500, 320, 240);
        const std::vector<Eigen::Vector3d> xyz = {
            {-.8,-.4,4}, {-.3,.5,4.5}, {.2,-.6,5}, {.7,.4,5.5}, {-.5,.1,6}, {.6,-.2,6.5}};
        std::vector<Keyframe> keys;
        for (std::size_t k = 0; k < 2; ++k) {
            Frame f; f.id = k * 3; f.association.rgb_timestamp = 1 + k;
            f.pose_valid = true; f.pose_world_from_camera.translation().x() = .3*k + .04*k;
            f.depth_image = cv::Mat::zeros(480, 640, CV_16UC1);
            f.features.descriptors = cv::Mat::zeros(6,32,CV_8UC1);
            for (const auto& p : xyz) {
                const Eigen::Vector3d pc = p - Eigen::Vector3d(.3*k,0,0);
                const auto uv = camera.cameraToPixel(pc);
                f.features.keypoints.emplace_back(static_cast<float>(uv.x()),static_cast<float>(uv.y()),1);
                f.depth_image.at<std::uint16_t>(std::lround(uv.y()),std::lround(uv.x())) =
                    static_cast<std::uint16_t>(std::lround(pc.z()*5000));
            }
            keys.emplace_back(k,f);
        }
        std::vector<MapPoint> points;
        for (std::size_t i = 0; i < xyz.size(); ++i) {
            points.emplace_back(100+3*i, xyz[i]+Eigen::Vector3d(.01,-.02,.03),
                                cv::Mat::zeros(1,32,CV_8UC1),MapObservation{0,i});
            points.back().addObservation({1,i});
        }
        LocalBundleAdjustmentConfig config; config.maximum_iterations = 100;
        const auto path = root / "input.snapshot";
        writeGeometrySnapshot(path,camera,config,keys,points,17);
        auto decoded = readGeometrySnapshot(path);
        if (decoded.state_version != 17 || decoded.points.size()!=6 || decoded.keyframes[1].sourceFrameId()!=3 ||
            decoded.points[2].id()!=106) throw std::runtime_error("Roundtrip metadata mismatch");
        bool refused = false;
        try { writeGeometrySnapshot(path,camera,config,keys,points,17); }
        catch (const std::runtime_error&) { refused = true; }
        if (!refused) throw std::runtime_error("Overwrite was not refused");
        LocalBundleAdjuster solver(camera,config);
        const auto native = solver.optimize(keys,points);
        const auto replay = solver.optimize(decoded.keyframes,decoded.points);
        if (!native.optimized || !replay.optimized ||
            !keys[1].poseWorldFromCamera().matrix().isApprox(decoded.keyframes[1].poseWorldFromCamera().matrix(),1e-12) ||
            std::abs(native.final_reprojection_rmse-replay.final_reprojection_rmse)>1e-12)
            throw std::runtime_error("Export changed Ceres solution");
        std::ofstream(root / "bad.snapshot") << "AFS_GEOMETRY 1 META truncated";
        refused = false;
        try { readGeometrySnapshot(root / "bad.snapshot"); }
        catch (const std::exception&) { refused = true; }
        if (!refused) throw std::runtime_error("Truncated input accepted");
        std::filesystem::remove_all(root);
        std::cout << "Geometry snapshot roundtrip and native/replay Ceres equivalence passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "; retained fixture at " << root << '\n';
        return 1;
    }
}
