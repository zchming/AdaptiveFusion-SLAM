#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/core.hpp>
#include "orb_feature_matcher.h"
#include "rgbd_correspondence_builder.h"
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
            for (int row=0; row<6; ++row) for (int col=0; col<32; ++col)
                f.features.descriptors.at<std::uint8_t>(row,col)=static_cast<std::uint8_t>((row*37+col*3+7)%256);
            for (const auto& p : xyz) {
                const Eigen::Vector3d pc = p - Eigen::Vector3d(.3*k,0,0);
                const auto uv = camera.cameraToPixel(pc);
                f.features.keypoints.emplace_back(static_cast<float>(uv.x()),static_cast<float>(uv.y()),1);
                f.depth_image.at<std::uint16_t>(std::lround(uv.y()),std::lround(uv.x())) =
                    static_cast<std::uint16_t>(std::lround(pc.z()*5000));
            }
            keys.emplace_back(k,f);
        }
        keys[0].associateMapPoint(2,106);
        const auto native_descriptors=keys[0].descriptors().clone();
        RgbdAssociation association;
        association.rgb_timestamp=1; association.depth_timestamp=.99;
        association.rgb_path=root/"rgb/a.png"; association.depth_path=root/"depth/a.png";
        const auto features_path=root/"native.features";
        writeKeyframeFeatures(features_path,keys[0],association,root,camera,{}, {}, {},0);
        std::ifstream features_input(features_path); std::string line; std::size_t rows=0;
        constexpr char digits[]="0123456789abcdef";
        while (std::getline(features_input,line)) if (line.rfind("F ",0)==0) {
            std::istringstream input(line); std::string tag,hex; std::size_t index; float u,v,size,angle,response;
            int octave,category,raw; long long id;
            input>>tag>>index>>u>>v>>size>>angle>>response>>octave>>category>>raw>>id>>hex;
            if (!input || index!=rows || hex.size()!=64 || (index==2 ? id!=106 : id!=-1))
                throw std::runtime_error("Native feature identity changed");
            const auto& key=keys[0].keypoints()[index];
            const int expected_raw=keys[0].depthImage().at<std::uint16_t>(std::lround(v),std::lround(u));
            if (u!=key.pt.x || v!=key.pt.y || size!=key.size || angle!=key.angle ||
                response!=key.response || octave!=key.octave || category!=key.class_id || raw!=expected_raw)
                throw std::runtime_error("Native feature fields changed");
            for (int col=0;col<32;++col) {
                const auto byte=native_descriptors.at<std::uint8_t>(static_cast<int>(index),col);
                if (hex[2*col]!=digits[byte>>4] || hex[2*col+1]!=digits[byte&15])
                    throw std::runtime_error("Native descriptor byte changed");
            }
            ++rows;
        }
        if (rows!=6 || cv::countNonZero(native_descriptors!=keys[0].descriptors()) ||
            std::filesystem::exists(features_path.string()+".tmp"))
            throw std::runtime_error("Native feature export changed source");
        bool feature_refused=false;
        try { writeKeyframeFeatures(features_path,keys[0],association,root,camera,{}, {}, {},0); }
        catch (const std::runtime_error&) { feature_refused=true; }
        if (!feature_refused) throw std::runtime_error("Feature overwrite accepted");
        association.depth_path=root.parent_path()/"outside.png"; feature_refused=false;
        try { writeKeyframeFeatures(root/"escape.features",keys[0],association,root,camera,{}, {}, {},0); }
        catch (const std::runtime_error&) { feature_refused=true; }
        if (!feature_refused) throw std::runtime_error("Escaping image reference accepted");
        association.depth_path=root/"depth/a.png";
        std::ofstream(root/"partial.features.tmp")<<"partial"; feature_refused=false;
        try { writeKeyframeFeatures(root/"partial.features",keys[0],association,root,camera,{}, {}, {},0); }
        catch (const std::runtime_error&) { feature_refused=true; }
        if (!feature_refused) throw std::runtime_error("Partial feature export overwritten");
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
