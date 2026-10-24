#include "geometry_snapshot.h"
#include "orb_feature_matcher.h"
#include "rgbd_correspondence_builder.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

#ifndef AFS_SOURCE_REVISION
#define AFS_SOURCE_REVISION "unknown"
#endif

namespace adaptive_fusion_slam {
namespace {
void token(std::istream& in, const char* expected) {
    std::string actual;
    if (!(in >> actual) || actual != expected)
        throw std::runtime_error(std::string("Invalid snapshot token: ") + expected);
}
void matrix(std::ostream& out, const Eigen::Isometry3d& pose) {
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col) out << ' ' << pose.matrix()(row, col);
}
Eigen::Isometry3d matrix(std::istream& in) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col) in >> pose.matrix()(row, col);
    if (!in || !pose.matrix().allFinite() ||
        !pose.matrix().row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-9) ||
        !(pose.rotation().transpose() * pose.rotation()).isApprox(
            Eigen::Matrix3d::Identity(), 1e-8) ||
        std::abs(pose.rotation().determinant() - 1) > 1e-8)
        throw std::runtime_error("Invalid snapshot SE(3) pose");
    return pose;
}
std::size_t count(std::istream& in, std::size_t limit) {
    std::size_t n = 0;
    if (!(in >> n) || n > limit) throw std::runtime_error("Invalid snapshot count");
    return n;
}
}  // namespace

void writeKeyframeFeatures(
    const std::filesystem::path& path, const Keyframe& k,
    const RgbdAssociation& association, const std::filesystem::path& dataset_root,
    const Camera& camera, const OrbFeatureConfig& features,
    const OrbMatcherConfig& matching, const DepthConversionConfig& depth_config,
    std::size_t state_version) {
    const auto temporary = std::filesystem::path(path.string() + ".tmp");
    if (std::filesystem::exists(path) || std::filesystem::exists(temporary))
        throw std::runtime_error("Feature sidecar already exists");
    const auto& pixels = k.keypoints(); const auto& descriptors = k.descriptors();
    const auto& depth = k.depthImage(); const auto& ids = k.mapPointIds();
    if (pixels.size() > 100000 || ids.size() != pixels.size() ||
        depth.type() != CV_16UC1 || depth.empty() ||
        (!pixels.empty() && (descriptors.type() != CV_8UC1 ||
         descriptors.rows != static_cast<int>(pixels.size()) || descriptors.cols != 32)) ||
        (!descriptors.empty() && pixels.empty()) ||
        !std::isfinite(association.rgb_timestamp + association.depth_timestamp) ||
        association.rgb_timestamp != k.timestamp() || state_version != k.sourceFrameId() ||
        std::abs(association.rgb_timestamp-association.depth_timestamp) > .02 ||
        !std::isfinite(features.scale_factor + matching.ratio_threshold) ||
        features.max_features <= 0 || features.scale_factor <= 1 || features.pyramid_levels <= 0 ||
        features.fast_threshold < 0 || matching.ratio_threshold <= 0 || matching.ratio_threshold >= 1 ||
        matching.max_hamming_distance < 0 || matching.max_hamming_distance > 256 ||
        !std::isfinite(depth_config.depth_scale + depth_config.min_depth_meters + depth_config.max_depth_meters) ||
        depth_config.depth_scale <= 0 || depth_config.min_depth_meters <= 0 ||
        depth_config.max_depth_meters <= depth_config.min_depth_meters)
        throw std::runtime_error("Invalid native feature metadata");
    const auto relative = [&](const auto& image) {
        auto result = std::filesystem::relative(image, dataset_root);
        if (result.empty() || result.is_absolute()) throw std::runtime_error("Invalid feature image reference");
        for (const auto& part : result) if (part == "..")
            throw std::runtime_error("Feature image reference escapes dataset");
        return result.generic_string();
    };
    const auto rgb = relative(association.rgb_path), depth_path = relative(association.depth_path);
    for (const auto& p : pixels) {
        if (!std::isfinite(p.pt.x + p.pt.y + p.size + p.angle + p.response) ||
            p.pt.x < 0 || p.pt.y < 0 || p.pt.x >= depth.cols || p.pt.y >= depth.rows ||
            p.size <= 0 || p.angle < -1 || p.angle >= 360 || p.octave < 0 || p.octave >= features.pyramid_levels)
            throw std::runtime_error("Invalid native keypoint");
    }
    std::ofstream out(temporary);
    if (!out) throw std::runtime_error("Cannot create feature sidecar");
    out << std::setprecision(17) << "AFS_FEATURES 1\nMETA " << AFS_SOURCE_REVISION
        << ' ' << state_version << " after_insertion_before_ba\nKEYFRAME " << k.id()
        << ' ' << k.sourceFrameId() << ' ' << association.rgb_timestamp << ' '
        << association.depth_timestamp << ' ' << depth.cols << ' ' << depth.rows
        << "\nCAMERA " << camera.fx() << ' ' << camera.fy() << ' ' << camera.cx() << ' ' << camera.cy()
        << "\nDEPTH " << depth_config.depth_scale << ' ' << depth_config.min_depth_meters << ' ' << depth_config.max_depth_meters
        << "\nORB " << features.max_features << ' ' << features.scale_factor << ' ' << features.pyramid_levels << ' ' << features.fast_threshold
        << "\nMATCH " << matching.ratio_threshold << ' ' << matching.max_hamming_distance << ' ' << matching.require_mutual_consistency
        << "\nIMAGES " << std::quoted(rgb) << ' ' << std::quoted(depth_path)
        << "\nFEATURES " << pixels.size() << " 32\n";
    constexpr char hex[] = "0123456789abcdef";
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        const auto& p = pixels[i];
        const int x = static_cast<int>(std::lround(p.pt.x)), y = static_cast<int>(std::lround(p.pt.y));
        const unsigned raw = x < depth.cols && y < depth.rows ? depth.at<std::uint16_t>(y,x) : 0;
        out << "F " << i << ' ' << p.pt.x << ' ' << p.pt.y << ' ' << p.size << ' ' << p.angle << ' '
            << p.response << ' ' << p.octave << ' ' << p.class_id << ' ' << raw << ' '
            << (ids[i] ? std::to_string(*ids[i]) : "-1") << ' ';
        for (int j = 0; j < 32; ++j) {
            const auto byte = descriptors.at<std::uint8_t>(static_cast<int>(i),j);
            out << hex[byte >> 4] << hex[byte & 15];
        }
        out << '\n';
    }
    out << "END\n"; out.flush();
    if (!out) throw std::runtime_error("Feature sidecar write failed; temporary retained");
    out.close(); if (!out) throw std::runtime_error("Feature sidecar close failed");
    std::filesystem::rename(temporary,path);
}

void writeGeometrySnapshot(
    const std::filesystem::path& path, const Camera& camera,
    const LocalBundleAdjustmentConfig& config,
    const std::vector<Keyframe>& keyframes,
    const std::vector<MapPoint>& points, std::size_t state_version,
    const std::string& source_revision) {
    // Caller owns a newly-created output directory. Never truncate an old file.
    if (std::filesystem::exists(path)) throw std::runtime_error("Snapshot already exists");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot create geometry snapshot");
    out << std::setprecision(17) << "AFS_GEOMETRY 1\nMETA "
        << (source_revision.empty() ? AFS_SOURCE_REVISION : source_revision)
        << ' ' << state_version << "\nCAMERA " << camera.fx() << ' ' << camera.fy()
        << ' ' << camera.cx() << ' ' << camera.cy() << "\nCONFIG "
        << config.window_size << ' ' << config.minimum_point_observations << ' '
        << config.maximum_iterations << ' ' << config.huber_delta_pixels << ' '
        << config.depth_scale << ' ' << config.depth_residual_weight
        << "\nKEYFRAMES " << keyframes.size() << '\n';
    for (const auto& k : keyframes) {
        out << "K " << k.id() << ' ' << k.sourceFrameId() << ' ' << k.timestamp()
            << ' ' << k.depthImage().cols << ' ' << k.depthImage().rows;
        matrix(out, k.poseWorldFromCamera());
        out << '\n';
    }
    const std::size_t begin = keyframes.size() > config.window_size
        ? keyframes.size() - config.window_size : 0;
    std::vector<const MapPoint*> touched;
    std::size_t observations = 0;
    for (const auto& p : points) {
        bool active = false;
        for (const auto& o : p.observations()) active |= o.keyframe_id >= begin;
        if (active) { touched.push_back(&p); observations += p.observations().size(); }
    }
    out << "POINTS " << touched.size() << '\n';
    for (const auto* p : touched) {
        out << "P " << p->id() << ' ' << p->observations().front().keyframe_id;
        for (int i = 0; i < 3; ++i) out << ' ' << p->positionWorld()[i];
        out << '\n';
    }
    out << "OBSERVATIONS " << observations << '\n';
    for (const auto* p : touched) for (const auto& o : p->observations()) {
        const auto& k = keyframes.at(o.keyframe_id);
        const auto pixel = k.keypoints().at(o.feature_index).pt;
        const auto& depth = k.depthImage();
        const int x = static_cast<int>(std::lround(pixel.x));
        const int y = static_cast<int>(std::lround(pixel.y));
        unsigned raw = 0;
        if (depth.type() == CV_16UC1 && x >= 0 && y >= 0 && x < depth.cols && y < depth.rows)
            raw = depth.at<std::uint16_t>(y, x);
        out << "O " << p->id() << ' ' << o.keyframe_id << ' ' << o.feature_index
            << ' ' << pixel.x << ' ' << pixel.y << ' ' << raw << '\n';
    }
    out << "END\n";
    out.flush();
    if (!out) throw std::runtime_error("Geometry snapshot write failed");
}

GeometrySnapshot readGeometrySnapshot(const std::filesystem::path& path) {
    std::ifstream in(path);
    token(in, "AFS_GEOMETRY");
    if (count(in, 1) != 1) throw std::runtime_error("Unsupported snapshot version");
    GeometrySnapshot s;
    token(in, "META"); in >> s.source_revision >> s.state_version;
    token(in, "CAMERA");
    double fx, fy, cx, cy;
    if (!(in >> fx >> fy >> cx >> cy) || !std::isfinite(fx + fy + cx + cy))
        throw std::runtime_error("Invalid camera");
    s.camera = Camera(fx, fy, cx, cy);
    token(in, "CONFIG");
    in >> s.config.window_size >> s.config.minimum_point_observations
       >> s.config.maximum_iterations >> s.config.huber_delta_pixels
       >> s.config.depth_scale >> s.config.depth_residual_weight;
    LocalBundleAdjuster validate_config(s.camera, s.config);
    if (!std::isfinite(s.config.depth_scale + s.config.huber_delta_pixels +
                       s.config.depth_residual_weight)) throw std::runtime_error("Invalid config");
    token(in, "KEYFRAMES");
    const auto nk = count(in, 10000);
    std::vector<Frame> frames;
    std::size_t total_pixels = 0;
    double previous_timestamp = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < nk; ++i) {
        token(in, "K");
        std::size_t id, source; double timestamp; int width, height;
        in >> id >> source >> timestamp >> width >> height;
        if (!in || id != i || !std::isfinite(timestamp) || timestamp <= previous_timestamp ||
            (i && source <= frames.back().id) || width <= 0 || height <= 0 ||
            width > 4096 || height > 4096) throw std::runtime_error("Invalid keyframe metadata");
        previous_timestamp = timestamp;
        total_pixels += static_cast<std::size_t>(width) * height;
        if (total_pixels > 250000000) throw std::runtime_error("Snapshot image budget exceeded");
        Frame f; f.id = source; f.association.rgb_timestamp = timestamp;
        f.pose_world_from_camera = matrix(in); f.pose_valid = true;
        f.depth_image = cv::Mat(height, width, CV_16UC1, cv::Scalar(0));
        frames.push_back(std::move(f));
    }
    token(in, "POINTS");
    const auto np = count(in, 1000000);
    struct Point { std::size_t id, host; Eigen::Vector3d xyz; std::vector<MapObservation> obs; };
    std::vector<Point> points;
    std::unordered_map<std::size_t, std::size_t> lookup;
    for (std::size_t i = 0; i < np; ++i) {
        token(in, "P"); Point p;
        in >> p.id >> p.host >> p.xyz[0] >> p.xyz[1] >> p.xyz[2];
        if (!in || p.host >= nk || !p.xyz.allFinite() || !lookup.emplace(p.id, i).second)
            throw std::runtime_error("Invalid point");
        points.push_back(std::move(p));
    }
    token(in, "OBSERVATIONS");
    const auto no = count(in, 5000000);
    std::set<std::pair<std::size_t, std::size_t>> used_features, used_observations;
    std::map<std::tuple<std::size_t, int, int>, unsigned> sampled_depths;
    for (std::size_t i = 0; i < no; ++i) {
        token(in, "O"); std::size_t p, k, feature; double u, v; unsigned raw;
        in >> p >> k >> feature >> u >> v >> raw;
        if (!in || !lookup.count(p) || k >= nk || feature > 100000 || raw > 65535 ||
            !std::isfinite(u + v) || static_cast<double>(static_cast<float>(u)) != u ||
            static_cast<double>(static_cast<float>(v)) != v || !used_features.emplace(k, feature).second ||
            !used_observations.emplace(p, k).second) throw std::runtime_error("Invalid observation");
        auto& f = frames[k];
        if (u < 0 || v < 0 || u >= f.depth_image.cols || v >= f.depth_image.rows)
            throw std::runtime_error("Observation outside image");
        if (f.features.keypoints.size() <= feature) f.features.keypoints.resize(feature + 1);
        f.features.keypoints[feature] = cv::KeyPoint(static_cast<float>(u), static_cast<float>(v), 1);
        const int x = static_cast<int>(std::lround(u)), y = static_cast<int>(std::lround(v));
        const auto sample = sampled_depths.emplace(std::make_tuple(k,x,y),raw);
        if (!sample.second && sample.first->second != raw)
            throw std::runtime_error("Conflicting depth sample");
        if (x < f.depth_image.cols && y < f.depth_image.rows) {
            auto& value = f.depth_image.at<std::uint16_t>(y, x);
            if (value && value != raw) throw std::runtime_error("Conflicting depth sample");
            value = static_cast<std::uint16_t>(raw);
        } else if (raw) throw std::runtime_error("Depth sample outside image");
        points[lookup.at(p)].obs.push_back({k, feature});
    }
    token(in, "END"); std::string extra;
    if (in >> extra) throw std::runtime_error("Trailing snapshot data");
    for (std::size_t k = 0; k < nk; ++k) {
        auto& f = frames[k];
        f.features.descriptors = cv::Mat::zeros(static_cast<int>(f.features.keypoints.size()), 32, CV_8UC1);
        s.keyframes.emplace_back(k, f);
    }
    for (const auto& p : points) {
        if (p.obs.empty() || p.obs.front().keyframe_id != p.host)
            throw std::runtime_error("Missing host observation");
        s.points.emplace_back(p.id, p.xyz, cv::Mat::zeros(1, 32, CV_8UC1), p.obs.front());
        for (std::size_t j = 1; j < p.obs.size(); ++j) s.points.back().addObservation(p.obs[j]);
        for (const auto& o : p.obs) s.keyframes[o.keyframe_id].associateMapPoint(o.feature_index, p.id);
    }
    return s;
}

void writeFrameGeometryHeader(std::ostream& out) {
    out << "frame_id timestamp valid reference_id anchor_id";
    for (int i = 0; i < 16; ++i) out << " t" << i;
    out << '\n';
}
void writeFrameGeometryRow(std::ostream& out, const Frame& f,
                          const Frame* reference, const Keyframe* anchor) {
    out << std::setprecision(17) << f.id << ' ' << f.association.rgb_timestamp << ' '
        << f.pose_valid << ' ' << (reference ? std::to_string(reference->id) : "-1")
        << ' ' << (anchor ? std::to_string(anchor->id()) : "-1");
    matrix(out, f.pose_world_from_camera);
    out << '\n';
    if (!out) throw std::runtime_error("Frame geometry write failed");
}
}  // namespace adaptive_fusion_slam
