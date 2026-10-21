#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

#include "local_bundle_adjuster.h"

namespace adaptive_fusion_slam {

// Offline interchange only. No reference-frame or live-map writeback API.
struct GeometrySnapshot {
    Camera camera{1, 1, 0, 0};
    LocalBundleAdjustmentConfig config;
    std::string source_revision;
    std::size_t state_version = 0;
    std::vector<Keyframe> keyframes;
    std::vector<MapPoint> points;
};

struct OrbFeatureConfig;
struct OrbMatcherConfig;
struct DepthConversionConfig;
// Full original descriptor rows; offline sidecar, never a tracking writeback.
void writeKeyframeFeatures(
    const std::filesystem::path& path, const Keyframe& keyframe,
    const RgbdAssociation& association, const std::filesystem::path& dataset_root,
    const Camera& camera, const OrbFeatureConfig& features,
    const OrbMatcherConfig& matching, const DepthConversionConfig& depth,
    std::size_t state_version);

void writeGeometrySnapshot(
    const std::filesystem::path& path, const Camera& camera,
    const LocalBundleAdjustmentConfig& config,
    const std::vector<Keyframe>& keyframes,
    const std::vector<MapPoint>& points, std::size_t state_version,
    const std::string& source_revision = "");
GeometrySnapshot readGeometrySnapshot(const std::filesystem::path& path);
void writeFrameGeometryHeader(std::ostream& output);
void writeFrameGeometryRow(std::ostream& output, const Frame& frame,
                          const Frame* reference, const Keyframe* anchor);

}  // namespace adaptive_fusion_slam
