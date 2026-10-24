#pragma once
// Experimental adapter only. Production RgbdOdometry and all map/BA code are unchanged.
#include "rgbd_odometry.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace adaptive_fusion_slam {
class OfflineReplayOdometry {
public:
    OfflineReplayOdometry(Camera camera, RgbdOdometryConfig config = {})
        : camera_(camera), baseline_(camera, config), extractor_(config.orb_features),
          estimator_(camera, config.pnp), health_(config.geometric_health) {
        mode_ = std::getenv("AFS_OFFLINE_MODE");
        if (mode_ == "baseline") return;
        const char* path = std::getenv("AFS_OFFLINE_SOCKET");
        if (!path || std::strlen(path) >= sizeof(sockaddr_un::sun_path))
            throw std::runtime_error("invalid offline socket path");
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        sockaddr_un address{}; address.sun_family = AF_UNIX;
        std::strcpy(address.sun_path, path);
        timeval timeout{60,0}; setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
        if (fd < 0 || connect(fd,reinterpret_cast<sockaddr*>(&address),sizeof(address)) != 0)
            throw std::runtime_error("cannot connect offline matching service");
        channel_ = fdopen(fd,"r+");
        if (!channel_) throw std::runtime_error("cannot open offline channel");
        trace_.open(std::getenv("AFS_OFFLINE_FRONTEND_LOG"));
        trace_ << "frame_id,reference_id,mode,rows,success,inliers,pnp_ms,frontend_ms,service_status,inlier_indices\n";
    }
    ~OfflineReplayOdometry() { if (channel_) std::fclose(channel_); }
    const Frame* referenceFrame() const {
        return mode_ == "baseline" ? baseline_.referenceFrame() : (reference_ ? &*reference_ : nullptr);
    }
    OdometryResult process(const RgbdFrame& rgbd) { return process(rgbd,RiskAdaptiveDecision{}); }
    OdometryResult process(const RgbdFrame& rgbd, const RiskAdaptiveDecision& decision) {
        if (mode_ == "baseline") return baseline_.process(rgbd,decision);
        const auto begin = std::chrono::steady_clock::now();
        OdometryResult result;
        result.frame.id=next_id_++; result.frame.association=rgbd.association;
        result.frame.rgb_image=rgbd.rgb_image; result.frame.depth_image=rgbd.depth_image;
        // Mapping/BA keep their original ORB rows. SIFT is used only for motion estimation.
        result.frame.features=extractor_.extract(rgbd.rgb_image);
        if (!reference_) {
            result.frame.pose_world_from_camera=Eigen::Isometry3d::Identity();
            result.frame.pose_valid=true; result.status=TrackingStatus::Initialized;
            result.health.tracking_success=true; reference_=result.frame;
            result.health.frame_id=result.frame.id;
            result.health.timestamp=rgbd.association.rgb_timestamp;
            result.health.feature_count=result.frame.features.keypoints.size();
            trace_ << result.frame.id << ",-1," << mode_ << ",0,1,0,0,0,initialized,\n";
            return result;
        }
        const auto reference_id=reference_->id;
        std::ostringstream request;
        request << mode_ << ' ' << reference_id << ' ' << result.frame.id << ' '
                << std::quoted(reference_->association.rgb_path.string()) << ' '
                << std::quoted(reference_->association.depth_path.string()) << ' '
                << std::quoted(rgbd.association.rgb_path.string()) << '\n';
        if (std::fputs(request.str().c_str(),channel_)<0 || std::fflush(channel_)!=0)
            throw std::runtime_error("offline request failed");
        std::istringstream header(readLine());
        std::string tag,status; std::size_t count=0;
        if (!(header>>tag>>count>>status) || tag!="ROWS" || count>100000)
            throw std::runtime_error("invalid offline response");
        std::vector<RgbdCorrespondence> points;
        std::vector<PixelCorrespondence> pixels;
        for (std::size_t i=0;i<count;++i) {
            std::istringstream line(readLine());
            std::size_t id; double sx,sy,x,y,z,u,v;
            if (!(line>>id>>sx>>sy>>x>>y>>z>>u>>v) || !std::isfinite(x+y+z+u+v) || z<=0)
                throw std::runtime_error("invalid offline correspondence");
            points.push_back({id,{x,y,z},{u,v},z});
            pixels.push_back({id,{static_cast<float>(sx),static_cast<float>(sy)},
                                   {static_cast<float>(u),static_cast<float>(v)}});
        }
        result.rgbd_correspondences=points.size();
        const auto pnp_begin=std::chrono::steady_clock::now();
        result.relative_pose=estimator_.estimate(points);
        const auto pnp_end=std::chrono::steady_clock::now();
        result.health=health_.compute(rgbd.rgb_image.size(),pixels,points.size(),{},result.relative_pose);
        result.health.frame_id=result.frame.id;
        result.health.timestamp=rgbd.association.rgb_timestamp;
        result.health.feature_count=result.frame.features.keypoints.size();
        if (result.relative_pose.success) {
            Eigen::Isometry3d current_from_reference=Eigen::Isometry3d::Identity();
            current_from_reference.linear()=result.relative_pose.rotation_current_from_previous;
            current_from_reference.translation()=result.relative_pose.translation_current_from_previous;
            result.frame.pose_world_from_camera=reference_->pose_world_from_camera*current_from_reference.inverse();
            result.frame.pose_valid=true;result.status=TrackingStatus::Tracked;
            reference_=result.frame;
        } else result.status=TrackingStatus::Lost;
        // Failed frames do not advance the trusted reference, matching original semantics.
        const auto end=std::chrono::steady_clock::now();
        trace_ << std::setprecision(17) << result.frame.id << ',' << reference_id << ',' << mode_ << ','
            << points.size() << ',' << result.relative_pose.success << ','
            << result.relative_pose.inlier_indices.size() << ','
            << std::chrono::duration<double,std::milli>(pnp_end-pnp_begin).count() << ','
            << std::chrono::duration<double,std::milli>(end-begin).count() << ',' << status << ',';
        for(auto index:result.relative_pose.inlier_indices) trace_ << index << ';';
        trace_ << '\n';
        trace_.flush();return result;
    }
private:
    std::string readLine() {
        char* buffer=nullptr;std::size_t capacity=0;
        const auto length=::getline(&buffer,&capacity,channel_);
        if(length<0) {std::free(buffer);throw std::runtime_error("offline service closed or timed out");}
        std::string value(buffer,length);std::free(buffer);return value;
    }
    Camera camera_;
    RgbdOdometry baseline_;
    OrbFeatureExtractor extractor_;
    PnpPoseEstimator estimator_;
    GeometricHealthMonitor health_;
    std::optional<Frame> reference_;
    std::size_t next_id_=0;
    std::string mode_;
    FILE* channel_=nullptr;
    std::ofstream trace_;
};
}
