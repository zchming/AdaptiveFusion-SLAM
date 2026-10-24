// One-way observer around the unchanged production entry; no pose feedback API.
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/socket.h>
#include <unistd.h>
#include "frame.h"

namespace afs_live {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t) {
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}
class Writer {
public:
    Writer(int descriptor, const std::filesystem::path& stats) : fd(descriptor) {
        if (fd < 0 || std::filesystem::exists(stats))
            throw std::runtime_error("invalid stream FD or existing stats output");
        output.open(stats);
        if (!output) throw std::runtime_error("cannot create stream stats");
        output << "frame_id,due_monotonic_s,start_monotonic_s,published_monotonic_s,"
                  "tracking_compute_ms,transport_write_us,stream_sent,send_errno\n";
    }
    void wait(const adaptive_fusion_slam::RgbdAssociation& a) {
        if (!started) {
            epoch = Clock::now(); first_timestamp = a.rgb_timestamp; started = true;
        }
        due = epoch + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(a.rgb_timestamp-first_timestamp+.02));
        std::this_thread::sleep_until(due);
    }
    void publish(const adaptive_fusion_slam::Frame& frame, bool keyframe,
                 long long keyframe_id, Clock::time_point start,
                 Clock::time_point end, double compute_ms) {
        const auto before = Clock::now();
        std::ostringstream s; s << std::setprecision(17);
        s << "{\"schema\":1,\"frame_id\":" << frame.id
          << ",\"timestamp\":" << frame.association.rgb_timestamp
          << ",\"depth_timestamp\":" << frame.association.depth_timestamp
          << ",\"valid\":" << (frame.pose_valid?"true":"false")
          << ",\"is_keyframe\":" << (keyframe?"true":"false")
          << ",\"keyframe_id\":" << keyframe_id
          << ",\"due_monotonic_s\":" << seconds(due)
          << ",\"start_monotonic_s\":" << seconds(start)
          << ",\"published_monotonic_s\":" << seconds(before)
          << ",\"tracking_compute_ms\":" << compute_ms << ",\"t_wc\":[";
        for (int r=0;r<4;++r) {
            if (r) s << ','; s << '[';
            for (int c=0;c<4;++c) {
                if (c) s << ','; s << frame.pose_world_from_camera.matrix()(r,c);
            }
            s << ']';
        }
        s << "]}\n"; const auto message=s.str();
        const auto result=::send(fd,message.data(),message.size(),MSG_DONTWAIT|MSG_NOSIGNAL);
        const auto error=result==static_cast<ssize_t>(message.size())?0:errno;
        const auto after=Clock::now();
        output << std::setprecision(17) << frame.id << ',' << seconds(due)
               << ',' << seconds(start) << ',' << seconds(before) << ',' << compute_ms
               << ',' << std::chrono::duration<double,std::micro>(after-before).count()
               << ',' << (result==static_cast<ssize_t>(message.size())) << ',' << error << '\n';
        output.flush();
    }
private:
    int fd;
    bool started=false;
    double first_timestamp=0.;
    Clock::time_point epoch,due;
    std::ofstream output;
};
std::unique_ptr<Writer> writer;
}
#ifndef AFS_LIVE_ENTRY
#error "Build with explicit AFS_LIVE_ENTRY pointing to the observed original runner"
#endif
#define main afs_production_main
#include AFS_LIVE_ENTRY
#undef main

int main(int argc,char** argv) {
    try {
        if (argc<8 || std::string(argv[1])!="--pose-fd" ||
            std::string(argv[3])!="--stream-stats" || std::string(argv[5])!="--")
            throw std::runtime_error("usage: run_streaming_odometry --pose-fd FD --stream-stats NEW.csv -- BASELINE_ARGS");
        afs_live::writer=std::make_unique<afs_live::Writer>(std::stoi(argv[2]),argv[4]);
        std::vector<char*> original; original.push_back(argv[0]);
        for (int i=6;i<argc;++i) original.push_back(argv[i]);
        const auto result=afs_production_main(static_cast<int>(original.size()),original.data());
        afs_live::writer.reset(); return result;
    } catch(const std::exception& e) {
        std::cerr << "Streamed odometry failed: " << e.what() << '\n'; return 1;
    }
}
