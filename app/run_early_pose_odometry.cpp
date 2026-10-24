// Publishes the immutable frontend pose before unchanged mapping/BA work.
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t) {return std::chrono::duration<double>(t.time_since_epoch()).count();}
class Writer {
public:
 Writer(int descriptor,const std::filesystem::path& stats):fd(descriptor) {
  if(fd<0 || std::filesystem::exists(stats))throw std::runtime_error("invalid FD/existing stats");
  output.open(stats);if(!output)throw std::runtime_error("cannot create stream stats");
  output<<"frame_id,due_monotonic_s,start_monotonic_s,published_monotonic_s,tracking_compute_ms,"
          "transport_write_us,stream_sent,send_errno,pose_ready_compute_ms,backend_after_publish_ms\n";
 }
 void wait(const adaptive_fusion_slam::RgbdAssociation& a) {
  if(!started){epoch=Clock::now();first=a.rgb_timestamp;started=true;}
  due=epoch+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(a.rgb_timestamp-first+.02));
  std::this_thread::sleep_until(due);
 }
 void publish(const adaptive_fusion_slam::Frame& frame,bool keyframe,long long key_id,
              Clock::time_point start,Clock::time_point ready) {
  pending.id=frame.id;pending.start=start;pending.ready=ready;pending.published=Clock::now();
  pending.front_ms=std::chrono::duration<double,std::milli>(ready-start).count();
  std::ostringstream s;s<<std::setprecision(17);
  s<<"{\"schema\":1,\"frame_id\":"<<frame.id<<",\"timestamp\":"<<frame.association.rgb_timestamp
   <<",\"depth_timestamp\":"<<frame.association.depth_timestamp
   <<",\"valid\":"<<(frame.pose_valid?"true":"false")
   <<",\"is_keyframe\":"<<(keyframe?"true":"false")<<",\"keyframe_id\":"<<key_id
   <<",\"due_monotonic_s\":"<<seconds(due)<<",\"start_monotonic_s\":"<<seconds(start)
   <<",\"published_monotonic_s\":"<<seconds(pending.published)
   <<",\"publication_phase\":\"before_mapping\",\"pose_ready_compute_ms\":"<<pending.front_ms<<",\"t_wc\":[";
  for(int r=0;r<4;++r){if(r)s<<',';s<<'[';for(int c=0;c<4;++c){if(c)s<<',';s<<frame.pose_world_from_camera.matrix()(r,c);}s<<']';}
  s<<"]}\n";const auto text=s.str();
  auto sent=::send(fd,text.data(),text.size(),MSG_DONTWAIT|MSG_NOSIGNAL);
  pending.sent=sent==static_cast<ssize_t>(text.size());pending.error=pending.sent?0:errno;
  pending.write_us=std::chrono::duration<double,std::micro>(Clock::now()-pending.published).count();
 }
 void finish(std::size_t frame_id,Clock::time_point end) {
  if(frame_id!=pending.id)throw std::runtime_error("publication/completion order mismatch");
  const auto total=std::chrono::duration<double,std::milli>(end-pending.start).count();
  const auto backend=std::chrono::duration<double,std::milli>(end-pending.published).count();
  output<<std::setprecision(17)<<frame_id<<','<<seconds(due)<<','<<seconds(pending.start)<<','
        <<seconds(pending.published)<<','<<total<<','<<pending.write_us<<','<<pending.sent<<','
        <<pending.error<<','<<pending.front_ms<<','<<backend<<'\n';output.flush();
 }
private:
 int fd;bool started=false;double first=0.;Clock::time_point epoch,due;std::ofstream output;
 struct Pending {std::size_t id=0;Clock::time_point start,ready,published;double front_ms=0.,write_us=0.;bool sent=false;int error=0;} pending;
};
std::unique_ptr<Writer> writer;
}
#ifndef AFS_LIVE_ENTRY
#error "explicit observed production entry is required"
#endif
#define main afs_production_main
#include AFS_LIVE_ENTRY
#undef main
int main(int argc,char** argv) {
 try {
  if(argc<8 || std::string(argv[1])!="--pose-fd" || std::string(argv[3])!="--stream-stats" || std::string(argv[5])!="--")
   throw std::runtime_error("usage: early_pose_odometry --pose-fd FD --stream-stats NEW.csv -- BASELINE_ARGS");
  afs_live::writer=std::make_unique<afs_live::Writer>(std::stoi(argv[2]),argv[4]);
  std::vector<char*> original{argv[0]};for(int i=6;i<argc;++i)original.push_back(argv[i]);
  const int result=afs_production_main(static_cast<int>(original.size()),original.data());
  afs_live::writer.reset();return result;
 }catch(const std::exception& e){std::cerr<<"Early publication failed: "<<e.what()<<'\n';return 1;}
}
