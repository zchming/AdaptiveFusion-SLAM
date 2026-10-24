#include <chrono>
#include <cmath>
#include <filesystem>
#include "offline_replay_odometry.h"

// Compile the original runner body with one experimental odometry substitution.
// The generated body adds CSV timing only; original source and backend are untouched.
#define RgbdOdometry OfflineReplayOdometry
#define main originalRunnerMain
#include AFS_REPLAY_ENTRY
#undef main
#undef RgbdOdometry

int main(int argc,char** argv) {
    if(argc!=5) {
        std::cerr << "Usage: run_offline_matching_replay DATASET NEW_OUTPUT baseline|lightglue|magsac SOCKET_OR_DASH\n";
        return 1;
    }
    const std::string mode=argv[3];
    if(mode!="baseline" && mode!="lightglue" && mode!="magsac") return 1;
    const std::filesystem::path output=argv[2];
    if(!std::filesystem::create_directory(output)) {
        std::cerr << "Refuse existing output\n";return 1;
    }
    setenv("AFS_OFFLINE_MODE",mode.c_str(),1);setenv("AFS_OFFLINE_SOCKET",argv[4],1);
    const auto log=(output/"frontend.csv").string();setenv("AFS_OFFLINE_FRONTEND_LOG",log.c_str(),1);
    std::vector<std::string> args={argv[0],argv[1],(output/"trajectory.tum").string(),"1000000","-","none",(output/"cache").string()};
    std::vector<char*> pointers;for(auto& arg:args)pointers.push_back(arg.data());
    return originalRunnerMain(static_cast<int>(pointers.size()),pointers.data());
}
