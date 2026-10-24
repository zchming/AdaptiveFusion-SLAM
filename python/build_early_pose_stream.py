#!/usr/bin/env python3
"""Build an isolated early-publication observer, retaining the original backend."""
import argparse,difflib,hashlib,json,shlex,subprocess
from pathlib import Path
def main():
 p=argparse.ArgumentParser()
 for name in ["source-root","reference-build","sysroot","output"]:p.add_argument("--"+name,type=Path,required=True)
 a=p.parse_args();root=a.source_root.resolve();ref=a.reference_build.resolve();out=a.output.resolve();sysroot=a.sysroot.resolve()
 if out.exists():raise ValueError("refuse overwrite")
 out.mkdir();original=(root/"app/run_rgbd_odometry.cpp").read_text()
 wait="        for (std::size_t index = 0; index < frame_count; ++index) {"
 early="            if (insert_keyframe) {"
 finish='            std::cerr << "Frame " << index << ": "'
 if any(original.count(x)!=1 for x in [wait,early,finish]):raise ValueError("source anchors changed")
 changed=original.replace(wait,wait+"\n            afs_live::writer->wait(dataset.associations()[index]);")
 changed=changed.replace(early,
  "            afs_live::writer->publish(result.frame,insert_keyframe,\n"
  "                insert_keyframe ? static_cast<long long>(sparse_map.keyframes().size()) : -1,\n"
  "                frame_start,std::chrono::steady_clock::now());\n"+early)
 changed=changed.replace(finish,"            afs_live::writer->finish(index,frame_end);\n"+finish)
 generated=out/"early_runner.inc";generated.write_text(changed)
 (out/"publication-only.diff").write_text("".join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile="unchanged_production",tofile="early_publication")))
 includes=["-I"+str(root/"include"),"-I"+str(root/"app"),"-isystem",str(sysroot/"include"),"-isystem",str(sysroot/"include/eigen3"),"-isystem",str(sysroot/"include/opencv4")]
 compile_cmd=["/usr/bin/c++","-O3","-DNDEBUG","-std=gnu++17","-DGFLAGS_IS_A_DLL=0","-DGLOG_CUSTOM_PREFIX_SUPPORT",*includes,
  '-DAFS_LIVE_ENTRY="'+str(generated)+'"',"-c",str(root/"app/run_early_pose_odometry.cpp"),"-o",str(out/"early.o")]
 link=shlex.split((ref/"CMakeFiles/run_rgbd_odometry.dir/link.txt").read_text())
 link=[str(out/"early.o") if x.endswith("app/run_rgbd_odometry.cpp.o") else x for x in link]
 link[link.index("-o")+1]=str(out/"run_early_pose_odometry")
 link.insert(3,"-Wl,-rpath-link,"+str(sysroot/"lib/x86_64-linux-gnu/blas")+",-rpath-link,"+str(sysroot/"lib/x86_64-linux-gnu/lapack"))
 (out/"commands.json").write_text(json.dumps([compile_cmd,link],indent=2))
 for command in [compile_cmd,link]:subprocess.run(command,check=True,cwd=ref)
 files=[root/"app/run_rgbd_odometry.cpp",root/"app/run_early_pose_odometry.cpp",ref/"libadaptivefusion_slam.a",generated,out/"run_early_pose_odometry"]
 (out/"provenance.json").write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
if __name__=="__main__":main()
