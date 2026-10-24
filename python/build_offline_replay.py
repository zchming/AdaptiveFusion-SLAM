#!/usr/bin/env python3
"""Build an isolated replay adapter against the retained unmodified native library."""
import argparse,hashlib,json,shlex,subprocess,difflib
from pathlib import Path

def main():
 p=argparse.ArgumentParser()
 for k in ['source-root','reference-build','sysroot','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();root=a.source_root.resolve();ref=a.reference_build.resolve();out=a.output.resolve()
 if out.exists():raise ValueError('refuse overwrite')
 out.mkdir();text=(root/'app/run_rgbd_odometry.cpp').read_text()
 first='        for (std::size_t index = 0; index < frame_count; ++index) {'
 second='            std::cerr << "Frame " << index << ": "'
 assert text.count(first)==1 and text.count(second)==1
 changed=text.replace(first,'        std::ofstream offline_times(std::string(argv[2])+".frame_times.csv");\n        offline_times << "frame_id,frame_ms\\n";\n'+first).replace(second,'            offline_times << index << "," << frame_times_ms.back() << "\\n";\n'+second)
 generated=out/'runner.inc';generated.write_text(changed)
 (out/'timing-only.diff').write_text(''.join(difflib.unified_diff(text.splitlines(True),changed.splitlines(True),fromfile='original',tofile='timing-instrumented')))
 sysroot=a.sysroot.resolve()
 includes=['-I'+str(root/'include'),'-I'+str(root/'app'),'-isystem',str(sysroot/'include'),'-isystem',str(sysroot/'include/eigen3'),'-isystem',str(sysroot/'include/opencv4')]
 compile_cmd=['/usr/bin/c++','-O3','-DNDEBUG','-std=gnu++17','-DGFLAGS_IS_A_DLL=0','-DGLOG_CUSTOM_PREFIX_SUPPORT',*includes,'-DAFS_REPLAY_ENTRY="'+str(generated)+'"','-c',str(root/'app/run_offline_matching_replay.cpp'),'-o',str(out/'replay.o')]
 link=shlex.split((ref/'CMakeFiles/run_rgbd_odometry.dir/link.txt').read_text());link[link.index('-o')+1]=str(out/'run_offline_matching_replay');link=[str(out/'replay.o') if x.endswith('app/run_rgbd_odometry.cpp.o') else x for x in link]
 link.insert(3,'-Wl,-rpath-link,'+str(sysroot/'lib/x86_64-linux-gnu/blas')+',-rpath-link,'+str(sysroot/'lib/x86_64-linux-gnu/lapack'))
 (out/'commands.json').write_text(json.dumps([compile_cmd,link],indent=2))
 subprocess.run(compile_cmd,check=True,cwd=ref);subprocess.run(link,check=True,cwd=ref)
 paths=[root/'app/run_rgbd_odometry.cpp',root/'app/run_offline_matching_replay.cpp',root/'app/offline_replay_odometry.h',ref/'libadaptivefusion_slam.a',out/'run_offline_matching_replay']
 (out/'provenance.json').write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},indent=2))
if __name__=='__main__':main()
