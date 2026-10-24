#!/usr/bin/env python3
"""Reuse the retained offline toolchain; no production build changes."""
import argparse,json,subprocess
from pathlib import Path
from adaptivefusion.state import file_sha256
def main():
 p=argparse.ArgumentParser()
 for k in ['source-root','reference-commands','reference-build','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise ValueError('refuse overwrite')
 a.output.mkdir();old=json.loads(a.reference_commands.read_text());compile_cmd=old[0]
 define=next(x for x in compile_cmd if x.startswith('-DAFS_REPLAY_ENTRY='))
 compile_cmd.remove(define);compile_cmd[compile_cmd.index('-c')+1]=str(a.source_root/'app/offline_keyframe_pose_graph.cpp');compile_cmd[compile_cmd.index('-o')+1]=str(a.output/'graph.o')
 link=[str(a.output/'graph.o') if x.endswith('/replay.o') else x for x in old[1]];link[link.index('-o')+1]=str(a.output/'offline_keyframe_pose_graph')
 (a.output/'commands.json').write_text(json.dumps([compile_cmd,link],indent=2))
 for cmd in [compile_cmd,link]:subprocess.run(cmd,check=True,cwd=a.reference_build)
 (a.output/'provenance.json').write_text(json.dumps({str(p):file_sha256(p) for p in [a.source_root/'app/offline_keyframe_pose_graph.cpp',a.output/'offline_keyframe_pose_graph']},indent=2))
if __name__=='__main__':main()
