#!/usr/bin/env python3
"""Build an isolated graph loss-mode extension against the retained toolchain."""
import argparse,json,subprocess
from pathlib import Path
from adaptivefusion.state import file_sha256
def main():
 p=argparse.ArgumentParser()
 for k in ['source','reference-commands','reference-build','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise ValueError('refuse overwrite')
 a.output.mkdir();old=json.loads(a.reference_commands.read_text());compile_cmd=old[0].copy()
 compile_cmd[compile_cmd.index('-c')+1]=str(a.source);compile_cmd[compile_cmd.index('-o')+1]=str(a.output/'confidence.o')
 link=[str(a.output/'confidence.o') if x.endswith('/graph.o') else x for x in old[1]];link[link.index('-o')+1]=str(a.output/'offline_confidence_pose_graph')
 (a.output/'commands.json').write_text(json.dumps([compile_cmd,link],indent=2))
 for command in [compile_cmd,link]:subprocess.run(command,check=True,cwd=a.reference_build)
 (a.output/'provenance.json').write_text(json.dumps({str(p):file_sha256(p) for p in [a.source,a.output/'offline_confidence_pose_graph']},indent=2))
if __name__=='__main__':main()
