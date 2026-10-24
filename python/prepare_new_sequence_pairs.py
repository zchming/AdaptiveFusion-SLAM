#!/usr/bin/env python3
"""Freeze unused-sequence pair inputs after unmodified baseline replay, before GT."""
import argparse,json,shlex
from pathlib import Path
import numpy as np
from adaptivefusion.trajectory import load_frames
from adaptivefusion.geometry import inverse
from adaptivefusion.state import file_sha256
from run_camera_pnp_comparison import write_json

def main():
 p=argparse.ArgumentParser()
 for k in ['sources','native-binary','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--pairs-per-sequence',type=int,default=24);a=p.parse_args()
 if a.output.exists():raise ValueError('output exists')
 sources=json.loads(a.sources.read_text());hashes={str(x.resolve()):file_sha256(x) for x in [a.sources,a.native_binary,Path(__file__)]};cases=[];inventories=[];rgb_hashes=set()
 for scene,src in sources.items():
  if set(src)!={'frames','features','dataset'}:raise ValueError('inference-only sources required')
  frames=load_frames(src['frames']);meta={};hashes[str(Path(src['frames']).resolve())]=file_sha256(Path(src['frames']))
  for fp in Path(src['features']).glob('*.features'):
   with fp.open() as stream:
    lines=[next(stream) for _ in range(8)]
   k=shlex.split(lines[2]);ims=shlex.split(lines[7]);kid=int(k[1]);fid=int(k[2]);assert frames[fid].timestamp==float(k[3]);meta[kid]=(fid,float(k[3]),fp,Path(src['dataset'])/ims[1])
  pool=[];occupied=set()
  for i in sorted(meta):
   j=i+1
   if j not in meta or i in occupied or j in occupied:continue
   if not frames[meta[i][0]].valid or not frames[meta[j][0]].valid:continue
   pool.append([i,j]);occupied.update([i,j])
  count=min(a.pairs_per_sequence,len(pool));indices=np.linspace(0,len(pool)-1,count,dtype=int).tolist() if count else []
  inventories.append(dict(scene=scene,all_input_frames=len(frames),valid_frames=sum(f.valid for f in frames),lost_frames=sum(not f.valid for f in frames),exported_keyframes=len(meta),candidate_pairs=pool,selected_indices=indices,selected_count=count))
  for index in indices:
   i,j=pool[index];f,g=[frames[meta[k][0]] for k in [i,j]]
   for k in [i,j]:
    for path in meta[k][2:]:hashes[str(path.resolve())]=file_sha256(path)
    rgb_hashes.add(file_sha256(meta[k][3]))
   cases.append(dict(scene=scene,pair=f'{i}-{j}',source_timestamp=f.timestamp,target_timestamp=g.timestamp,raw_baseline=(inverse(g.t_wc)@f.t_wc).tolist()))
 a.output.mkdir();write_json(a.output/'selection-protocol.json',dict(status='frozen_before_candidate_matching_and_GT',inventories=inventories,cases=len(cases),unique_RGB_hashes=len(rgb_hashes),scope='conditional on exported baseline keyframes; baseline lost frames retained separately; no recovery claim',input_sha256=hashes));write_json(a.output/'fit-only.json',dict(stage='pair_manifest_only_no_fitting',cases=cases,input_sha256=hashes,truth_read=False));write_json(a.output/'fit-seal.json',{n:file_sha256(a.output/n) for n in ['fit-only.json','selection-protocol.json']});print(json.dumps(inventories,indent=2))
if __name__=='__main__':main()
