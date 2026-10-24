#!/usr/bin/env python3
"""Freeze baseline keyframe IDs, exact images and a one-call-per-edge budget, no GT."""
import argparse,json,shlex,subprocess
from pathlib import Path
from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames

def main():
 p=argparse.ArgumentParser()
 for k in ['baseline-runs','sources','native-binary','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise ValueError('refuse overwrite')
 sources=json.loads(a.sources.read_text());records={};hashes={}
 def digest(path):hashes[str(Path(path).resolve())]=file_sha256(path)
 for path in [a.sources,a.native_binary,Path(__file__)]:digest(path)
 for scene,s in sources.items():
  if set(s)!={'frames','features','dataset'}:raise ValueError('inference sources only')
  root=a.baseline_runs/scene/'baseline';frames=load_frames(root/'cache/frames.txt')
  snapshot=root/'cache/final.snapshot';keys=[]
  for line in snapshot.read_text().splitlines():
   if line.startswith('K '):
    v=line.split();keys.append(dict(id=int(v[1]),frame_id=int(v[2]),timestamp=float(v[3])))
  meta={}
  for path in Path(s['features']).glob('*.features'):
   with path.open() as f:lines=[next(f) for _ in range(8)]
   k=shlex.split(lines[2]);im=shlex.split(lines[7])
   meta[int(k[1])]=dict(frame_id=int(k[2]),timestamp=float(k[3]),depth_timestamp=float(k[4]),rgb=str((Path(s['dataset'])/im[1]).resolve()),depth=str((Path(s['dataset'])/im[2]).resolve()))
  for k in keys:
   m=meta[k['id']]
   if (m['frame_id'],m['timestamp'])!=(k['frame_id'],k['timestamp']):raise ValueError('keyframe inventory mismatch')
   if not frames[k['frame_id']].valid:raise ValueError('keyframe invalid')
   k.update(m)
   for path in [m['rgb'],m['depth']]:digest(path)
  for path in [snapshot,root/'cache/frames.txt',root/'trajectory.tum']:digest(path)
  records[scene]=dict(dataset=s['dataset'],baseline=str(root),keyframes=keys,requests=[dict(source=keys[i-1]['id'],target=k['id']) for i,k in enumerate(keys) if i],request_budget=max(0,len(keys)-1),input_frames=len(frames),valid_frames=sum(f.valid for f in frames))
 plan=dict(stage='V31 offline keyframe auxiliary pose graph',status='frozen_before_keyframe_matching',truth_access=False,scenes=records,input_sha256=hashes,native_binary=str(a.native_binary.resolve()),baseline_runs=str(a.baseline_runs.resolve()),matcher=dict(implementation='unchanged offline_matching_service.Matcher',RootSIFT_features=2048,F_threshold_px=3.,F_confidence=.999,F_max_iterations=10000,seed=17,camera=[517.3,516.5,318.6,255.3],depth='unchanged source-only3x3,spread<=0.05m,scale5000'),budget=dict(total_calls=sum(r['request_budget'] for r in records.values()),retries=0,history_search=0,per_scene_timeout_s=600),constraint_acceptance='unchanged native PnP success min6; F/service/native failures omit auxiliary edge, retain LK edge; record support diagnostics without posthoc gates',graph=dict(nodes='baseline keyframes, raw sequential frame poses, never optimized snapshot poses',baseline_edges='one composite LK relative edge between adjacent keyframes',auxiliary_edges='one native PnP relative edge per successful frozen pair',direction='nodes T_wc; measurement T_source_target=inverse(native T_target_source)',translation_sigma_m=.05,rotation_sigma_deg=5.,baseline_loss='linear',auxiliary_loss='Huber(delta=1) on whitened six-dimensional block',information='equal fixed diagonal baseline/aux scale, engineering prior not calibrated covariance',gauge='first keyframe fixed',maximum_iterations=50,num_threads=1,writeback='left world correction per keyframe, SLERP rotation and linear translation by time, constant correction after last keyframe, no lost-frame filling'),evaluation='after all outputs sealed; same V30 canonical one-to-oneGT .02s, SE3 no-scaleATE,1sRPE excludingLost; additional first-poseanchored drift',selection_scope='same two previously examined sequences; engineering verification, not independent generalization',parameter_sweep=False,online_integration=False,commit_push=False)
 a.output.mkdir();(a.output/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
 print(json.dumps({s:dict(keyframes=len(v['keyframes']),requests=v['request_budget']) for s,v in records.items()}))
if __name__=='__main__':main()
