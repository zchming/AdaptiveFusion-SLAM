#!/usr/bin/env python3
"""Evaluate sealed confidence-fusion trajectories; no fitting or weight changes."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from adaptivefusion.evaluation import read_tum,associate
from adaptivefusion.trajectory import load_frames
from adaptivefusion.state import file_sha256
from evaluate_full_replay import metric
def main():
 p=argparse.ArgumentParser()
 for k in ['sources','runs','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise ValueError('refuse overwrite')
 seal=json.loads((a.runs/'fit-seal.json').read_text())
 def check():
  for path,sha in seal.items():
   if file_sha256(a.runs/path)!=sha:raise ValueError('sealed fit changed')
 check();sources=json.loads(a.sources.read_text());status=json.loads((a.runs/'status.json').read_text());records=[];errors=[];failures=[];GT_hashes={}
 for src in sources:
  scene=src['scene'];m=json.loads(Path(src['manifest']).read_text())['scenes'][scene];frames=load_frames(Path(m['baseline'])/'cache/frames.txt');n=len(frames);times={f.id:f.timestamp for f in frames};tt,g=read_tum(Path(m['dataset'])/'groundtruth.txt');GT_hashes[scene]=file_sha256(Path(m['dataset'])/'groundtruth.txt');matches=associate(tt,np.array([f.timestamp for f in frames]),.02);ground={i:g[j] for j,i in matches};gts={i:tt[j] for j,i in matches}
  paths=dict(baseline=Path(m['baseline'])/'trajectory.tum',fixed=Path(src['evidence'])/'corrected.tum')
  for variant in ['legacy_fixed_replay','equal_quadratic','confidence_quadratic']:
   state=next((x for x in status if x['scene']==scene and x['variant']==variant),None)
   if state and state['status']=='complete':paths[variant]=a.runs/scene/(variant+'.tum')
   else:failures.append(dict(scene=scene,variant=variant,reason='fit failed',fit_status=state))
  methods={}
  for name,path in paths.items():
   if not path.exists():failures.append(dict(scene=scene,variant=name,reason='trajectory missing'));continue
   ts,e=read_tum(path);valid=[f for f in frames if f.valid]
   if len(ts)!=len(valid) or not np.allclose(ts,[f.timestamp for f in valid],rtol=0,atol=1e-6):raise ValueError('frame inventory changed')
   methods[name]={f.id:pose for f,pose in zip(valid,e)}
  common=set(ground)
  for poses in methods.values():common &= set(poses)
  for name,poses in methods.items():
   for scope,ids in [('own',sorted(set(poses)&set(ground))),('common',sorted(common))]:
    value,perframe=metric(ids,poses,ground,gts,times,[f.valid for f in frames]);extra={}
    if len(ids)>=3:
     e=np.array([poses[i] for i in ids]);gg=np.array([ground[i] for i in ids]);delta=np.linalg.inv(gg)@((gg[0]@np.linalg.inv(e[0]))@e);trans=np.linalg.norm(delta[:,:3,3],axis=1)*100;rot=np.rad2deg(Rotation.from_matrix(delta[:,:3,:3]).magnitude())
     extra=dict(anchor_RMSE_cm=float(np.sqrt(np.mean(trans**2))),endpoint_cm=float(trans[-1]),endpoint_rotation_deg=float(rot[-1]))
     errors.extend(dict(scene=scene,split=src['split'],method=name,scope=scope,anchor_translation_cm=float(x),**r) for r,x in zip(perframe,trans))
    records.append(dict(scene=scene,split=src['split'],method=name,scope=scope,input_frames=n,valid_frames=len(poses),lost_frames=n-len(poses),coverage=len(poses)/n,common_methods=len(methods),**value,**extra))
 a.output.mkdir()
 result=dict(records=records,failures=failures,fit_status=status,GT_sha256=GT_hashes,seal_sha256=file_sha256(a.runs/'fit-seal.json'),scope='Development RPY/360; reused floor/plant validation, not a new blind test; no GT rule selection',new_matching_calls=0,new_PnP_calls=0)
 (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 for name,rows in [('summary.csv',records),('frame_errors.csv',errors)]:
  if rows:
   with (a.output/name).open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
 check();print(json.dumps(result,indent=2))
if __name__=='__main__':main()
