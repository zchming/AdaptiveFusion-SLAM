#!/usr/bin/env python3
"""Evaluate sealed offline keyframe corrections against all retained V30 arms."""
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
 for k in ['protocol','runs','truth-sources','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise ValueError('refuse overwrite')
 seal=json.loads((a.runs/'fit-seal.json').read_text())
 for path,sha in seal.items():
  if file_sha256(a.runs/path)!=sha:raise ValueError('fit changed')
 plan=json.loads(a.protocol.read_text());sources=json.loads(a.truth_sources.read_text());records=[];errors=[];failures=[]
 for scene,s in plan['scenes'].items():
  frames=load_frames(Path(s['baseline'])/'cache/frames.txt');frame_times={f.id:f.timestamp for f in frames}
  tt,truth=read_tum(sources[scene]['truth']);association=associate(tt,np.array([f.timestamp for f in frames]),.02)
  ground={i:truth[j] for j,i in association};gts={i:tt[j] for j,i in association}
  paths={m:Path(plan['baseline_runs'])/scene/m/'trajectory.tum' for m in ['baseline','lightglue','magsac']}
  paths.update(keyframe_control=a.runs/scene/'control.tum',keyframe_auxiliary=a.runs/scene/'corrected.tum')
  methods={}
  for name,path in paths.items():
   if not path.exists():failures.append(dict(scene=scene,method=name,error='trajectory missing'));continue
   ts,poses=read_tum(path);by_ts={round(f.timestamp,6):f.id for f in frames};pp={by_ts[round(float(t),6)]:v for t,v in zip(ts,poses)}
   if len(pp)!=len(ts):raise ValueError('timestamp inventory duplicate')
   methods[name]=dict(poses=pp,valid=[f.valid and f.id in pp for f in frames])
  common=set(ground)
  for m in methods.values():common &= set(m['poses'])
  common_valid=[all(m['valid'][i] for m in methods.values()) for i in range(len(frames))]
  for name,m in methods.items():
   ids=sorted(common);value,perframe=metric(ids,m['poses'],ground,gts,frame_times,common_valid)
   e=np.array([m['poses'][i] for i in ids]);g=np.array([ground[i] for i in ids]);aligned=(g[0]@np.linalg.inv(e[0]))@e;delta=np.linalg.inv(g)@aligned
   translation=np.linalg.norm(delta[:,:3,3],axis=1)*100;angle=np.rad2deg(Rotation.from_matrix(delta[:,:3,:3]).magnitude())
   records.append(dict(scene=scene,method=name,scope='common',input_frames=len(frames),valid_frames=sum(m['valid']),lost_frames=len(frames)-sum(m['valid']),coverage=sum(m['valid'])/len(frames),anchor_translation_RMSE_cm=float(np.sqrt(np.mean(translation**2))),endpoint_translation_cm=float(translation[-1]),endpoint_rotation_deg=float(angle[-1]),**value))
   errors.extend(dict(scene=scene,method=name,anchor_translation_cm=float(x),anchor_rotation_deg=float(y),**r) for r,x,y in zip(perframe,translation,angle))
 a.output.mkdir()
 (a.output/'summary.json').write_text(json.dumps(dict(records=records,failures=failures,common_methods=5,input_seal_sha256=file_sha256(a.runs/'fit-seal.json'),truth_sha256={s:file_sha256(v['truth']) for s,v in sources.items()},same_previously_evaluated_sequences=True),indent=2)+'\n')
 for file,rows in [('summary.csv',records),('per_frame.csv',errors)]:
  with (a.output/file).open('x',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for path,sha in seal.items():
  if file_sha256(a.runs/path)!=sha:raise ValueError('fit changed')
 print(json.dumps(records,indent=2))
if __name__=='__main__':main()
