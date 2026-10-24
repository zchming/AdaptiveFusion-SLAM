#!/usr/bin/env python3
"""Independent four-arm evaluation, after grand seal, retaining failed arms."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames
from adaptivefusion.evaluation import read_tum,associate
from evaluate_full_replay import metric
def main():
 p=argparse.ArgumentParser()
 for k in ['runs','datasets','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise ValueError('refuse overwrite')
 seal=json.loads((a.runs/'grand-fit-seal.json').read_text())
 def check():
  for p,sha in seal.items():
   if file_sha256(a.runs/p)!=sha:raise ValueError('sealed fit changed')
 check();data=json.loads(a.datasets.read_text())['records'];study=a.runs/'study';stage={x['stage']:x for x in json.loads((study/'status.json').read_text())};aux={x['scene']:x for x in json.loads((study/'auxiliary/status.json').read_text())} if (study/'auxiliary/status.json').exists() else {};confidence={x['scene']:x for x in json.loads((a.runs/'confidence/status.json').read_text())} if (a.runs/'confidence/status.json').exists() else {};records=[];errors=[];failures=[];cost=[];GT_hashes={}
 for d in data:
  scene=d['scene'];root=study/'runs'/scene;inventory=root/'baseline/cache/frames.txt'
  if not inventory.exists():inventory=root/'magsac/cache/frames.txt'
  if not inventory.exists():failures.append(dict(scene=scene,method='all',reason='missing frame inventory'));continue
  frames=load_frames(inventory);n=len(frames);times={f.id:f.timestamp for f in frames};truth=Path(d['dataset'])/'groundtruth.txt';tt,g=read_tum(truth);GT_hashes[scene]=file_sha256(truth);pairs=associate(tt,np.array([f.timestamp for f in frames]),.02);ground={i:g[j] for j,i in pairs};gts={i:tt[j] for j,i in pairs}
  descriptions=[('baseline',root/'baseline/trajectory.tum',root/'baseline/cache/frames.txt',stage.get(scene+'-baseline',{}).get('status')=='complete'),('fixed',study/'auxiliary'/scene/'corrected.tum',root/'baseline/cache/frames.txt',aux.get(scene,{}).get('status')=='complete'),('confidence',a.runs/'confidence'/scene/'confidence_quadratic.tum',root/'baseline/cache/frames.txt',confidence.get(scene,{}).get('status')=='complete'),('magsac_every_frame',root/'magsac/trajectory.tum',root/'magsac/cache/frames.txt',stage.get(scene+'-magsac',{}).get('status')=='complete')]
  methods={}
  for name,path,framefile,complete in descriptions:
   if not complete or not path.exists():failures.append(dict(scene=scene,method=name,reason='incomplete or missing trajectory'));continue
   ff=load_frames(framefile);ts,e=read_tum(path);valid=[f for f in ff if f.valid]
   if len(ff)!=n or len(ts)!=len(valid) or not np.allclose(ts,[f.timestamp for f in valid],rtol=0,atol=1e-6):raise ValueError('frame inventory changed')
   methods[name]=dict(poses={f.id:p for f,p in zip(valid,e)},valid=[f.valid for f in ff])
  common=set(ground)
  for m in methods.values():common &= set(m['poses'])
  vv=[all(m['valid'][i] for m in methods.values()) for i in range(n)]
  for name,m in methods.items():
   longest=0;run=0
   for valid in m['valid']:run=0 if valid else run+1;longest=max(longest,run)
   for scope,ids,v in [('own',sorted(set(m['poses'])&set(ground)),m['valid']),('common',sorted(common),vv)]:
    value,perframe=metric(ids,m['poses'],ground,gts,times,v);extra={}
    if len(ids)>=3:
     e=np.array([m['poses'][i] for i in ids]);gg=np.array([ground[i] for i in ids]);delta=np.linalg.inv(gg)@((gg[0]@np.linalg.inv(e[0]))@e);trans=np.linalg.norm(delta[:,:3,3],axis=1)*100;angle=np.rad2deg(Rotation.from_matrix(delta[:,:3,:3]).magnitude());extra=dict(anchor_RMSE_cm=float(np.sqrt(np.mean(trans**2))),endpoint_cm=float(trans[-1]),endpoint_rotation_deg=float(angle[-1]),first_evaluated_frame=ids[0],last_evaluated_frame=ids[-1]);errors.extend(dict(scene=scene,method=name,scope=scope,anchor_translation_cm=float(x),anchor_rotation_deg=float(y),**r) for r,x,y in zip(perframe,trans,angle))
    records.append(dict(scene=scene,method=name,scope=scope,input_frames=n,valid_frames=sum(m['valid']),lost_frames=n-sum(m['valid']),longest_lost_frames=longest,coverage=sum(m['valid'])/n,common_methods=len(methods),**value,**extra))
  entry=dict(scene=scene,input_frames=n,baseline=stage.get(scene+'-baseline'),every_frame=stage.get(scene+'-magsac'),every_frame_service=stage.get(scene+'-service'),fixed=aux.get(scene),confidence=confidence.get(scene))
  for kind,path in [('every_frame',study/(scene+'-service.jsonl')),('keyframe',study/'auxiliary'/scene/'service.jsonl')]:
   if path.exists():
    rr=[json.loads(line) for line in path.read_text().splitlines()];entry[kind+'_calls']=len(rr);entry[kind+'_F_failures']=sum(x.get('status')=='F_failed' for x in rr);entry[kind+'_service_errors']=sum(x.get('status')=='service_error' for x in rr);entry[kind+'_components_s']={k:sum(x.get(k,0) for x in rr) for k in ['extraction_s','matching_s','F_s','wall_s']};entry[kind+'_GPU_allocated_MiB']=max([x.get('peak_allocated_bytes',0) for x in rr],default=0)/1024**2;entry[kind+'_GPU_reserved_MiB']=max([x.get('peak_reserved_bytes',0) for x in rr],default=0)/1024**2
  front=root/'magsac/frontend.csv'
  if front.exists():entry['every_frame_native_failures']=sum(x['success']=='0' for x in csv.DictReader(front.open()))
  for method,path in [('baseline',root/'baseline/trajectory.tum.health.csv'),('every_frame',root/'magsac/trajectory.tum.health.csv')]:
   if path.exists():entry[method+'_ORB_fallback_calls']=sum(x['used_orb_fallback']=='1' for x in csv.DictReader(path.open()))
  cost.append(entry)
 a.output.mkdir();payload=dict(records=records,failures=failures,cost=cost,GT_sha256=GT_hashes,grand_seal_sha256=file_sha256(a.runs/'grand-fit-seal.json'),scope='new desk2 recording, same Freiburg1 camera and office as earlier desk; no new-scene or cross-sensor claim',parameters_changed=False,test_time_tuning=False)
 (a.output/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
 for name,rows in [('summary.csv',records),('frame_errors.csv',errors)]:
  if rows:
   with (a.output/name).open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
 check();print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
