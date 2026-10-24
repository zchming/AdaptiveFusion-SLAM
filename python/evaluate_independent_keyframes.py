#!/usr/bin/env python3
"""Evaluate sealed independent-sequence outputs, including lost/failed arms."""
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
 for k in ['study','datasets','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise ValueError('refuse overwrite')
 seal=json.loads((a.study/'fit-seal.json').read_text())
 def check():
  for path,sha in seal.items():
   if file_sha256(a.study/path)!=sha:raise ValueError('sealed output changed')
 check();datasets=json.loads(a.datasets.read_text())['records'];status=json.loads((a.study/'status.json').read_text());stage_status={r['stage']:r for r in status};metrics=[];errors=[];failures=[];cost=[];hashes={}
 aux_status={r['scene']:r for r in json.loads((a.study/'auxiliary/status.json').read_text())} if (a.study/'auxiliary/status.json').exists() else {}
 for d in datasets:
  scene=d['scene'];root=a.study/'runs'/scene
  metadata=root/'baseline/cache/frames.txt'
  if not metadata.exists():metadata=root/'magsac/cache/frames.txt'
  if not metadata.exists():failures.append(dict(scene=scene,method='all',reason='no frame inventory'));continue
  frames=load_frames(metadata);n=len(frames);frame_times={f.id:f.timestamp for f in frames};tt,truth=read_tum(Path(d['dataset'])/'groundtruth.txt');hashes[scene]=file_sha256(Path(d['dataset'])/'groundtruth.txt');association=associate(tt,np.array([f.timestamp for f in frames]),.02);ground={i:truth[j] for j,i in association};gts={i:tt[j] for j,i in association}
  descriptions=[('baseline',root/'baseline/trajectory.tum',root/'baseline/cache/frames.txt',stage_status.get(scene+'-baseline',{}).get('status')=='complete'),('magsac_every_frame',root/'magsac/trajectory.tum',root/'magsac/cache/frames.txt',stage_status.get(scene+'-magsac',{}).get('status')=='complete'),('keyframe_auxiliary',a.study/'auxiliary'/scene/'corrected.tum',root/'baseline/cache/frames.txt',aux_status.get(scene,{}).get('status')=='complete'),('keyframe_control',a.study/'auxiliary'/scene/'control.tum',root/'baseline/cache/frames.txt',(a.study/'auxiliary'/scene/'control.tum').exists())]
  methods={}
  for name,path,framefile,complete in descriptions:
   if not complete or not path.exists():
    failures.append(dict(scene=scene,method=name,reason='incomplete or missing arm; retained as failure'));continue
   try:
    ff=load_frames(framefile);t,e=read_tum(path);valid=[f for f in ff if f.valid]
    if len(ff)!=n or len(valid)!=len(t) or not np.allclose(t,[f.timestamp for f in valid],rtol=0,atol=1e-6):raise ValueError('frame/trajectory inventory mismatch')
    methods[name]=dict(poses={f.id:p for f,p in zip(valid,e)},valid=[f.valid for f in ff])
   except (ValueError,OSError) as error:failures.append(dict(scene=scene,method=name,reason=str(error)))
  common=set(ground)
  for m in methods.values():common &= set(m['poses'])
  common_valid=[all(m['valid'][i] for m in methods.values()) for i in range(n)]
  for name,m in methods.items():
   longest=0;run=0
   for valid in m['valid']:run=0 if valid else run+1;longest=max(longest,run)
   for scope,ids,vv in [('own',sorted(set(m['poses'])&set(ground)),m['valid']),('common',sorted(common),common_valid)]:
    value,perframe=metric(ids,m['poses'],ground,gts,frame_times,vv);anchor={}
    if len(ids)>=3:
     e=np.array([m['poses'][i] for i in ids]);g=np.array([ground[i] for i in ids]);delta=np.linalg.inv(g)@((g[0]@np.linalg.inv(e[0]))@e);trans=np.linalg.norm(delta[:,:3,3],axis=1)*100;rot=np.rad2deg(Rotation.from_matrix(delta[:,:3,:3]).magnitude())
     anchor=dict(anchor_RMSE_cm=float(np.sqrt(np.mean(trans**2))),endpoint_cm=float(trans[-1]),endpoint_rotation_deg=float(rot[-1]),first_evaluated_frame=ids[0],last_evaluated_frame=ids[-1])
     errors.extend(dict(scene=scene,method=name,scope=scope,anchor_translation_cm=float(x),anchor_rotation_deg=float(y),**record) for record,x,y in zip(perframe,trans,rot))
    metrics.append(dict(scene=scene,method=name,scope=scope,input_frames=n,valid_frames=sum(m['valid']),lost_frames=n-sum(m['valid']),longest_lost_frames=longest,coverage=sum(m['valid'])/n,common_methods=len(methods),**value,**anchor))
  entry=dict(scene=scene,input_frames=n,baseline_process_s=stage_status.get(scene+'-baseline',{}).get('seconds'),every_frame_process_s=stage_status.get(scene+'-magsac',{}).get('seconds'),every_frame_including_startup_s=stage_status.get(scene+'-service',{}).get('total_seconds_including_startup'),auxiliary=aux_status.get(scene))
  service=a.study/(scene+'-service.jsonl')
  if service.exists():
   ss=[json.loads(x) for x in service.read_text().splitlines()];calls=[x for x in ss if 'frame_id' in x];entry.update(every_frame_calls=len(calls),every_frame_F_failures=sum(x.get('status')=='F_failed' for x in ss),every_frame_service_errors=sum(x.get('status')=='service_error' for x in ss),every_frame_service_s=sum(x.get('wall_s',0) for x in ss),every_frame_peak_GPU_allocated_MiB=max([x.get('peak_allocated_bytes',0) for x in ss],default=0)/1024**2)
  aux_service=a.study/'auxiliary'/scene/'service.jsonl'
  if aux_service.exists():
   ss=[json.loads(x) for x in aux_service.read_text().splitlines()];entry.update(auxiliary_calls=len(ss),auxiliary_F_failures=sum(x.get('status')=='F_failed' for x in ss),auxiliary_service_s=sum(x.get('wall_s',0) for x in ss),auxiliary_peak_GPU_allocated_MiB=max([x.get('peak_allocated_bytes',0) for x in ss],default=0)/1024**2)
  front=root/'magsac/frontend.csv'
  if front.exists():
   rr=list(csv.DictReader(front.open()));entry['every_frame_native_failures']=sum(x['success']=='0' for x in rr)
  cost.append(entry)
 a.output.mkdir()
 payload=dict(records=metrics,failures=failures,cost=cost,truth_sha256=hashes,fit_seal_sha256=file_sha256(a.study/'fit-seal.json'),scope='two new independent recording sequences, same Freiburg1 camera family; no independent-sensor or cross-dataset claim',timing='fresh instrumented offline runs; baseline includes keyframe metadata exports; serial baseline+aux cost is sum of measured stage components, not online FPS')
 (a.output/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
 for filename,rows in [('summary.csv',metrics),('frame_errors.csv',errors)]:
  if rows:
   with (a.output/filename).open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
 check();print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
