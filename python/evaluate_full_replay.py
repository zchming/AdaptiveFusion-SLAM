#!/usr/bin/env python3
"""Evaluate sealed complete replay trajectories; inference never imports this module."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from adaptivefusion.evaluation import read_tum,associate,rigid_alignment,fixed_interval_pairs,rpe
from adaptivefusion.trajectory import load_frames
from adaptivefusion.state import file_sha256


def metric(ids,poses,ground,gt_times,frame_times,valid):
 if len(ids)<3:return dict(status='insufficient_GT',matched=len(ids)),[]
 e=np.asarray([poses[i] for i in ids]);g=np.asarray([ground[i] for i in ids]);ts=np.asarray([frame_times[i] for i in ids]);gts=np.asarray([gt_times[i] for i in ids])
 alignment=rigid_alignment(e[:,:3,3],g[:,:3,3]);aligned=e[:,:3,3]@alignment[:3,:3].T+alignment[:3,3];errors=np.linalg.norm(aligned-g[:,:3,3],axis=1)
 pairs=fixed_interval_pairs(ts,1.,.02)
 pairs=[(i,j) for i,j in pairs if abs(gts[j]-gts[i]-1.)<=.02 and all(valid[k] for k in range(ids[i],ids[j]+1))]
 motion=rpe(g,e,pairs)
 result=dict(status='evaluated',matched=len(ids),ATE_RMSE_cm=float(np.sqrt(np.mean(errors**2))*100),ATE_median_cm=float(np.median(errors)*100),ATE_P95_cm=float(np.quantile(errors,.95)*100),ATE_max_cm=float(errors.max()*100),RPE_pairs=motion['pairs'],RPE_translation_RMSE_cm=motion['translation_rmse_m']*100 if motion['pairs'] else None,RPE_rotation_RMSE_deg=float(np.rad2deg(motion['rotation_rmse_rad'])) if motion['pairs'] else None)
 return result,[dict(frame_id=i,timestamp=frame_times[i],ATE_cm=float(error*100)) for i,error in zip(ids,errors)]


def main():
 p=argparse.ArgumentParser()
 for k in ['runs','sources','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise ValueError('refuse overwrite')
 seal=json.loads((a.runs/'fit-seal.json').read_text())
 for rel,sha in seal.items():
  path=(a.runs/rel).resolve()
  if not path.is_relative_to(a.runs.resolve()) or file_sha256(path)!=sha:raise ValueError('fit seal changed')
 sources=json.loads(a.sources.read_text());records=[];errors=[];failures=[];truth_hashes={};status={(r['scene'],r['mode']):r for r in json.loads((a.runs/'status.json').read_text())}
 for scene,source in sources.items():
  try:times,truth=read_tum(source['truth']);truth_hashes[scene]=file_sha256(Path(source['truth']))
  except (OSError,ValueError,KeyError) as error:failures.append(dict(scene=scene,error=str(error)));continue
  base_frames=load_frames(a.runs/scene/'baseline/cache/frames.txt');frame_times={f.id:f.timestamp for f in base_frames};all_ts=np.array([f.timestamp for f in base_frames]);association=associate(times,all_ts,.02);ground={fid:truth[gi] for gi,fid in association};gt_times={fid:times[gi] for gi,fid in association};methods={}
  for mode in ['baseline','lightglue','magsac']:
   record=status.get((scene,mode),{})
   try:
    if record.get('status')!='complete':raise ValueError('replay did not complete')
    root=a.runs/scene/mode;frames=load_frames(root/'cache/frames.txt');ts,poses=read_tum(root/'trajectory.tum');vf=[f for f in frames if f.valid]
    if len(vf)!=len(ts) or len(frames)!=len(base_frames) or not np.allclose(ts,[f.timestamp for f in vf],rtol=0,atol=1e-6):raise ValueError('trajectory/frame inventory mismatch')
    methods[mode]=dict(poses={f.id:pose for f,pose in zip(vf,poses)},valid=[f.valid for f in frames],frames=frames)
   except (ValueError,OSError) as error:failures.append(dict(scene=scene,mode=mode,error=str(error)))
  common=set(ground)
  for m in methods.values():common &= set(m['poses'])
  common_valid=[all(m['valid'][i] for m in methods.values()) for i in range(len(base_frames))]
  for mode,m in methods.items():
   root=a.runs/scene/mode;frame_cost=list(csv.DictReader((root/'trajectory.tum.frame_times.csv').open()));cost=np.asarray([float(r['frame_ms']) for r in frame_cost]);lost=0;longest=0
   for valid in m['valid']:lost=0 if valid else lost+1;longest=max(longest,lost)
   for scope,ids,valid in [('own',sorted(set(m['poses'])&set(ground)),m['valid']),('common',sorted(common),common_valid)]:
    value,per_frame=metric(ids,m['poses'],ground,gt_times,frame_times,valid)
    record=dict(scene=scene,method=mode,scope=scope,common_methods=len(methods),input_frames=len(base_frames),valid_frames=sum(m['valid']),lost_frames=sum(not v for v in m['valid']),longest_lost_frames=longest,tracking_coverage=sum(m['valid'])/len(base_frames),frame_median_ms=float(np.median(cost)),frame_P95_ms=float(np.quantile(cost,.95)),process_wall_s=status[scene,mode]['seconds'],total_wall_including_startup_s=status[scene,mode]['total_seconds_including_startup'],**value)
    records.append(record);errors.extend(dict(scene=scene,method=mode,scope=scope,**r) for r in per_frame)
 a.output.mkdir();(a.output/'summary.json').write_text(json.dumps(dict(records=records,failures=failures,truth_sha256=truth_hashes,alignment='singleSE3 per trajectory/selection, no scale',association='canonical one-to-one against all baseline input timestamps,0.02s',RPE='1s +/-0.02s; sensor andGT interval checks; no Lost crossing',partial_common=any(r['common_methods']<3 for r in records),seal_sha256=file_sha256(a.runs/'fit-seal.json')),indent=2)+'\n')
 for filename,rows in [('summary.csv',records),('frame_errors.csv',errors)]:
  if rows:
   fields=list(dict.fromkeys(k for row in rows for k in row))
   with (a.output/filename).open('x',newline='') as stream:w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
 for rel,sha in seal.items():
  if file_sha256(a.runs/rel)!=sha:raise ValueError('fit changed during evaluation')
 print(json.dumps(dict(records=records,failures=failures),indent=2))


if __name__=='__main__':main()
