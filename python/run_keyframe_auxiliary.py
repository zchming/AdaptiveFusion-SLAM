#!/usr/bin/env python3
"""Frozen offline keyframe edges + Ceres correction of retained LK trajectories."""
import argparse,csv,json,os,shlex,subprocess,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from adaptivefusion.geometry import inverse,check_pose
from adaptivefusion.trajectory import load_frames,write_tum
from adaptivefusion.state import file_sha256
from adaptivefusion.native_pnp import native_fit
from offline_matching_service import Matcher

def numbers(pose):
 return [*pose[:3,3],*Rotation.from_matrix(pose[:3,:3]).as_quat()]
def write_graph(path,poses,aux,config):
 edges=[(i-1,i,0,inverse(poses[i-1])@poses[i]) for i in range(1,len(poses))]
 edges += [(i,j,1,inverse(z)) for i,j,z in aux]
 with path.open('x') as f:
  f.write(f'AFS_KF_GRAPH 1\nNODES {len(poses)}\n')
  for i,p in enumerate(poses):f.write(f'V {i} '+' '.join(f'{x:.17g}' for x in numbers(p))+'\n')
  f.write(f'EDGES {len(edges)}\n')
  for i,j,kind,z in edges:
   values=[*numbers(z),config['translation_sigma_m'],np.deg2rad(config['rotation_sigma_deg'])]
   f.write(f'E {i} {j} {kind} '+' '.join(f'{x:.17g}' for x in values)+'\n')
  f.write('END\n')
def read_nodes(path):
 result=[]
 for line in path.read_text().splitlines():
  v=list(map(float,line.split()));assert int(v[0])==len(result)
  p=np.eye(4);p[:3,3]=v[1:4];p[:3,:3]=Rotation.from_quat(v[4:8]).as_matrix();result.append(check_pose(p))
 return np.array(result)
def correct(frames,keys,optimized):
 raw=np.array([frames[k['frame_id']].t_wc for k in keys]);times=np.array([k['timestamp'] for k in keys])
 corrections=optimized@np.linalg.inv(raw);corrections[0]=np.eye(4)
 rotations=Slerp(times,Rotation.from_matrix(corrections[:,:3,:3]))
 result=[]
 for f in frames:
  if not f.valid:continue
  if f.timestamp>=times[-1]:c=corrections[-1]
  elif f.timestamp<=times[0]:c=np.eye(4)
  else:
   i=int(np.searchsorted(times,f.timestamp,side='right'))-1;alpha=(f.timestamp-times[i])/(times[i+1]-times[i])
   c=np.eye(4);c[:3,:3]=rotations([f.timestamp]).as_matrix()[0];c[:3,3]=(1-alpha)*corrections[i][:3,3]+alpha*corrections[i+1][:3,3]
  result.append((f.timestamp,check_pose(c@f.t_wc)))
 return result

def solve(binary,path,output,env):
 start=time.perf_counter()
 r=subprocess.run([str(binary),str(path),str(output)],env=env,capture_output=True,text=True,timeout=120)
 output.parent.joinpath(output.name+'.log').write_text(r.stdout+r.stderr)
 report=json.loads((output/'report.json').read_text()) if (output/'report.json').exists() else {}
 report.update(returncode=r.returncode,process_wall_s=time.perf_counter()-start)
 return report

def main():
 p=argparse.ArgumentParser()
 for k in ['protocol','graph-binary','lightglue-source','weights','native-library-path','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();plan=json.loads(a.protocol.read_text())
 if a.output.exists():raise ValueError('refuse overwrite')
 for path,sha in plan['input_sha256'].items():
  if file_sha256(path)!=sha:raise ValueError('frozen input changed')
 a.output.mkdir();start=time.perf_counter();all_records=[];status=[]
 # Child native processes need retained shared libraries; do not change Python loader paths.
 env=os.environ.copy();prefix=str(a.native_library_path);env['LD_LIBRARY_PATH']=':'.join([prefix,prefix+'/blas',prefix+'/lapack',str(a.native_library_path.parent),'/usr/lib/wsl/lib',env.get('LD_LIBRARY_PATH','')])
 for scene,s in plan['scenes'].items():
  scene_start=time.perf_counter();dest=a.output/scene;dest.mkdir();frames=load_frames(Path(s['baseline'])/'cache/frames.txt');keys=s['keyframes'];poses=np.array([frames[k['frame_id']].t_wc for k in keys]);aux=[]
  write_graph(dest/'control.graph',poses,[],plan['graph'])
  control=solve(a.graph_binary,dest/'control.graph',dest/'control',env)
  if control.get('returncode')!=0 or not control.get('usable'):raise ValueError('baseline graph unusable')
  control_nodes=read_nodes(dest/'control/nodes.txt');drift=float(np.max(np.abs(control_nodes-poses)))
  if drift>1e-8:raise ValueError('baseline-only graph changed poses')
  control['maximum_pose_element_change']=drift
  write_tum(dest/'control.tum',correct(frames,keys,control_nodes))
  matcher=Matcher(SimpleNamespace(dataset=Path(s['dataset']),lightglue_source=a.lightglue_source,weights=a.weights,log=dest/'service.jsonl'))
  attempted=0;match_start=time.perf_counter()
  for req in s['requests']:
   i,j=req['source'],req['target'];src,dst=keys[i],keys[j];beg=time.perf_counter();record=dict(scene=scene,source_keyframe=i,target_keyframe=j,source_frame=src['frame_id'],target_frame=dst['frame_id'],accepted=False,reason=None)
   if attempted>=s['request_budget']:raise ValueError('request budget exhausted')
   attempted+=1;request_index=matcher.requests
   try:
    line=shlex.join(['magsac',str(src['frame_id']),str(dst['frame_id']),src['rgb'],src['depth'],dst['rgb']])
    matcher.request(line)
    with np.load(matcher.evidence/f'{request_index:06d}.npz') as evidence:rows=evidence['rows'].copy()
    service=json.loads((dest/'service.jsonl').read_text().splitlines()[-1]);record.update(service_status=service['status'],matches=service['matches'],F_retained=service['F_retained'],depth_rows=len(rows))
    if service['status']!='ok':record['reason']=service.get('failure') or service['status']
    else:
     old=os.environ.get('LD_LIBRARY_PATH');os.environ['LD_LIBRARY_PATH']=env['LD_LIBRARY_PATH']
     try:fit=native_fit(plan['native_binary'],rows[:,3:6],rows[:,6:8],plan['matcher']['camera'],rows[:,0].astype(np.int64),inverse(poses[j])@poses[i],17,dest/f'pair-{i}-{j}')
     finally:
      if old is None:os.environ.pop('LD_LIBRARY_PATH',None)
      else:os.environ['LD_LIBRARY_PATH']=old
     record.update(accepted=fit['applied'],reason=fit['reason'],native_returncode=fit['process_returncode'],inliers=len(fit.get('inlier_source_ids',[])),native_wall_s=fit['wall_s'],mean_reprojection_error_pixels=fit.get('mean_reprojection_error_pixels'))
     if fit['applied']:
      z=check_pose(np.array(fit['t_target_source']));aux.append((i,j,z));record['t_target_source']=z.tolist()
      used=np.isin(rows[:,0].astype(np.int64),fit['inlier_source_ids']);selected=rows[used]
      hull=matcher.cv2.convexHull(selected[:,6:8].astype(np.float32))
      record['target_hull_fraction']=float(matcher.cv2.contourArea(hull)/(640*480))
      singular=np.linalg.svd(selected[:,3:6]-selected[:,3:6].mean(0),compute_uv=False)
      record['source_depth_median_m']=float(np.median(selected[:,5]));record['source_3D_singular_ratios']=(singular/max(singular[0],1e-12)).tolist()
   except Exception as error:
    record.update(reason=str(error),traceback=traceback.format_exc())
    if matcher.requests==request_index:matcher.requests+=1
   record['wall_s']=time.perf_counter()-beg;all_records.append(record)
   with (dest/'constraints.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
   if attempted%25==0:print(scene,attempted,'/',s['request_budget'],flush=True)
   if time.perf_counter()-match_start>plan['budget']['per_scene_timeout_s']:raise TimeoutError('scene matching budget')
  matcher.log.close();write_graph(dest/'auxiliary.graph',poses,aux,plan['graph'])
  graph=solve(a.graph_binary,dest/'auxiliary.graph',dest/'auxiliary',env)
  if graph.get('usable') and graph.get('returncode')==0:
   optimized=read_nodes(dest/'auxiliary/nodes.txt');write_tum(dest/'corrected.tum',correct(frames,keys,optimized))
  record=dict(scene=scene,status='complete' if graph.get('usable') and graph.get('returncode')==0 else 'graph_failed',requests=attempted,accepted=len(aux),rejected=attempted-len(aux),control=control,graph=graph,model_load_s=matcher.load_s,total_wall_s=time.perf_counter()-scene_start,baseline_frames=len(frames),baseline_valid=sum(f.valid for f in frames))
  status.append(record);(a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(record),flush=True)
 for path,sha in plan['input_sha256'].items():
  if file_sha256(path)!=sha:raise ValueError('frozen input changed')
 (a.output/'constraint_summary.json').write_text(json.dumps(dict(records=all_records,wall_s=time.perf_counter()-start),indent=2)+'\n')
 seal={str(p.relative_to(a.output)):file_sha256(p) for p in a.output.rglob('*') if p.is_file()}
 (a.output/'fit-seal.json').write_text(json.dumps(seal,indent=2)+'\n')
if __name__=='__main__':main()
