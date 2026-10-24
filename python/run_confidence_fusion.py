#!/usr/bin/env python3
"""Freeze observable confidences, fit all variants, then seal; never reads GT."""
import argparse,json,os,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
from adaptivefusion.edge_confidence import confidence_pair
from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames,write_tum
from adaptivefusion.evaluation import read_tum
from run_keyframe_auxiliary import read_nodes,correct

def main():
 p=argparse.ArgumentParser()
 for k in ['protocol','sources','binary','native-library-path','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();plan=json.loads(a.protocol.read_text());sources=json.loads(a.sources.read_text())
 if a.output.exists():raise ValueError('refuse overwrite')
 if file_sha256(Path(__file__).parent/'adaptivefusion/edge_confidence.py')!=plan['confidence_source_sha256']:raise ValueError('frozen confidence code changed')
 a.output.mkdir();frozen={str(a.protocol.resolve()):file_sha256(a.protocol),str(a.sources.resolve()):file_sha256(a.sources),str(a.binary.resolve()):file_sha256(a.binary)};status=[];begin=time.perf_counter();env=os.environ.copy();lib=str(a.native_library_path);env['LD_LIBRARY_PATH']=':'.join([lib,lib+'/blas',lib+'/lapack',str(a.native_library_path.parent),env.get('LD_LIBRARY_PATH','')])
 def digest(path):frozen[str(path.resolve())]=file_sha256(path)
 for s in sources:
  manifest=Path(s['manifest']);digest(manifest);m=json.loads(manifest.read_text())['scenes'][s['scene']];root=Path(s['evidence']);frames=load_frames(Path(m['baseline'])/'cache/frames.txt');keys=m['keyframes'];dest=a.output/s['scene'];dest.mkdir()
  for path in [root/'constraints.jsonl',root/'auxiliary.graph',root/'corrected.tum',Path(m['baseline'])/'cache/frames.txt',Path(m['baseline'])/'trajectory.tum']:digest(path)
  weights={};observations=[]
  for index,line in enumerate((root/'constraints.jsonl').read_text().splitlines()):
   c=json.loads(line);i,j=c['source_keyframe'],c['target_keyframe'];entry=dict(scene=s['scene'],split=s['split'],source_keyframe=i,target_keyframe=j,source_frame=c['source_frame'],target_frame=c['target_frame'],native_accepted=c['accepted'])
   try:
    if not c['accepted']:entry.update(status='native_rejected',baseline=dict(weight=1.),auxiliary=dict(weight=plan['config']['minimum_information_weight']))
    else:
     npz=root/'service.matches'/f'{index:06d}.npz';depth=Path(keys[j]['depth']);digest(npz);digest(depth)
     with np.load(npz) as f:rows=f['rows'].copy()
     score=confidence_pair(np.linalg.inv(frames[c['target_frame']].t_wc)@frames[c['source_frame']].t_wc,np.array(c['t_target_source']),rows,depth,plan['config']);entry.update(score)
   except Exception as error:entry.update(status='confidence_error',reason=str(error),traceback=traceback.format_exc(),baseline=dict(weight=1.),auxiliary=dict(weight=plan['config']['minimum_information_weight']))
   weights[i,j]=entry;observations.append(entry)
  # Save every confidence before running any graph for this scene.
  (dest/'confidence-before-fit.json').write_text(json.dumps(observations,indent=2,allow_nan=False)+'\n')
  original=(root/'auxiliary.graph').read_text()
  for variant in plan['variants']:
   path=dest/(variant+'.graph')
   if variant=='confidence_quadratic':
    lines=[]
    for line in original.splitlines():
     tokens=line.split()
     if tokens and tokens[0]=='E':
      i,j,aux=map(int,tokens[1:4]);confidence=weights.get((i,j));w=confidence['auxiliary' if aux else 'baseline']['weight'] if confidence else 1.
      if not 0<w<=1:raise ValueError('invalid confidence')
      tokens[-2]=format(float(tokens[-2])/np.sqrt(w),'.17g');tokens[-1]=format(float(tokens[-1])/np.sqrt(w),'.17g');line=' '.join(tokens)
     lines.append(line)
    path.write_text('\n'.join(lines)+'\n')
   else:path.write_text(original)
   mode='legacy' if variant=='legacy_fixed_replay' else 'quadratic';output=dest/variant;start=time.perf_counter()
   record=dict(scene=s['scene'],split=s['split'],variant=variant,loss_mode=mode,status='process_failed')
   try:
    command=[str(a.binary),str(path),str(output),mode]
    run=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120);(dest/(variant+'.log')).write_text(run.stdout+run.stderr);record.update(command=command,returncode=run.returncode)
    if run.returncode==0:
     native=json.loads((output/'report.json').read_text());record['solver']=native
     if not native['usable']:raise ValueError('unusable graph')
     poses=correct(frames,keys,read_nodes(output/'nodes.txt'));write_tum(dest/(variant+'.tum'),poses);record.update(status='complete',valid_frames=len(poses))
     if variant=='legacy_fixed_replay':
      times,old=read_tum(root/'corrected.tum');actual=np.array([p for _,p in poses]);difference=float(np.max(np.abs(old-actual)));record['fixed_pose_max_element_difference']=difference
      if difference>1e-7:raise ValueError('legacy fixed replay parity failed')
   except Exception as error:record.update(status='failed',reason=str(error),traceback=traceback.format_exc())
   record['wall_s']=time.perf_counter()-start;status.append(record);(a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(record),flush=True)
 (a.output/'inputs.json').write_text(json.dumps(frozen,indent=2)+'\n')
 for path,sha in frozen.items():
  if file_sha256(path)!=sha:raise ValueError('input changed')
 (a.output/'wall.json').write_text(json.dumps(dict(seconds=time.perf_counter()-begin,new_matcher_calls=0,new_native_PnP_calls=0,GT_read=False),indent=2))
 seal={str(p.relative_to(a.output)):file_sha256(p) for p in a.output.rglob('*') if p.is_file()}
 (a.output/'fit-seal.json').write_text(json.dumps(seal,indent=2)+'\n')
if __name__=='__main__':main()
