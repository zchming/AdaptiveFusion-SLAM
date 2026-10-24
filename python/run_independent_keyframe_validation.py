#!/usr/bin/env python3
"""Run the frozen independent-sequence study without touching production tracking."""
import argparse,csv,json,os,resource,signal,subprocess,time
from pathlib import Path
from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames

def main():
 p=argparse.ArgumentParser()
 for k in ['selection','datasets','parent-protocol','baseline-binary','replay-binary','graph-binary','native-binary','lightglue-source','weights','native-library-path','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();repo=Path(__file__).resolve().parents[1];selection=json.loads(a.selection.read_text());data=json.loads(a.datasets.read_text())['records'];parent=json.loads(a.parent_protocol.read_text())
 if a.output.exists():raise ValueError('refuse overwrite')
 if [x['scene'] for x in data]!=selection['selected_sequences']:raise ValueError('sequence list changed')
 for path,sha in selection['frozen_source_sha256'].items():
  if file_sha256(repo/path)!=sha:raise ValueError('frozen source changed: '+path)
 a.output.mkdir();begin=time.perf_counter();records=[];commands=[];sources={};inventory={};hashes={}
 def digest(path):hashes[str(Path(path).resolve())]=file_sha256(path)
 for path in [a.selection,a.datasets,a.parent_protocol,a.baseline_binary,a.replay_binary,a.graph_binary,a.native_binary,a.weights,Path(__file__),repo/'python/run_keyframe_auxiliary.py',repo/'python/offline_matching_service.py']:digest(path)
 for d in data:
  scene=d['scene'];dataset=Path(d['dataset']);counts={}
  for index in ['rgb.txt','depth.txt']:
   digest(dataset/index);paths=[dataset/line.split()[1] for line in (dataset/index).read_text().splitlines() if line.strip() and not line.startswith('#')]
   for path in paths:digest(path)
   counts[index]=len(paths)
  inventory[scene]=dict(dataset=str(dataset),RGB_index_entries=counts['rgb.txt'],depth_index_entries=counts['depth.txt'],keyframe_request_upper_bound=(counts['rgb.txt']-1)//5,every_frame_request_upper_bound=counts['rgb.txt']-1)
 (a.output/'inputs-before-baseline.json').write_text(json.dumps(dict(inventory=inventory,input_sha256=hashes,GT_read=False),indent=2)+'\n')
 env=os.environ.copy();lib=str(a.native_library_path);env['LD_LIBRARY_PATH']=':'.join([lib,lib+'/blas',lib+'/lapack',str(a.native_library_path.parent),'/usr/lib/wsl/lib',env.get('LD_LIBRARY_PATH','')])
 def memory_budget():resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
 def run(stage,cmd,timeout=1200,native=False):
  commands.append(dict(stage=stage,command=list(map(str,cmd))));(a.output/'commands.json').write_text(json.dumps(commands,indent=2))
  start=time.perf_counter();timed=False
  with (a.output/(stage+'.log')).open('x') as log:
   process=subprocess.Popen(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,env=env if native else None,start_new_session=True,preexec_fn=memory_budget if native else None)
   try:code=process.wait(timeout=timeout)
   except subprocess.TimeoutExpired:timed=True;os.killpg(process.pid,signal.SIGKILL);code=process.wait()
  record=dict(stage=stage,returncode=code,timeout=timed,seconds=time.perf_counter()-start,status='complete' if code==0 else 'process_failed')
  records.append(record);(a.output/'status.json').write_text(json.dumps(records,indent=2));print(json.dumps(record),flush=True);return record
 runs=a.output/'runs';runs.mkdir()
 for d in data:
  scene=d['scene'];dest=runs/scene/'baseline';dest.mkdir(parents=True)
  record=run(scene+'-baseline',['/usr/bin/time','-f','%M','-o',a.output/(scene+'-baseline.rss.txt'),a.baseline_binary,d['dataset'],dest/'trajectory.tum','1000000','-','none',dest/'cache','--export-keyframe-features'],native=True)
  if record['returncode']==0:
   frames=load_frames(dest/'cache/frames.txt');sources[scene]=dict(frames=str(dest/'cache/frames.txt'),features=str(dest/'cache/features'),dataset=d['dataset']);inventory[scene]['synchronized_frames']=len(frames)
 (a.output/'inference-sources.json').write_text(json.dumps(sources,indent=2))
 if sources:
  record=run('freeze-keyframes',[os.sys.executable,'-B',repo/'python/freeze_keyframe_auxiliary.py','--baseline-runs',runs,'--sources',a.output/'inference-sources.json','--native-binary',a.native_binary,'--output',a.output/'freeze-template'])
  if record['returncode']!=0:raise ValueError('keyframe freezing failed')
  frozen=json.loads((a.output/'freeze-template/protocol.json').read_text())
  if frozen['matcher']!=parent['matcher'] or frozen['graph']!=parent['graph']:raise ValueError('parameters differ from V31')
  frozen.update(stage='V32 independent keyframe auxiliary',selection_scope='two new full sequences frozen before any baseline/candidate output; no test-time tuning',parent_protocol_sha256=file_sha256(a.parent_protocol))
  for scene,s in frozen['scenes'].items():
   if s['request_budget']>inventory[scene]['keyframe_request_upper_bound']:raise ValueError('predeclared keyframe budget exceeded')
   inventory[scene].update(keyframes=len(s['keyframes']),keyframe_exact_calls=s['request_budget'],every_frame_exact_max_calls=s['input_frames']-1)
  (a.output/'keyframe-protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
  (a.output/'budgets-before-matching.json').write_text(json.dumps(inventory,indent=2)+'\n')
 else:
  (a.output/'budgets-before-matching.json').write_text(json.dumps(inventory,indent=2)+'\n')
 # All declared scenes retain their every-frame arm even if baseline failed.
 for d in data:
  scene=d['scene'];socket=a.output/(scene[4]+'.sock');ready=a.output/(scene+'.ready.json');service_log=(a.output/(scene+'-service.log')).open('x')
  cmd=[os.sys.executable,'-B',str(repo/'python/offline_matching_service.py'),'--dataset',d['dataset'],'--lightglue-source',str(a.lightglue_source),'--weights',str(a.weights),'--socket',str(socket),'--log',str(a.output/(scene+'-service.jsonl')),'--ready',str(ready)]
  commands.append(dict(stage=scene+'-service',command=cmd));startup=time.perf_counter()
  service=subprocess.Popen(cmd,stdout=service_log,stderr=subprocess.STDOUT,start_new_session=True)
  while not ready.exists() and service.poll() is None and time.perf_counter()-startup<60:time.sleep(.1)
  if ready.exists():
   record=run(scene+'-magsac',['/usr/bin/time','-f','%M','-o',a.output/(scene+'-magsac.rss.txt'),a.replay_binary,d['dataset'],runs/scene/'magsac','magsac',socket],native=True)
  else:
   records.append(dict(stage=scene+'-magsac',status='service_start_failed',seconds=time.perf_counter()-startup));(a.output/'status.json').write_text(json.dumps(records,indent=2))
  if service.poll() is None:
   try:service.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(service.pid,signal.SIGKILL);service.wait()
  service_log.close()
  records.append(dict(stage=scene+'-service',returncode=service.returncode,total_seconds_including_startup=time.perf_counter()-startup,status='complete' if service.returncode==0 else 'service_failed'));(a.output/'status.json').write_text(json.dumps(records,indent=2))
 if sources:
  run('keyframe-auxiliary',['/usr/bin/time','-f','%e %M','-o',a.output/'auxiliary-resource.txt',os.sys.executable,'-B',repo/'python/run_keyframe_auxiliary.py','--protocol',a.output/'keyframe-protocol.json','--graph-binary',a.graph_binary,'--lightglue-source',a.lightglue_source,'--weights',a.weights,'--native-library-path',a.native_library_path,'--output',a.output/'auxiliary'],timeout=1400)
 for path,sha in hashes.items():
  if file_sha256(path)!=sha:raise ValueError('frozen input changed during study')
 (a.output/'study-wall.json').write_text(json.dumps(dict(seconds=time.perf_counter()-begin,GT_read=False),indent=2))
 # Seal the entire study, including all failure/partial outputs, before external evaluation.
 seal={str(path.relative_to(a.output)):file_sha256(path) for path in a.output.rglob('*') if path.is_file()}
 (a.output/'fit-seal.json').write_text(json.dumps(seal,indent=2)+'\n')
 print('SEALED_WITHOUT_GT',flush=True)
if __name__=='__main__':main()
