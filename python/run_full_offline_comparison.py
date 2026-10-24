#!/usr/bin/env python3
"""Full ordered native replay, no GT access and no writeback into existing runs."""
import argparse,json,os,resource,signal,subprocess,time
from pathlib import Path
from adaptivefusion.state import file_sha256

def main():
 p=argparse.ArgumentParser()
 for k in ['sources','binary','service-python','lightglue-source','weights','baseline-reference','protocol','output']:
  p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise ValueError('refuse overwrite')
 plan=json.loads(a.protocol.read_text());sources=json.loads(a.sources.read_text())
 for path,sha in plan['input_sha256'].items():
  if file_sha256(Path(path))!=sha:raise ValueError('frozen input changed: '+path)
 a.output.mkdir();records=[];commands=[]
 def budget():resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
 for mode in ['baseline','lightglue','magsac']:
  for scene,src in sources.items():
   if 'truth' in src:raise ValueError('inference-only sources required')
   scene_root=a.output/scene;scene_root.mkdir(exist_ok=True);label=scene+'-'+mode
   sock=a.output/(label+'.sock');ready=a.output/(label+'.ready.json');service=None;service_log=None;startup=time.perf_counter()
   if mode!='baseline':
    service_log=(a.output/(label+'.service.log')).open('x')
    cmd=[str(a.service_python),'-B',str(Path(__file__).with_name('offline_matching_service.py')),'--dataset',src['dataset'],'--lightglue-source',str(a.lightglue_source),'--weights',str(a.weights),'--socket',str(sock),'--log',str(a.output/(label+'.service.jsonl')),'--ready',str(ready)]
    commands.append(cmd);service=subprocess.Popen(cmd,stdout=service_log,stderr=subprocess.STDOUT,start_new_session=True)
    while not ready.exists() and service.poll() is None and time.perf_counter()-startup<60:time.sleep(.1)
    if not ready.exists():
     if service.poll() is None:os.killpg(service.pid,signal.SIGKILL)
     records.append(dict(scene=scene,mode=mode,status='service_start_failed',returncode=service.wait(),seconds=time.perf_counter()-startup));(a.output/'status.json').write_text(json.dumps(records,indent=2));service_log.close();continue
   cmd=['/usr/bin/time','-f','%M','-o',str(a.output/(label+'.rss.txt')),str(a.binary),src['dataset'],str(scene_root/mode),mode,str(sock) if service else '-'];commands.append(cmd)
   start=time.perf_counter();timed=False
   with (a.output/(label+'.log')).open('x') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=budget)
    try:code=child.wait(timeout=1200)
    except subprocess.TimeoutExpired:timed=True;os.killpg(child.pid,signal.SIGKILL);code=child.wait()
   wall=time.perf_counter()-start
   service_code=None
   if service:
    try:service_code=service.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(service.pid,signal.SIGKILL);service_code=service.wait()
    service_log.close()
   record=dict(scene=scene,mode=mode,returncode=code,timeout=timed,seconds=wall,total_seconds_including_startup=time.perf_counter()-startup,service_returncode=service_code,status='complete' if code==0 else 'process_failed')
   if mode=='baseline' and code==0:
    record['trajectory_matches_original']=((scene_root/mode/'trajectory.tum').read_bytes()==(a.baseline_reference/scene/'baseline.tum').read_bytes())
    record['frame_states_match_original']=((scene_root/mode/'cache/frames.txt').read_bytes()==(a.baseline_reference/scene/'cache/frames.txt').read_bytes())
    if not record['trajectory_matches_original'] or not record['frame_states_match_original']:
     records.append(record);(a.output/'status.json').write_text(json.dumps(records,indent=2));raise ValueError('baseline adapter parity failed; do not run candidates')
   records.append(record);(a.output/'status.json').write_text(json.dumps(records,indent=2));(a.output/'commands.json').write_text(json.dumps(commands,indent=2));print(label,record,flush=True)
 for path,sha in plan['input_sha256'].items():
  if file_sha256(Path(path))!=sha:raise ValueError('input changed during replay')
 (a.output/'fit-seal.json').write_text(json.dumps({str(p.relative_to(a.output)):file_sha256(p) for p in a.output.rglob('*') if p.is_file()},indent=2)+'\n')
if __name__=='__main__':main()
