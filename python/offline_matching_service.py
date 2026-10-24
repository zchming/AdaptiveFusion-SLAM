#!/usr/bin/env python3
"""Local offline replay matcher. Accepts RGB/depth paths only; never reads truth."""
import argparse,json,shlex,socket,sys,time,traceback,subprocess,resource
from collections import OrderedDict
from pathlib import Path
import numpy as np
from PIL import Image
from adaptivefusion.bridge_candidates import sample_depth
from adaptivefusion.geometry import unproject
from adaptivefusion.matching_verification import checked_mask
from adaptivefusion.state import file_sha256
from run_rootsift_comparison import COMMIT,WEIGHT_SHA,validate_features

class Matcher:
 def __init__(self,a):
  import cv2,torch
  sys.path.insert(0,str(a.lightglue_source));from lightglue import LightGlue
  from lightglue.sift import sift_to_rootsift,filter_dog_point
  assert subprocess.check_output(['git','-C',str(a.lightglue_source),'rev-parse','HEAD'],text=True).strip()==COMMIT
  assert file_sha256(a.weights)==WEIGHT_SHA and torch.cuda.is_available()
  self.cv2,self.torch=cv2,torch;self.convert,self.filter=sift_to_rootsift,filter_dog_point
  torch.set_num_threads(8);torch.manual_seed(17);cv2.setNumThreads(8);cv2.ocl.setUseOpenCL(False);cv2.setRNGSeed(17)
  torch.cuda.set_per_process_memory_fraction(11*1024**3/torch.cuda.get_device_properties(0).total_memory)
  self.sift=cv2.SIFT_create(nfeatures=2048,contrastThreshold=.0066667,edgeThreshold=10,nOctaveLayers=4)
  start=time.perf_counter();self.model=LightGlue(features=None,input_dim=128,add_scale_ori=True).eval()
  state=torch.load(a.weights,map_location='cpu',weights_only=True)
  for i in range(9):state={k.replace(f'self_attn.{i}',f'transformers.{i}.self_attn').replace(f'cross_attn.{i}',f'transformers.{i}.cross_attn'):v for k,v in state.items()}
  loaded=self.model.load_state_dict(state,strict=False);assert set(loaded.missing_keys)=={'confidence_thresholds'} and not loaded.unexpected_keys
  self.model=self.model.cuda();torch.cuda.synchronize();self.load_s=time.perf_counter()-start
  self.features=OrderedDict();self.root=a.dataset.resolve();self.log=a.log.open('x');self.requests=0
  self.evidence=a.log.with_suffix('.matches');self.evidence.mkdir()
 def feature(self,path):
  if path in self.features:self.features.move_to_end(path);return self.features[path],0.
  cv2,torch=self.cv2,self.torch;start=time.perf_counter();bgr=cv2.imread(path,cv2.IMREAD_COLOR)
  if bgr is None or bgr.shape!=(480,640,3):raise ValueError('expected640x480 image')
  detections,descriptors=self.sift.detectAndCompute(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),None)
  if descriptors is None:
   f=dict(keypoints=np.empty((0,2),np.float32),descriptors=np.empty((0,128),np.float32),scales=np.empty(0,np.float32),oris=np.empty(0,np.float32),image_size=np.array([640,480],np.float32))
  else:
   xy=np.asarray([k.pt for k in detections],np.float32);scores=np.asarray([k.response for k in detections],np.float32);scales=np.asarray([k.size for k in detections],np.float32);angles=np.deg2rad(np.asarray([k.angle for k in detections],np.float32))
   keep=self.filter(xy,scales,angles,bgr.shape[:2],0,scores=scores)
   if len(keep)>2048:keep=keep[torch.topk(torch.from_numpy(scores[keep]),2048).indices.numpy()]
   f=dict(keypoints=xy[keep],descriptors=self.convert(torch.from_numpy(descriptors[keep])).numpy(),scales=scales[keep],oris=angles[keep],image_size=np.array([640,480],np.float32))
  validate_features(f);self.features[path]=f
  while len(self.features)>4:self.features.popitem(last=False)
  return f,time.perf_counter()-start
 def request(self,line):
  cv2,torch=self.cv2,self.torch;mode,reference,current,rgb0,dp,rgb1=shlex.split(line)
  if mode not in ['lightglue','magsac']:raise ValueError('unknown mode')
  for path in [rgb0,dp,rgb1]:
   if not Path(path).resolve().is_relative_to(self.root):raise ValueError('path outside explicit dataset')
  start=time.perf_counter();f0,t0=self.feature(rgb0);f1,t1=self.feature(rgb1);torch.cuda.reset_peak_memory_stats();matching_start=time.perf_counter()
  if not len(f0['keypoints']) or not len(f1['keypoints']):pairs=np.empty((0,2),np.int64)
  else:
   gpu=[{k:torch.from_numpy(v)[None].cuda() for k,v in f.items()} for f in [f0,f1]]
   with torch.inference_mode():result=self.model({'image0':gpu[0],'image1':gpu[1]})
   pairs=result['matches'][0].cpu().numpy();order=np.argsort(pairs[:,0],kind='stable');pairs=pairs[order]
  torch.cuda.synchronize();matching_s=time.perf_counter()-matching_start;mask=np.ones(len(pairs),bool);status='ok';f_s=0.;failure=None
  if mode=='magsac':
   begin=time.perf_counter()
   try:
    if len(pairs)<8:raise ValueError('insufficient2D matches')
    cv2.setRNGSeed(17);matrix,mm=cv2.findFundamentalMat(f0['keypoints'][pairs[:,0]],f1['keypoints'][pairs[:,1]],cv2.USAC_MAGSAC,ransacReprojThreshold=3.,confidence=.999,maxIters=10000)
    matrix,mask=checked_mask(matrix,mm,len(pairs))
   except Exception as error:mask[:]=False;status='F_failed';failure=str(error)
   f_s=time.perf_counter()-begin
  with Image.open(dp) as im:depth=np.array(im)
  rows=[]
  for i,j in pairs[mask]:
   uv=f0['keypoints'][i];z=sample_depth(depth,uv,5000.)
   if z is not None:rows.append((int(i),*map(float,uv),*unproject([517.3,516.5,318.6,255.3],uv,z),*map(float,f1['keypoints'][j])))
  np.savez_compressed(self.evidence/f'{self.requests:06d}.npz',matches=pairs,F_mask=mask,rows=np.asarray(rows,float).reshape(-1,8))
  record=dict(mode=mode,reference_id=int(reference),frame_id=int(current),matches=len(pairs),F_retained=int(mask.sum()),depth_rows=len(rows),status=status,failure=failure,extraction_s=t0+t1,matching_s=matching_s,F_s=f_s,wall_s=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  self.log.write(json.dumps(record)+'\n');self.log.flush();self.requests+=1
  return ('ROWS '+str(len(rows))+' '+status+'\n'+''.join(' '.join(format(x,'.17g') if not isinstance(x,int) else str(x) for x in r)+'\n' for r in rows)).encode()

def main():
 p=argparse.ArgumentParser()
 for k in ['dataset','lightglue-source','weights','socket','log','ready']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 if a.socket.exists() or a.log.exists() or a.ready.exists():raise ValueError('refuse overwrite')
 m=Matcher(a);server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(str(a.socket));server.listen(1)
 a.ready.write_text(json.dumps(dict(model_load_s=m.load_s,opencv=m.cv2.__version__,gpu=m.torch.cuda.get_device_name(0),truth_read=False)))
 connection,_=server.accept()
 try:
  with connection,connection.makefile('rwb') as stream:
   while True:
    line=stream.readline()
    if not line:break
    try:stream.write(m.request(line.decode()));stream.flush()
    except Exception as error:
     m.log.write(json.dumps(dict(status='service_error',error=str(error),traceback=traceback.format_exc()))+'\n');m.log.flush();stream.write(b'ROWS 0 service_error\n');stream.flush()
 finally:server.close();m.log.close();a.socket.unlink(missing_ok=True)
if __name__=='__main__':main()
