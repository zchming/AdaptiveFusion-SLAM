#!/usr/bin/env python3
"""Fixed confidence and spatial controls at the cached F candidate cardinality."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from adaptivefusion.state import file_sha256
from run_camera_pnp_comparison import write_json
from run_magsac_comparison import fit

def select(rows,scores,count,balanced):
 rank=sorted(range(len(rows)),key=lambda i:(-scores[i],rows[i]['source_feature']))
 if balanced:
  cells={}
  for i in rank:
   u,v=rows[i]['source_pixel'];cell=(min(3,int(v/120)),min(3,int(u/160)));cells.setdefault(cell,[]).append(i)
  rank=[];level=0
  while len(rank)<len(rows):
   for cell in sorted(cells):
    if level<len(cells[cell]):rank.append(cells[cell][level])
   level+=1
 return [rows[i] for i in sorted(rank[:count])]

def main():
 p=argparse.ArgumentParser()
 for k in ['baseline-study','reference-study','matching','native-binary','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise ValueError('output exists')
 datasets=[];hashes={}
 for study in [a.baseline_study,a.reference_study]:
  f=study/'fit-only.json';assert file_sha256(f)==json.loads((study/'fit-seal.json').read_text())['fit-only.json'];datasets.append(json.loads(f.read_text()));hashes[str(f.resolve())]=file_sha256(f)
 old,ref=datasets;lookup={(c['scene'],c['pair']):c for c in ref['cases']};cases=[]
 for c in old['cases']:
  original=next(x for x in c['arms'] if x['camera_profile']=='rootsift_lightglue');reference=next(x for x in lookup[c['scene'],c['pair']]['arms'] if x['camera_profile']=='F_MAGSAC');rows=original['rows'];count=len(reference['rows']);path=a.matching/original['match_file'];assert old['input_sha256'][str(path.resolve())]==file_sha256(path);hashes[str(path.resolve())]=file_sha256(path)
  with np.load(path) as z:scores={tuple(m):float(s) for m,s in zip(z['matches'],z['scores'])}
  values=[scores[r['source_feature'],r['target_feature']] for r in rows];assert np.isfinite(values).all()
  case={k:c[k] for k in ['scene','pair','source_timestamp','target_timestamp','raw_baseline']};case['arms']=[]
  for name,balanced in [('confidence_topk',False),('grid_balanced',True)]:
   selected=select(rows,values,count,balanced);assert len(selected)==count
   arm=dict(camera_profile=name,estimator='existing_native_pnp_ransac',seed=17,rows=selected)
   if 'failure' in reference:arm['failure']='F count unavailable: '+reference['failure']
   case['arms'].append(arm)
  cases.append(case)
 hashes[str(a.native_binary.resolve())]=file_sha256(a.native_binary);hashes[str(Path(__file__).resolve())]=file_sha256(Path(__file__))
 a.output.mkdir();(a.output/'verification').mkdir()
 write_json(a.output/'verification/verification.json',dict(cases=cases,input_sha256=hashes,truth_read=False))
 fit(SimpleNamespace(output=a.output/'study',verification=a.output/'verification',baseline_study=a.baseline_study,native_binary=a.native_binary))
if __name__=='__main__':main()
