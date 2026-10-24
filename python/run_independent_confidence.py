#!/usr/bin/env python3
"""Confidence-only rows with an independently frozen retained fraction; no F inputs."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from adaptivefusion.state import file_sha256
from run_camera_pnp_comparison import write_json
from run_simple_matching_controls import select
from run_magsac_comparison import fit

def main():
 p=argparse.ArgumentParser()
 for k in ['baseline-study','matching','budget-protocol','native-binary','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise ValueError('refuse overwrite')
 policy=json.loads(a.budget_protocol.read_text());num,den=policy['retained_fraction'];assert (num,den)==(9,10)
 f=a.baseline_study/'fit-only.json';assert file_sha256(f)==json.loads((a.baseline_study/'fit-seal.json').read_text())['fit-only.json'];old=json.loads(f.read_text());hashes={str(p.resolve()):file_sha256(p) for p in [f,a.budget_protocol,a.native_binary,Path(__file__),Path(__file__).with_name('run_simple_matching_controls.py')]};cases=[];start=time.perf_counter()
 for c in old['cases']:
  original=next(x for x in c['arms'] if x['camera_profile']=='rootsift_lightglue');rows=original['rows'];count=len(rows)*num//den;path=a.matching/original['match_file'];assert file_sha256(path)==old['input_sha256'][str(path.resolve())];hashes[str(path.resolve())]=file_sha256(path)
  with np.load(path) as z:scores={tuple(m):float(s) for m,s in zip(z['matches'],z['scores'])}
  values=[scores[r['source_feature'],r['target_feature']] for r in rows];assert np.isfinite(values).all()
  selected=select(rows,values,count,False);assert len(selected)==count
  case={k:c[k] for k in ['scene','pair','source_timestamp','target_timestamp','raw_baseline']};case['budget']=dict(original=len(rows),retained=count);case['arms']=[dict(camera_profile='confidence_90pct',estimator='existing_native_pnp_ransac',seed=17,rows=selected)];cases.append(case)
 a.output.mkdir();(a.output/'verification').mkdir();write_json(a.output/'selection-timing.json',dict(selection_s=time.perf_counter()-start,F_read=False,F_count_read=False,truth_read=False))
 write_json(a.output/'verification/verification.json',dict(cases=cases,input_sha256=hashes,truth_read=False))
 fit(SimpleNamespace(output=a.output/'study',verification=a.output/'verification',baseline_study=a.baseline_study,native_binary=a.native_binary))
if __name__=='__main__':main()
