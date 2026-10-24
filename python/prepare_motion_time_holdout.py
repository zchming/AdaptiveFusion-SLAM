#!/usr/bin/env python3
"""Preselect stratified, image-disjoint inputs using estimated motion and timestamps only."""
import argparse,json,shlex
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from adaptivefusion.geometry import inverse
from adaptivefusion.trajectory import load_frames
from adaptivefusion.state import file_sha256
from run_camera_pnp_comparison import write_json

def main():
 p=argparse.ArgumentParser()
 for k in ['sources','native-binary','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--exclude-study',type=Path,action='append',required=True)
 p.add_argument('--per-cell',type=int,default=4)
 a=p.parse_args()
 if a.output.exists():raise ValueError('refuse overwrite')
 sources=json.loads(a.sources.read_text()); cases=[];inventory=[];hashes={str(a.native_binary.resolve()):file_sha256(a.native_binary),str(a.sources.resolve()):file_sha256(a.sources),str(Path(__file__).resolve()):file_sha256(Path(__file__))};old=[]
 for study in a.exclude_study:
  fp=study/'fit-only.json';assert file_sha256(fp)==json.loads((study/'fit-seal.json').read_text())['fit-only.json'];old+=json.loads(fp.read_text())['cases'];hashes[str(fp.resolve())]=file_sha256(fp)
 old_rgb_hashes=set();new_rgb_hashes=set()
 for scene,src in sources.items():
  assert set(src)=={'frames','features','dataset'}
  frames=load_frames(src['frames']);meta={};hashes[str(Path(src['frames']).resolve())]=file_sha256(Path(src['frames']))
  for fp in Path(src['features']).glob('*.features'):
   lines=fp.read_text().splitlines()[:8];key=shlex.split(lines[2]);ims=shlex.split(lines[7]);meta[int(key[1])]=dict(fid=int(key[2]),ts=float(key[3]),dts=float(key[4]),path=fp,rgb=Path(src['dataset'])/ims[1])
  oldids={int(i) for c in old if c['scene']==scene for i in c['pair'].split('-')};times=[meta[i]['ts'] for i in oldids]
  old_rgb_hashes.update(file_sha256(meta[i]['rgb']) for i in oldids)
  pool=[];occupied=set()
  for i in sorted(meta):
   j=i+1
   if j not in meta or i in occupied or j in occupied:continue
   x,y=meta[i],meta[j];f,g=frames[x['fid']],frames[y['fid']]
   if not f.valid or not g.valid or any(abs(meta[k]['ts']-t)<1 for k in [i,j] for t in times):continue
   assert f.timestamp==x['ts'] and g.timestamp==y['ts'];dt=y['ts']-x['ts']
   if dt<=0:continue
   raw=inverse(g.t_wc)@f.t_wc;speed=float(np.linalg.norm(raw[:3,3])/dt);omega=float(np.rad2deg(Rotation.from_matrix(raw[:3,:3]).magnitude())/dt)
   pool.append(dict(scene=scene,pair=f'{i}-{j}',source_timestamp=x['ts'],target_timestamp=y['ts'],raw_baseline=raw.tolist(),translation_speed_m_s=speed,rotation_speed_deg_s=omega,motion_score=max(speed/.1,omega/10),rgb_depth_max_abs_ms=max(abs(x['dts']-x['ts']),abs(y['dts']-y['ts']))*1000))
   occupied.update([i,j])
  if not pool:inventory.append(dict(scene=scene,candidates=[],selected=[],reason='no eligible unseen pairs'));continue
  medm=float(np.median([r['motion_score'] for r in pool]));medt=float(np.median([r['rgb_depth_max_abs_ms'] for r in pool]));selected=[];counts={}
  for r in pool:r['motion_bin']='high' if r['motion_score']>=medm else 'low';r['time_bin']='high' if r['rgb_depth_max_abs_ms']>=medt else 'low'
  for mb in ['low','high']:
   for tb in ['low','high']:
    subset=[r for r in pool if r['motion_bin']==mb and r['time_bin']==tb];n=min(a.per_cell,len(subset));counts[mb+'_'+tb]=dict(available=len(subset),selected=n)
    if n:selected.extend(subset[i] for i in np.linspace(0,len(subset)-1,n,dtype=int))
  for r in selected:
   for k in map(int,r['pair'].split('-')):
    m=meta[k];sha=file_sha256(m['rgb']);assert sha not in old_rgb_hashes and sha not in new_rgb_hashes;new_rgb_hashes.add(sha)
    for fp in [m['path'],m['rgb']]:hashes[str(fp.resolve())]=file_sha256(fp)
  inventory.append(dict(scene=scene,motion_score_median=medm,time_ms_median=medt,counts=counts,candidates=pool,selected=[r['pair'] for r in selected]));cases+=selected
 assert not(new_rgb_hashes & old_rgb_hashes)
 a.output.mkdir(parents=True)
 protocol=dict(status='frozen_before_matching_fitting_GT',cases=len(cases),per_scene_cell_cap=a.per_cell,buffer_s=1,selection='greedy vertex-disjoint adjacent valid keyframes; within-scene median bins from eligible pool; evenly spaced per cell; never refill empty cells',motion='estimated endpoint relative speed score=max(translation_m_s/0.1,rotation_deg_s/10); not GT or instantaneous exposure motion',time='maximum absolute RGB-depth endpoint timestamp gap',scope='within-sequence observational stratification; no causal claim',inventories=inventory,images=len(new_rgb_hashes),rgb_overlap_with_prior=0,input_sha256=hashes)
 write_json(a.output/'selection-protocol.json',protocol);write_json(a.output/'fit-only.json',dict(stage='pair_manifest_only_no_fitting',cases=cases,input_sha256=hashes,truth_read=False));write_json(a.output/'fit-seal.json',{n:file_sha256(a.output/n) for n in ['fit-only.json','selection-protocol.json']})
 print(json.dumps(dict(cases=len(cases),counts=[dict(scene=r['scene'],counts=r.get('counts')) for r in inventory])))
if __name__=='__main__':main()
