"""Truth-free common-observation checks for cached relative pose constraints."""
from pathlib import Path
import numpy as np
from PIL import Image
from .bridge_candidates import sample_depth
from .geometry import check_pose

def common_observations(rows,target_depth,depth_scale):
 """One model-independent subset: fixed matched target pixels with usable depth."""
 with Image.open(target_depth) as image:depth=np.asarray(image)
 samples=[sample_depth(depth,uv,depth_scale) for uv in rows[:,6:8]]
 valid=np.array([z is not None for z in samples])
 return rows[valid],np.array([z for z in samples if z is not None],float)

def score_pose(transform,rows,target_depths,config):
 """Mean Cauchy influence weight of standardized pixel+axial-depth residuals.

 This is a bounded engineering information multiplier, not a posterior
 probability or calibrated pose covariance. Native PnP inlier masks are unused.
 """
 check_pose(transform)
 points=rows[:,3:6]@transform[:3,:3].T+transform[:3,3]
 positive=points[:,2]>1e-8
 pixel=np.full(len(rows),np.inf);depth_error=np.abs(points[:,2]-target_depths)
 fx,fy,cx,cy=config['camera']
 projected=points[positive,:2]/points[positive,2,None]*[fx,fy]+[cx,cy]
 pixel[positive]=np.linalg.norm(projected-rows[positive,6:8],axis=1)
 energy=(pixel/config['pixel_scale_px'])**2+(depth_error/config['depth_scale_m'])**2
 influence=1/(1+energy)
 weight=float(np.clip(np.mean(influence),config['minimum_information_weight'],1.))
 return dict(weight=weight,mean_cauchy_influence=float(np.mean(influence)),samples=len(rows),positive_depth_samples=int(sum(positive)),median_pixel_error_px=float(np.median(pixel)) if np.isfinite(np.median(pixel)) else None,median_depth_error_m=float(np.median(depth_error)),pixel_consensus=int(sum(pixel<=config['pixel_scale_px'])),depth_consensus=int(sum(depth_error<=config['depth_scale_m'])),joint_consensus=int(sum((pixel<=config['pixel_scale_px'])&(depth_error<=config['depth_scale_m']))))

def confidence_pair(baseline,auxiliary,rows,target_depth,config):
 shared,z=common_observations(rows,target_depth,config['raw_depth_scale'])
 if len(shared)<config['minimum_depth_checks']:
  return dict(status='insufficient_target_depth',shared_rows=len(shared),baseline=dict(weight=1.),auxiliary=dict(weight=config['minimum_information_weight']))
 return dict(status='scored',shared_rows=len(shared),baseline=score_pose(baseline,shared,z,config),auxiliary=score_pose(auxiliary,shared,z,config))
