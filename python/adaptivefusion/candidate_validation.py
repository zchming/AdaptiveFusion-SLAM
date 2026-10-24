"""Independent evaluation of fixed proposals; never imported by tracking/BA.

Depth probes reuse the endpoint images but exclude all native ORB depth-sample
neighborhoods. They are correlated scene measurements, not independent scenes.
Truth errors are evaluation-only and must never select measurements for BA.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .geometry import check_pose, inverse
from .evaluation import associate

CONFIG=dict(grid_step=16,grid_offset=8,feature_exclusion_radius=4,
            depth_spread_m=.05,agreement_m=.03,minimum_comparisons=100,
            minimum_projection_coverage=.3,minimum_agreement_fraction=.7,
            maximum_median_m=.02,truth_tolerance_s=.02,
            maximum_truth_translation_m=.03,maximum_truth_rotation_deg=2.,
            meaningful_translation_m=.005,meaningful_rotation_deg=.5,
            meaningful_relative_improvement=.1)


def _depth_mask(depth, scale, minimum, maximum, keypoints):
    if depth.dtype!=np.uint16 or depth.ndim!=2 or min(depth.shape)<3:
        raise ValueError('native uint16 depth image required')
    if not np.isfinite([scale,minimum,maximum]).all() or not 0<minimum<maximum or scale<=0:
        raise ValueError('invalid depth calibration')
    mask=np.zeros(depth.shape,bool)
    windows=np.lib.stride_tricks.sliding_window_view(depth,(3,3))
    low=windows.min(axis=(-2,-1));high=windows.max(axis=(-2,-1))
    z=depth[1:-1,1:-1]/scale
    mask[1:-1,1:-1]=(low>0)&((high.astype(float)-low)/scale<=CONFIG['depth_spread_m'])&(z>=minimum)&(z<=maximum)
    pixels=np.asarray(keypoints,float).reshape(-1,2)
    if not np.isfinite(pixels).all():raise ValueError('invalid feature coordinates')
    radius=CONFIG['feature_exclusion_radius']
    for x,y in np.floor(pixels+.5).astype(int):
        if not 0<=x<depth.shape[1] or not 0<=y<depth.shape[0]:continue
        mask[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]=False
    return mask


def depth_probe(source_depth,target_depth,source_features,target_features,camera,depth_config,t_target_source):
    """Project a fixed grid; report every valid residual, including occlusion.

    No ICP, correspondence search, residual-based filtering or pose update.
    In-plane motion on a single plane can pass: truth/multiview checks remain
    necessary. Low comparison coverage is explicitly not an evaluation pass.
    """
    transform=check_pose(t_target_source);camera=np.asarray(camera,float)
    if camera.shape!=(4,) or not np.isfinite(camera).all() or min(camera[:2])<=0:
        raise ValueError('invalid camera')
    scale,minimum,maximum=depth_config
    a=_depth_mask(source_depth,scale,minimum,maximum,source_features)
    b=_depth_mask(target_depth,scale,minimum,maximum,target_features)
    fx,fy,cx,cy=camera;step=CONFIG['grid_step'];offset=CONFIG['grid_offset']
    u,v=np.meshgrid(np.arange(offset,a.shape[1],step),np.arange(offset,a.shape[0],step));u,v=u.ravel(),v.ravel()
    grid_count=len(u);keep=a[v,u];u,v=u[keep],v[keep];source_count=len(u)
    z=source_depth[v,u]/scale
    xyz=np.c_[(u-cx)/fx*z,(v-cy)/fy*z,z]
    predicted=xyz@transform[:3,:3].T+transform[:3,3]
    positive=np.isfinite(predicted).all(axis=1)&(predicted[:,2]>1e-8)
    projected=np.zeros((len(predicted),2),float)
    projected[positive]=predicted[positive,:2]/predicted[positive,2,None]*[fx,fy]+[cx,cy]
    inside=positive&(projected[:,0]>=0)&(projected[:,0]<b.shape[1]-.5)&(projected[:,1]>=0)&(projected[:,1]<b.shape[0]-.5)
    indices=np.flatnonzero(inside);q=np.floor(projected[indices]+.5).astype(int)
    usable=b[q[:,1],q[:,0]];indices=indices[usable];q=q[usable]
    signed=predicted[indices,2]-target_depth[q[:,1],q[:,0]]/scale
    residual=np.abs(signed);count=len(residual);coverage=count/source_count if source_count else 0.
    result={'grid_pixels':grid_count,'source_eligible':source_count,'positive_predictions':int(positive.sum()),
            'inside_predictions':int(inside.sum()),'comparisons':count,'projection_coverage':coverage,
            'agreement_fraction':float(np.mean(residual<=CONFIG['agreement_m'])) if count else None,
            'median_abs_m':float(np.median(residual)) if count else None,
            'p95_abs_m':float(np.quantile(residual,.95)) if count else None,
            'predicted_behind_fraction':float(np.mean(signed>CONFIG['agreement_m'])) if count else None,
            'predicted_in_front_fraction':float(np.mean(signed < -CONFIG['agreement_m'])) if count else None}
    sufficient=count>=CONFIG['minimum_comparisons'] and coverage>=CONFIG['minimum_projection_coverage']
    result['status']='evaluated' if sufficient else 'insufficient_coverage'
    result['depth_consistency_pass']=bool(sufficient and result['agreement_fraction']>=CONFIG['minimum_agreement_fraction'] and result['median_abs_m']<=CONFIG['maximum_median_m'])
    return result


def relative_error(t_target_source, truth_source_wc, truth_target_wc):
    expected=inverse(truth_target_wc)@check_pose(truth_source_wc)
    error=inverse(expected)@check_pose(t_target_source)
    translation=float(np.linalg.norm(error[:3,3]));rotation=float(np.rad2deg(Rotation.from_matrix(error[:3,:3]).magnitude()))
    return {'translation_m':translation,'rotation_deg':rotation,
            'gt_fidelity_pass':translation<=CONFIG['maximum_truth_translation_m'] and rotation<=CONFIG['maximum_truth_rotation_deg']}


def truth_endpoints(times,poses,source_timestamp,target_timestamp):
    """Existing nearest-unused protocol and predeclared bracketed interpolation."""
    if not source_timestamp<target_timestamp:raise ValueError('ordered distinct timestamps required')
    times=np.asarray(times,float);poses=np.asarray(poses,float)
    if times.ndim!=1 or len(times)!=len(poses) or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('invalid ground-truth stream')
    for p in poses:check_pose(p)
    query=np.array([source_timestamp,target_timestamp]);matched=associate(times,query,CONFIG['truth_tolerance_s'])
    result={'nearest':None,'interpolated':None,'association_differences_s':[],'bracket_spans_s':[]}
    if len(matched)==2 and [j for _,j in matched]==[0,1]:
        result['nearest']=[poses[i] for i,j in matched]
        result['association_differences_s']=[float(abs(times[i]-query[j])) for i,j in matched]
    interpolated=[]
    for stamp in query:
        i=int(np.searchsorted(times,stamp))
        if i<len(times) and times[i]==stamp:interpolated.append(poses[i]);result['bracket_spans_s'].append(0.);continue
        if i==0 or i==len(times) or max(stamp-times[i-1],times[i]-stamp)>CONFIG['truth_tolerance_s']:break
        ratio=(stamp-times[i-1])/(times[i]-times[i-1]);t=np.eye(4)
        t[:3,:3]=Slerp(times[i-1:i+1],Rotation.from_matrix(poses[i-1:i+1,:3,:3]))([stamp]).as_matrix()[0]
        t[:3,3]=(1-ratio)*poses[i-1,:3,3]+ratio*poses[i,:3,3]
        interpolated.append(t);result['bracket_spans_s'].append(float(times[i]-times[i-1]))
    if len(interpolated)==2:result['interpolated']=interpolated
    return result


def pair_value(candidate,baseline):
    """Evaluation label only; never an edge-selection or optimization rule."""
    dt=baseline['translation_m']-candidate['translation_m'];dr=baseline['rotation_deg']-candidate['rotation_deg']
    translation=(dt>=CONFIG['meaningful_translation_m'] and dt>=CONFIG['meaningful_relative_improvement']*baseline['translation_m'] and dr>=-CONFIG['meaningful_rotation_deg'])
    rotation=(dr>=CONFIG['meaningful_rotation_deg'] and dr>=CONFIG['meaningful_relative_improvement']*baseline['rotation_deg'] and dt>=-CONFIG['meaningful_translation_m'])
    return {'translation_change_m':-dt,'rotation_change_deg':-dr,
            'pair_level_value_supported':bool(candidate['gt_fidelity_pass'] and (translation or rotation))}
