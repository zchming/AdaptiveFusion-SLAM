"""Read-only RGB-D correspondence proposals; never edits a map or claims accuracy."""
import numpy as np
from .geometry import unproject

# Fixed before real-image inspection. Pixel/depth gates are conservative heuristics.
CONFIG = dict(fb_pixels=1., ownership_pixels=2., depth_spread_m=.05,
              rigid_threshold_m=.03, ransac_trials=256, seed=17,
              minimum_fit=12, minimum_validation=6, minimum_ratio=.5)


def rigid_fit(source, target):
    """Kabsch transform: target = R @ source + t, without a scale parameter."""
    a, b = np.asarray(source,float), np.asarray(target,float)
    if (a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a)<3
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('invalid rigid correspondences')
    x, y = a-a.mean(0), b-b.mean(0)
    for cloud in [x,y]:
        sv=np.linalg.svd(cloud,compute_uv=False)
        if sv[0]<=1e-10 or sv[1]<=1e-8*sv[0]:
            raise ValueError('collinear rigid support')
    u,_,vt=np.linalg.svd(x.T@y)
    sign=np.eye(3);sign[2,2]=np.linalg.det(vt.T@u.T)
    rotation=vt.T@sign@u.T
    t=np.eye(4);t[:3,:3]=rotation;t[:3,3]=b.mean(0)-rotation@a.mean(0)
    return t


def residuals(transform, source, target):
    return np.linalg.norm(source@transform[:3,:3].T+transform[:3,3]-target,axis=1)


def verify_rigid(source, target, ids):
    """Fit on even point IDs; odd IDs validate without refitting or pose priors.

    This split tests within-pair consistency only, not independent scene accuracy.
    Returned inliers are proposals, never permission to mutate cached ownership.
    """
    source,target=np.asarray(source,float).reshape(-1,3),np.asarray(target,float).reshape(-1,3)
    ids=np.asarray(ids,dtype=np.int64)
    if source.shape!=target.shape or len(ids)!=len(source) or len(set(ids.tolist()))!=len(ids):
        raise ValueError('invalid rigid IDs/shapes')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('non-finite rigid correspondences')
    train=ids%2==0; valid=~train
    result={'accepted':False,'reason':'insufficient split support','fit_count':int(train.sum()),
            'validation_count':int(valid.sum()),'inlier_point_ids':[]}
    if train.sum()<CONFIG['minimum_fit'] or valid.sum()<CONFIG['minimum_validation']:return result
    a,b=source[train],target[train];rng=np.random.default_rng(CONFIG['seed'])
    best=np.zeros(len(a),bool)
    for _ in range(CONFIG['ransac_trials']):
        sample=rng.choice(len(a),3,replace=False)
        try: t=rigid_fit(a[sample],b[sample])
        except ValueError:continue
        mask=residuals(t,a,b)<=CONFIG['rigid_threshold_m']
        if mask.sum()>best.sum():best=mask
    if best.sum()<CONFIG['minimum_fit']:
        result['reason']='insufficient rigid inliers';return result
    try:t=rigid_fit(a[best],b[best])
    except ValueError:result['reason']='degenerate rigid support';return result
    errors=residuals(t,source,target);mask=errors<=CONFIG['rigid_threshold_m']
    fit_in=int((mask&train).sum());validation_in=int((mask&valid).sum())
    accepted=(fit_in>=CONFIG['minimum_fit'] and validation_in>=CONFIG['minimum_validation']
              and fit_in/train.sum()>=CONFIG['minimum_ratio']
              and validation_in/valid.sum()>=CONFIG['minimum_ratio'])
    result.update(accepted=bool(accepted),reason='consistent_pair_not_accuracy_certificate' if accepted else 'split consistency failed',
                  fit_inliers=fit_in,validation_inliers=validation_in,
                  validation_median_m=float(np.median(errors[valid])),
                  validation_p95_m=float(np.quantile(errors[valid],.95)),
                  t_target_source=t.tolist(),
                  inlier_point_ids=ids[mask].tolist() if accepted else [])
    return result


def sample_depth(depth, uv, scale):
    """Native uint16 nearest sample; reject discontinuities and image borders."""
    if depth.dtype!=np.uint16 or depth.ndim!=2 or not np.isfinite(scale) or scale<=0:
        raise ValueError('expected native metric-scaled uint16 depth')
    if not np.isfinite(uv).all():return None
    x,y=np.floor(np.asarray(uv)+.5).astype(int)
    if not 1<=x<depth.shape[1]-1 or not 1<=y<depth.shape[0]-1:return None
    patch=depth[y-1:y+2,x-1:x+2]
    if not patch.all() or (int(patch.max())-int(patch.min()))/scale>CONFIG['depth_spread_m']:return None
    return int(depth[y,x])/scale


def propose_pair(snapshot, source_id, target_id, gray_source, gray_target, depth_source, depth_target):
    """Track existing cached source observations using OpenCV LK, not a new frontend.

    Target positions near any cached observation are conservatively excluded.
    No cached pose is used for matching, filtering, or geometric verification.
    """
    import cv2
    if source_id==target_id:raise ValueError('distinct keyframes required')
    for kid,gray,depth in [(source_id,gray_source,depth_source),(target_id,gray_target,depth_target)]:
        k=snapshot.keyframes[kid]
        if gray.dtype!=np.uint8 or gray.shape!=(k.height,k.width) or depth.shape!=gray.shape or depth.dtype!=np.uint16:
            raise ValueError('image type/size mismatch')
    src=sorted([o for o in snapshot.observations if o.keyframe_id==source_id],key=lambda o:o.point_id)
    dst=[o for o in snapshot.observations if o.keyframe_id==target_id]
    shared={o.point_id for o in dst}
    # Confirm that supplied depth is the very same immutable sample stream.
    for obs,depth in [(o,depth_source) for o in src]+[(o,depth_target) for o in dst]:
        x,y=np.floor([obs.u+.5,obs.v+.5]).astype(int)
        raw=int(depth[y,x]) if 0<=y<depth.shape[0] and 0<=x<depth.shape[1] else 0
        if raw!=obs.raw_depth:raise ValueError('cached raw-depth provenance mismatch')
    src=[o for o in src if o.point_id not in shared]
    result={'source_keyframe':source_id,'target_keyframe':target_id,'source_unshared_observations':len(src),
            'existing_shared_landmarks':len(shared&{o.point_id for o in snapshot.observations if o.keyframe_id==source_id}),
            'candidate_count':0,'proposals':[],'accepted':False,'reason':'no eligible source observations'}
    if not src:return result
    uv=np.array([[o.u,o.v] for o in src],np.float32).reshape(-1,1,2)
    kw=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01))
    forward,ok,_=cv2.calcOpticalFlowPyrLK(gray_source,gray_target,uv,None,**kw)
    if forward is None:result['reason']='forward LK failed';return result
    finite=np.isfinite(forward).all(axis=(1,2));forward[~finite]=uv[~finite]
    back,ok_back,_=cv2.calcOpticalFlowPyrLK(gray_target,gray_source,forward,None,**kw)
    if back is None:result['reason']='backward LK failed';return result
    good=finite & ok.ravel().astype(bool)&ok_back.ravel().astype(bool)
    good &= np.linalg.norm(back[:,0]-uv[:,0],axis=1)<=CONFIG['fb_pixels']
    occupied=np.array([[o.u,o.v] for o in dst],float).reshape(-1,2)
    used=[];a=[];b=[];ids=[];proposals=[]
    for i,o in enumerate(src):
        if not good[i]:continue
        q=forward[i,0].astype(float)
        if len(occupied) and np.min(np.linalg.norm(occupied-q,axis=1))<=CONFIG['ownership_pixels']:continue
        if used and np.min(np.linalg.norm(np.array(used)-q,axis=1))<=CONFIG['ownership_pixels']:continue
        z1=sample_depth(depth_source,uv[i,0],snapshot.config.depth_scale)
        z2=sample_depth(depth_target,q,snapshot.config.depth_scale)
        if z1 is None or z2 is None:continue
        used.append(q);ids.append(o.point_id)
        a.append(unproject(snapshot.camera,uv[i,0],z1));b.append(unproject(snapshot.camera,q,z2))
        proposals.append({'point_id':o.point_id,'target_uv':q.tolist(),'target_raw_depth':int(round(z2*snapshot.config.depth_scale))})
    result['candidate_count']=len(ids)
    result.update(verify_rigid(a,b,ids))
    accepted=set(result['inlier_point_ids'])
    result['proposals']=[p for p in proposals if p['point_id'] in accepted]
    return result
