"""Read-only duplicate-landmark evidence. No map merge or accuracy certificate."""
from collections import defaultdict
from itertools import combinations
from scipy.spatial.transform import Rotation
import numpy as np
from .bridge_candidates import CONFIG as GEOMETRY_CONFIG, sample_depth, verify_rigid
from .geometry import unproject

CONFIG = dict(window_each_side=2, match_pixels=2., minimum_pair_views=2,
              minimum_keyframe_views=3, cycle_translation_m=.03, cycle_rotation_rad=.02, geometry=dict(GEOMETRY_CONFIG))


def reciprocal_matches(source_uv, target_uv, forward, backward, source_ok, target_ok):
    """Unambiguous reciprocal nearest features, never many-to-one ownership."""
    a,b=np.asarray(source_uv,float).reshape(-1,2),np.asarray(target_uv,float).reshape(-1,2)
    f,g=np.asarray(forward,float).reshape(-1,2),np.asarray(backward,float).reshape(-1,2)
    if len(f)!=len(a) or len(g)!=len(b):raise ValueError('flow shape mismatch')
    if not len(a) or not len(b):return []
    def unique_neighbor(query,reference,valid):
        distances=np.linalg.norm(query[:,None,:]-reference[None,:,:],axis=2)
        close=distances<=CONFIG['match_pixels']
        return {i:int(np.flatnonzero(close[i])[0]) for i in range(len(query))
                if valid[i] and np.isfinite(query[i]).all() and close[i].sum()==1}
    ab=unique_neighbor(f,b,source_ok);ba=unique_neighbor(g,a,target_ok)
    return [(i,j) for i,j in ab.items() if ba.get(j)==i]


def _flow(gray_source, gray_target, uv):
    import cv2
    if not len(uv):return np.empty((0,2)),np.empty(0,bool)
    points=np.asarray(uv,np.float32).reshape(-1,1,2)
    kw=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01))
    forward,status,_=cv2.calcOpticalFlowPyrLK(gray_source,gray_target,points,None,**kw)
    if forward is None:return np.zeros((len(points),2)),np.zeros(len(points),bool)
    finite=np.isfinite(forward).all(axis=(1,2));forward[~finite]=points[~finite]
    back,back_status,_=cv2.calcOpticalFlowPyrLK(gray_target,gray_source,forward,None,**kw)
    if back is None:return forward[:,0],np.zeros(len(points),bool)
    good=(finite & status.ravel().astype(bool) & back_status.ravel().astype(bool)
          & (np.linalg.norm(back[:,0]-points[:,0],axis=1)<=GEOMETRY_CONFIG['fb_pixels']))
    return forward[:,0],good


def audit_pair(snapshot, first, second, gray_first, gray_second, depth_first, depth_second, *, endpoint_flows=None):
    """Both native feature streams drive LK; existing shared IDs are controls.

    Fitting and validation split by source point-ID parity as in the prior audit.
    Repeated views of a point are not new independent validation units.
    """
    if not 0<=first<second<len(snapshot.keyframes):raise ValueError('ordered distinct keyframes required')
    observations=[]
    for kid,gray,depth in [(first,gray_first,depth_first),(second,gray_second,depth_second)]:
        k=snapshot.keyframes[kid]
        if gray.dtype!=np.uint8 or gray.shape!=(k.height,k.width) or depth.dtype!=np.uint16 or depth.shape!=gray.shape:
            raise ValueError('native image type/size mismatch')
        obs=sorted([o for o in snapshot.observations if o.keyframe_id==kid],key=lambda o:o.point_id)
        for o in obs:
            x,y=np.floor([o.u+.5,o.v+.5]).astype(int)
            raw=int(depth[y,x]) if 0<=x<k.width and 0<=y<k.height else 0
            if raw!=o.raw_depth:raise ValueError('cached raw-depth provenance mismatch')
        observations.append(obs)
    source,target=observations
    a=np.array([[o.u,o.v] for o in source]).reshape(-1,2)
    b=np.array([[o.u,o.v] for o in target]).reshape(-1,2)
    if endpoint_flows is None:
        forward,ok_a=_flow(gray_first,gray_second,a);backward,ok_b=_flow(gray_second,gray_first,b)
    else:
        forward,ok_a,backward,ok_b=endpoint_flows
    matches=reciprocal_matches(a,b,forward,backward,ok_a,ok_b)
    records=[];x=[];y=[];ids=[]
    for i,j in matches:
        z1=sample_depth(depth_first,a[i],snapshot.config.depth_scale)
        z2=sample_depth(depth_second,b[j],snapshot.config.depth_scale)
        if z1 is None or z2 is None:continue
        ids.append(source[i].point_id)
        x.append(unproject(snapshot.camera,a[i],z1));y.append(unproject(snapshot.camera,b[j],z2))
        records.append({'source_point_id':source[i].point_id,'target_point_id':target[j].point_id,
                        'source_keyframe':first,'target_keyframe':second,
                        'source_uv':a[i].tolist(),'target_uv':b[j].tolist(),
                        'source_raw_depth':source[i].raw_depth,'target_raw_depth':target[j].raw_depth})
    fit=verify_rigid(x,y,ids);inliers=set(fit['inlier_point_ids'])
    for record in records:record['pair_geometric_inlier']=record['source_point_id'] in inliers
    return {'source_keyframe':first,'target_keyframe':second,
            'source_observations':len(source),'target_observations':len(target),
            'reciprocal_matches':len(matches),'depth_valid_matches':len(records),
            'different_id_matches':sum(v['source_point_id']!=v['target_point_id'] for v in records),
            'verification':fit,'records':records}


def audit_cycles(pair_reports, window_keyframes):
    """Three accepted pair transforms must compose in the same camera convention."""
    transforms={(p['source_keyframe'],p['target_keyframe']):np.array(p['verification']['t_target_source'])
                for p in pair_reports if p['verification']['accepted']}
    passed=set();failed=set();cycles=[]
    for a,b,c in combinations(sorted(window_keyframes),3):
        keys=[(a,b),(b,c),(a,c)]
        if not all(k in transforms for k in keys):continue
        composed=transforms[b,c]@transforms[a,b];direct=transforms[a,c]
        translation=float(np.linalg.norm(composed[:3,3]-direct[:3,3]))
        rotation=float(Rotation.from_matrix(composed[:3,:3]@direct[:3,:3].T).magnitude())
        ok=translation<=CONFIG['cycle_translation_m'] and rotation<=CONFIG['cycle_rotation_rad']
        cycles.append({'keyframes':[a,b,c],'translation_m':translation,'rotation_rad':rotation,'consistent':ok})
        (passed if ok else failed).update(keys)
    return {'cycles':cycles,'qualified_pairs':[list(k) for k in sorted(passed-failed)],
            'rule':'at least one accepted triangle and no failing accepted triangle per pair; repeated images are correlated'}


def identity_evidence(snapshot, pair_reports, window_keyframes):
    """Aggregate distinct pairs, then reject transitive co-observation conflicts.

    Pair acceptance alone is not identity. Three-view repeated evidence and
    one-observation-per-keyframe compatibility are necessary, still insufficient.
    The output is diagnostic; no ID or coordinate is rewritten.
    """
    evidence=defaultdict(set);tracks=defaultdict(set)
    for o in snapshot.observations:tracks[o.point_id].add(o.keyframe_id)
    allowed=set(window_keyframes)
    for report in pair_reports:
        if not report['verification']['accepted']:continue
        for rec in report['records']:
            if not rec['pair_geometric_inlier']:continue
            a,b=rec['source_point_id'],rec['target_point_id']
            if a==b:continue
            if a not in snapshot.points or b not in snapshot.points:raise ValueError('unknown point ID')
            i,j=rec['source_keyframe'],rec['target_keyframe']
            if i not in tracks[a] or j not in tracks[b]:raise ValueError('invalid evidence ownership')
            if i in allowed and j in allowed:evidence[tuple(sorted((a,b)))].add(tuple(sorted((i,j))))
    edges=[];parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for (a,b),views in sorted(evidence.items()):
        frames=set(k for view in views for k in view)
        repeated=(len(views)>=CONFIG['minimum_pair_views'] and len(frames)>=CONFIG['minimum_keyframe_views'])
        compatible=not bool(tracks[a]&tracks[b])
        edge={'point_ids':[a,b],'view_pairs':[list(v) for v in sorted(views)],
              'distinct_keyframes':len(frames),'repeated_support':repeated,
              'pairwise_ownership_compatible':compatible,'candidate':False}
        edges.append(edge)
        if repeated and compatible:parent[find(b)]=find(a)
    groups=defaultdict(list)
    for pid in list(parent):groups[find(pid)].append(pid)
    clusters=[]
    for ids in sorted(groups.values()):
        frame_owner={};conflicts=set()
        for pid in ids:
            for kid in tracks[pid]:
                if kid in frame_owner:conflicts.add(kid)
                frame_owner[kid]=pid
        clusters.append({'point_ids':sorted(ids),'ownership_conflict_keyframes':sorted(conflicts),
                         'candidate':not bool(conflicts)})
    viable={pid for c in clusters if c['candidate'] for pid in c['point_ids']}
    for edge in edges:
        edge['candidate']=bool(edge['repeated_support'] and edge['pairwise_ownership_compatible']
                               and set(edge['point_ids'])<=viable)
    return {'edges':edges,'clusters':clusters,'candidate_edges':sum(e['candidate'] for e in edges),
            'candidate_clusters':sum(c['candidate'] for c in clusters),
            'scope':'repeated pairwise geometric evidence only; no independent identity/accuracy certificate or map mutation'}
