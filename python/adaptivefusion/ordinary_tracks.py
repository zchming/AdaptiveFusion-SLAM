"""Propagate cached feature seeds through ordinary images, without re-detection."""
import numpy as np
from .landmark_identity import _flow


def track_chain(images, seed_uv):
    """Each adjacent step uses the unchanged LK/1px forward-backward gate.

    Lost tracks never reactivate. Depth/rigid gates are applied at the original
    endpoints by audit_pair, exactly as in the direct-keyframe control. No
    intermediate depth gate, pose prior or new feature detection is added.
    """
    if len(images)<2:raise ValueError('at least two ordered images required')
    shape=images[0].shape
    if any(im.dtype!=np.uint8 or im.ndim!=2 or im.shape!=shape for im in images):
        raise ValueError('native gray image type/size mismatch')
    points=np.asarray(seed_uv,np.float32).reshape(-1,2).copy()
    if not np.isfinite(points).all():raise ValueError('non-finite seed')
    valid=np.ones(len(points),bool);history=[]
    for first,second in zip(images,images[1:]):
        indices=np.flatnonzero(valid)
        if len(indices):
            moved,ok=_flow(first,second,points[indices])
            points[indices]=moved
            valid[indices]=ok
        history.append(int(valid.sum()))
    return points,valid,history


def chain_endpoints(snapshot, first, second, images):
    """Image order must be from first's source frame through second's, inclusive."""
    if not 0<=first<second<len(snapshot.keyframes):raise ValueError('ordered distinct keyframes required')
    expected=snapshot.keyframes[second].source_frame_id-snapshot.keyframes[first].source_frame_id+1
    if len(images)!=expected:raise ValueError('missing or skipped ordinary frame')
    seeds=[]
    for kid in [first,second]:
        obs=sorted([o for o in snapshot.observations if o.keyframe_id==kid],key=lambda o:o.point_id)
        seeds.append(np.array([[o.u,o.v] for o in obs],np.float32).reshape(-1,2))
    forward,ok_a,audit_a=track_chain(images,seeds[0])
    backward,ok_b,audit_b=track_chain(images[::-1],seeds[1])
    return (forward,ok_a,backward,ok_b),{'forward_survivors_per_step':audit_a,
                                       'backward_survivors_per_step':audit_b,
                                       'ordinary_intermediate_images':len(images)-2,
                                       'forward_seeds':len(seeds[0]),'backward_seeds':len(seeds[1])}
