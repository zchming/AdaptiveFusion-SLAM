"""State-consistent replay of fixed cached observations; never live tracking."""
import numpy as np
import copy
from .geometry import inverse,project
from .state import load_snapshot
from .local_ba import reanchor_singletons
from .trajectory import correct_history_continuous


class CachedState:
    def __init__(self,frames):
        self.raw={f.id:f for f in frames if f.valid}
        self.poses={};self.points={};self.hosts={}

    def prepare(self,snapshot):
        result=snapshot.clone()
        previous=None
        for key in result.keyframes:
            if key.source_frame_id not in self.raw:raise ValueError('missing raw keyframe')
            if key.id in self.poses:
                key.t_wc=self.poses[key.id].copy()
            elif previous is None:
                key.t_wc=self.raw[key.source_frame_id].t_wc.copy()
            else:
                # Reuse measured raw relative motion, anchored to corrected state.
                relative=inverse(self.raw[previous.source_frame_id].t_wc)@self.raw[key.source_frame_id].t_wc
                key.t_wc=previous.t_wc@relative
            previous=key
        for pid,p in result.points.items():
            if pid in self.points:
                if p.host!=self.hosts[pid]:raise ValueError('point host changed')
                p.xyz=self.points[pid].copy()
            else:
                # C++ initializes a landmark in the raw host pose at birth.
                # Preserve that exact local XYZ, including its depth sampling.
                host=snapshot.keyframes[p.host]
                local=inverse(self.raw[host.source_frame_id].t_wc)@np.r_[p.xyz,1]
                p.xyz=(result.keyframes[p.host].t_wc@local)[:3]
        return result.validate()

    def commit(self,before,after,repair=True):
        if repair:after,_=reanchor_singletons(before,after)
        for k in after.keyframes:self.poses[k.id]=k.t_wc.copy()
        for pid,p in after.points.items():
            self.points[pid]=p.xyz.copy();self.hosts[pid]=p.host
        return after


def full_history(frames,template,poses,interpolate=True):
    """Final deformation through all keyframes, preserving every valid timestamp."""
    before=template.clone();after=template.clone()
    before.config.window_size=after.config.window_size=max(2,len(template.keyframes))
    raw={f.id:f for f in frames if f.valid}
    for k in before.keyframes:k.t_wc=raw[k.source_frame_id].t_wc.copy()
    for k in after.keyframes:k.t_wc=poses.get(k.id,raw[k.source_frame_id].t_wc).copy()
    return correct_history_continuous(frames,before,after,interpolate=interpolate)


def merge_cached_problem(paths,frames,initialization='raw',initial_poses=None):
    """Offline batch diagnostic: union actual observations, never create matches."""
    raw={f.id:f for f in frames if f.valid}
    points={};observations={};last=None
    for path in paths:
        snapshot=path if hasattr(path,'keyframes') else load_snapshot(path)
        last=snapshot
        for pid,p in snapshot.points.items():
            if pid not in points:
                # First occurrence is the point's birth (before its first BA).
                points[pid]=copy.deepcopy(p)
        for o in snapshot.observations:
            key=(o.point_id,o.keyframe_id)
            if key in observations and observations[key]!=o:raise ValueError('cached observation changed')
            observations[key]=o
    if last is None:raise ValueError('no cached problems')
    result=last.clone()
    counts={}
    for pid,_ in observations:counts[pid]=counts.get(pid,0)+1
    result.points={pid:p for pid,p in points.items() if counts.get(pid,0)>=result.config.minimum_point_observations}
    result.observations=[o for o in observations.values() if o.point_id in result.points]
    result.config.window_size=len(result.keyframes)
    if initialization=='raw':
        for k in result.keyframes:k.t_wc=raw[k.source_frame_id].t_wc.copy()
    elif initialization=='provided':
        if initial_poses is None:raise ValueError('initial poses required')
        for k in result.keyframes:k.t_wc=initial_poses[k.id].copy()
        for pid,p in result.points.items():
            host=result.keyframes[p.host]
            local=inverse(raw[host.source_frame_id].t_wc)@np.r_[p.xyz,1]
            p.xyz=(host.t_wc@local)[:3]
    else:raise ValueError('invalid batch initialization')
    return result.validate()


def split_landmark_holdout(snapshot,modulus=10):
    """Hold out whole landmarks, not one observation of an optimized point."""
    if modulus<2:raise ValueError('holdout modulus must be >=2')
    held={pid for pid in snapshot.points if pid%modulus==0}
    if not held or held==set(snapshot.points):raise ValueError('empty training or holdout split')
    train=snapshot.clone()
    train.points={pid:p for pid,p in train.points.items() if pid not in held}
    train.observations=[o for o in train.observations if o.point_id not in held]
    return train.validate(),held


def heldout_geometry(original,poses,held):
    """Cross-camera validation from held-out host-local XYZ, without truth."""
    pixel_errors=[];depth_errors=[];invalid=0
    for o in original.observations:
        if o.point_id not in held:continue
        p=original.points[o.point_id]
        if o.keyframe_id==p.host:continue
        local=inverse(original.keyframes[p.host].t_wc)@np.r_[p.xyz,1]
        predicted=(inverse(poses[o.keyframe_id])@poses[p.host]@local)[:3]
        if predicted[2]<=1e-8 or not np.isfinite(predicted).all():invalid+=1;continue
        pixel_errors.append(float(np.linalg.norm(project(original.camera,predicted)-[o.u,o.v])))
        if o.raw_depth:depth_errors.append(float(predicted[2]-o.raw_depth/original.config.depth_scale))
    return {'heldout_landmarks':len(held),'cross_camera_observations':len(pixel_errors),
            'invalid_predictions':invalid,
            'pixel_rmse':float(np.sqrt(np.mean(np.square(pixel_errors)))) if pixel_errors else None,
            'pixel_median':float(np.median(pixel_errors)) if pixel_errors else None,
            'pixel_p95':float(np.percentile(pixel_errors,95)) if pixel_errors else None,
            'depth_rmse_m':float(np.sqrt(np.mean(np.square(depth_errors)))) if depth_errors else None}
