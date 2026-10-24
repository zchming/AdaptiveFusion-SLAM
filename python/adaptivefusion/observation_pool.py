"""Recover cached singleton observations for matching audits, never BA insertion."""
from collections import Counter
import copy
import json
from pathlib import Path
import numpy as np
from .state import file_sha256, load_snapshot


def observation_pool(cache, baseline):
    """Union existing before-snapshot observations without BA's >=2-view filter.

    Baseline poses, existing points and observations are preserved exactly.
    Extra point birth positions are provenance only; matching uses image/depth.
    This read-only candidate view must not be treated as an optimized BA graph.
    """
    cache=Path(cache)
    marker=json.loads((cache/'complete.json').read_text())
    if marker.get('risk_enabled') or marker.get('online_writeback'):
        raise ValueError('requires baseline offline cache')
    if marker.get('processed_frames',0)<=baseline.keyframes[-1].source_frame_id:
        raise ValueError('incomplete cache')
    paths=sorted(cache.glob('*.before.snapshot'),key=lambda p:int(p.name.split('.')[0]))
    if not paths:raise ValueError('missing before snapshots')
    points={};observations={};manifest={};seen_keyframes=set()
    for path in paths:
        part=load_snapshot(path)
        if (part.source_revision!=baseline.source_revision or not np.array_equal(part.camera,baseline.camera)
                or part.config.depth_scale!=baseline.config.depth_scale):
            raise ValueError('incompatible snapshot source/camera/depth')
        for k in part.keyframes:
            if k.id>=len(baseline.keyframes):raise ValueError('extra cached keyframe')
            expected=baseline.keyframes[k.id]
            if (k.source_frame_id,k.timestamp,k.width,k.height)!=(expected.source_frame_id,expected.timestamp,expected.width,expected.height):
                raise ValueError('incompatible cached keyframe metadata')
            seen_keyframes.add(k.id)
        for pid,p in part.points.items():
            if pid in points and points[pid].host!=p.host:raise ValueError('point host changed')
            if pid not in points:points[pid]=p
        for o in part.observations:
            pair=(o.point_id,o.keyframe_id)
            if pair in observations and observations[pair]!=o:raise ValueError('cached observation changed')
            observations[pair]=o
        manifest[path.name]=file_sha256(path)
    if seen_keyframes!=set(range(len(baseline.keyframes))):raise ValueError('incomplete keyframe coverage')
    counts=Counter(pid for pid,kid in observations)
    expected_points={pid for pid,n in counts.items() if n>=baseline.config.minimum_point_observations}
    if set(baseline.points)!=expected_points:raise ValueError('baseline does not match shared cached points')
    for o in baseline.observations:
        if observations.get((o.point_id,o.keyframe_id))!=o:raise ValueError('baseline observation mismatch')
    if len(baseline.observations)!=sum(n for pid,n in counts.items() if pid in baseline.points):
        raise ValueError('missing baseline observations')
    view=baseline.clone()
    for pid,p in points.items():
        if pid not in view.points:view.points[pid]=copy.deepcopy(p)
    view.observations=sorted(observations.values(),key=lambda o:(o.keyframe_id,o.feature_index))
    view.validate()
    before=Counter(o.keyframe_id for o in baseline.observations)
    after=Counter(o.keyframe_id for o in view.observations)
    return view,{'purpose':'read-only candidate observations; no BA insertion or map mutation',
                 'cached_before_snapshot_sha256':manifest,
                 'baseline_points':len(baseline.points),'candidate_points':len(view.points),
                 'baseline_observations':len(baseline.observations),'candidate_observations':len(view.observations),
                 'singleton_points_recovered':sum(n==1 for n in counts.values()),
                 'per_keyframe':[{'keyframe':k.id,'baseline_observations':before[k.id],
                                  'candidate_observations':after[k.id]} for k in baseline.keyframes]}
