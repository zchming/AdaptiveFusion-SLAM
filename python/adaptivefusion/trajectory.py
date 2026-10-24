"""Offline anchor-relative history correction. Never fills a lost frame."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .geometry import check_pose, inverse


@dataclass
class FrameRecord:
    id: int
    timestamp: float
    valid: bool
    reference_id: int
    anchor_id: int
    t_wc: np.ndarray


def load_frames(path):
    rows = Path(path).read_text().splitlines()
    expected = 'frame_id timestamp valid reference_id anchor_id ' + ' '.join(f't{i}' for i in range(16))
    if not rows or rows[0] != expected: raise ValueError('invalid frame schema')
    frames, seen, last = [], set(), -np.inf
    for line in rows[1:]:
        row = line.split()
        if len(row) != 21: raise ValueError('invalid frame row')
        fid, ts, valid, ref, anchor = int(row[0]), float(row[1]), int(row[2]), int(row[3]), int(row[4])
        if (fid != len(frames) or not np.isfinite(ts) or ts <= last or valid not in [0, 1]
                or ref < -1 or anchor < -1 or (ref != -1 and ref not in seen)):
            raise ValueError('invalid frame metadata')
        pose = check_pose(np.array(list(map(float, row[5:]))).reshape(4, 4))
        frames.append(FrameRecord(fid, ts, bool(valid), ref, anchor, pose))
        if valid: seen.add(fid)
        last = ts
    return frames


def correct_history(frames, keyframes, corrected_poses):
    """T_wf' = T_wk' (T_wk_raw^-1 T_wf_raw).

    Raw anchors come from the original frame stream, never from an already
    optimized snapshot. Only explicitly supplied anchor IDs are corrected.
    """
    raw = {f.id: f for f in frames}
    anchors = {k.id: k for k in keyframes}
    if not set(corrected_poses) <= set(anchors): raise ValueError('unknown corrected anchor')
    for kid, pose in corrected_poses.items():
        check_pose(pose)
        source = anchors[kid].source_frame_id
        if source not in raw or not raw[source].valid or raw[source].anchor_id != kid:
            raise ValueError('missing original keyframe pose')
        if abs(raw[source].timestamp - anchors[kid].timestamp) > 1e-9:
            raise ValueError('anchor timestamp mismatch')
    result = []
    for f in frames:
        if not f.valid: continue
        if f.anchor_id not in corrected_poses:
            result.append((f.timestamp, f.t_wc.copy())); continue
        k = anchors[f.anchor_id]
        if k.source_frame_id > f.id: raise ValueError('future anchor used')
        relative = inverse(raw[k.source_frame_id].t_wc) @ f.t_wc
        result.append((f.timestamp, check_pose(corrected_poses[f.anchor_id] @ relative)))
    return result


def correct_history_continuous(frames,before,after,*,interpolate=True,carry=True):
    """Anchor the window to raw history, interpolate corrections, carry its tail.

    This is offline pose-history deformation, not additional frame optimization
    or live feedback. The old piecewise absolute-pose rule remains available.
    """
    before.validate();after.validate()
    if (before.state_version!=after.state_version or before.config!=after.config or
            len(before.keyframes)!=len(after.keyframes)):
        raise ValueError('window state mismatch')
    raw={f.id:f for f in frames if f.valid}
    begin=max(0,len(before.keyframes)-before.config.window_size)
    keys=before.keyframes[begin:]
    if not keys:raise ValueError('empty window')
    for a,b in zip(before.keyframes,after.keyframes):
        if (a.id,a.source_frame_id,a.timestamp)!=(b.id,b.source_frame_id,b.timestamp):
            raise ValueError('keyframe ownership mismatch')
    for k in keys:
        if k.source_frame_id not in raw or raw[k.source_frame_id].anchor_id!=k.id:
            raise ValueError('missing raw window anchor')
    if not np.allclose(keys[0].t_wc,after.keyframes[begin].t_wc,atol=1e-12,rtol=0):
        raise ValueError('fixed window anchor moved')
    gauge=raw[keys[0].source_frame_id].t_wc@inverse(keys[0].t_wc)
    corrections=[gauge@after.keyframes[k.id].t_wc@inverse(raw[k.source_frame_id].t_wc) for k in keys]
    # Exactly preserve the oldest anchor; numerical identity noise is unnecessary.
    corrections[0]=np.eye(4)
    times=np.array([k.timestamp for k in keys])
    rotation=Slerp(times,Rotation.from_matrix(np.array(corrections)[:,:3,:3])) if len(keys)>1 else None
    result=[]
    for f in frames:
        if not f.valid:continue
        if f.timestamp<times[0] or (f.timestamp>times[-1] and not carry):
            result.append((f.timestamp,f.t_wc.copy()));continue
        if f.timestamp>=times[-1]:correction=corrections[-1]
        elif interpolate:
            i=max(0,int(np.searchsorted(times,f.timestamp,side='right'))-1)
            alpha=(f.timestamp-times[i])/(times[i+1]-times[i])
            correction=np.eye(4);correction[:3,:3]=rotation([f.timestamp]).as_matrix()[0]
            correction[:3,3]=(1-alpha)*corrections[i][:3,3]+alpha*corrections[i+1][:3,3]
        else:
            i=max(0,int(np.searchsorted(times,f.timestamp,side='right'))-1)
            correction=corrections[i]
        result.append((f.timestamp,check_pose(correction@f.t_wc)))
    return result


def write_tum(path, poses):
    with Path(path).open('x') as out:
        for timestamp, pose in poses:
            check_pose(pose)
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()  # x y z w
            # Unix timestamps are ~1e9: 12 significant digits lose milliseconds.
            # Preserve the C++ TUM timestamp precision for fair GT association.
            out.write(f'{timestamp:.9f} ' + ' '.join(f'{x:.17g}' for x in [*pose[:3, 3], *q]) + '\n')
