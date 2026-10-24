"""Output-state audit for full-cache offline BA; no optimization or ground truth.

Recompute geometry directly from exported T_wc matrices, without the optimizer's
pose vectors, residual function or history-correction implementation. This is a
consistency audit, not a certificate of observation identity or pose accuracy.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def geometry_metrics(snapshot, active):
    active = set(active)
    counts = {}
    for o in snapshot.observations:
        counts[o.point_id] = counts.get(o.point_id, 0)+1
    points = {pid for pid, p in snapshot.points.items()
              if p.host in active and counts[pid] >= snapshot.config.minimum_point_observations}
    obs = [o for o in snapshot.observations if o.point_id in points]
    if not obs or any(o.keyframe_id not in active for o in obs):
        raise ValueError('invalid active observation scope')
    poses = np.array([snapshot.keyframes[o.keyframe_id].t_wc for o in obs])
    xyz = np.array([snapshot.points[o.point_id].xyz for o in obs])
    # Row-vector world-to-camera transform, independently of BA parameterization.
    pc = np.einsum('nji,nj->ni', poses[:, :3, :3], xyz-poses[:, :3, 3])
    if not np.isfinite(pc).all() or np.any(pc[:, 2] <= 1e-8):
        raise ValueError('nonpositive active geometry')
    fx, fy, cx, cy = snapshot.camera
    pixel = pc[:, :2]/pc[:, 2, None]*[fx, fy]+[cx, cy]-np.array([[o.u, o.v] for o in obs])
    pixel_norm = np.linalg.norm(pixel, axis=1)
    depth_mask = np.array([o.raw_depth > 0 for o in obs])
    depth = (pc[depth_mask, 2]-np.array([o.raw_depth for o in obs])[depth_mask]/snapshot.config.depth_scale)*snapshot.config.depth_residual_weight
    delta = snapshot.config.huber_delta_pixels
    def rho(norm):
        return np.where(norm <= delta, norm**2, 2*delta*norm-delta**2)
    return dict(robust_cost=float(.5*(rho(pixel_norm).sum()+rho(np.abs(depth)).sum())),
                pixel_rmse=float(np.sqrt(np.mean(pixel_norm**2))),
                weighted_depth_rmse=float(np.sqrt(np.mean(depth**2))) if len(depth) else None,
                active_points=len(points), active_observations=len(obs))


def audit_geometry(before, after, protocol, report):
    before.validate(); after.validate()
    if (before.config != after.config or before.source_revision != after.source_revision
            or before.state_version != after.state_version or not np.array_equal(before.camera, after.camera)
            or set(before.points) != set(after.points) or before.observations != after.observations
            or len(before.keyframes) != len(after.keyframes)):
        raise ValueError('immutable geometry/schema changed')
    if before.config.window_size != len(before.keyframes) or protocol['holdout_modulus'] != 0:
        raise ValueError('audit supports full-cache, no-holdout reference only')
    for a, b in zip(before.keyframes, after.keyframes):
        if (a.id, a.source_frame_id, a.timestamp, a.width, a.height) != (b.id, b.source_frame_id, b.timestamp, b.width, b.height):
            raise ValueError('keyframe metadata changed')
    if any(before.points[p].host != after.points[p].host for p in before.points):
        raise ValueError('landmark host changed')
    active = protocol['active_keyframe_ids']
    if not active or active[0] != 0 or active != sorted(set(active)) or not set(active) <= set(range(len(before.keyframes))):
        raise ValueError('invalid active scope')
    if protocol['component_policy'] not in ['all', 'anchor'] or (protocol['component_policy'] == 'all' and len(active) != len(before.keyframes)):
        raise ValueError('invalid component policy')
    if protocol['truth_used'] is not False or protocol['online'] is not False:
        raise ValueError('not an offline truth-free reference')
    if report['diagnostics']['active_keyframe_ids'] != active:
        raise ValueError('protocol/report active scope mismatch')
    status = report['status']
    accepted = report['accepted']
    if (type(accepted) is not bool or status not in ['accepted', 'accepted_partial', 'rejected']
            or accepted != (status != 'rejected')
            or (accepted and (status == 'accepted_partial') != (len(active) < len(before.keyframes)))):
        raise ValueError('ambiguous completion status')
    if not np.array_equal(before.keyframes[0].t_wc, after.keyframes[0].t_wc):
        raise ValueError('fixed anchor moved')
    for a, b in zip(before.keyframes, after.keyframes):
        if (not accepted or a.id not in active) and not np.array_equal(a.t_wc, b.t_wc):
            raise ValueError('rollback/excluded pose changed')
    for pid, a in before.points.items():
        if (not accepted or a.host not in active) and not np.array_equal(a.xyz, after.points[pid].xyz):
            raise ValueError('rollback/excluded point changed')
    initial, final = geometry_metrics(before, active), geometry_metrics(after, active)
    for label, measured in [('initial', initial), ('final', final)]:
        for key, claimed in report[label].items():
            if key not in ['robust_cost', 'pixel_rmse', 'weighted_depth_rmse']:
                raise ValueError('unknown objective metric')
            if claimed is None:
                if measured[key] is not None: raise ValueError('metric availability mismatch')
            elif not np.isfinite(claimed) or not np.isclose(claimed, measured[key], rtol=1e-8, atol=1e-7):
                raise ValueError('reported objective mismatch: '+label+'/'+key)
    if accepted and (not report['initial'] or not report['final'] or final['robust_cost'] > initial['robust_cost']+max(1e-7, initial['robust_cost']*1e-8)):
        raise ValueError('accepted objective not decreasing')
    if (initial['active_points'] != protocol['active_points']
            or initial['active_observations'] != protocol['active_observations']):
        raise ValueError('active geometry count mismatch')
    return dict(status=status, accepted=accepted, active_keyframes=len(active),
                total_keyframes=len(before.keyframes), initial=initial, final=final,
                exact_rollback=not accepted, fixed_anchor_preserved=True,
                accuracy_certified=False)


def audit_history(before, after, frames, timestamps, trajectory, *, accepted, active):
    valid = [f for f in frames if f.valid]
    if (len(timestamps) != len(valid) or np.shape(trajectory) != (len(valid), 4, 4)
            or not np.array_equal(timestamps, [f.timestamp for f in valid])):
        raise ValueError('valid trajectory coverage/timestamps changed')
    raw = {f.id: f for f in valid}
    corrections = []
    for a, b in zip(before.keyframes, after.keyframes):
        if a.source_frame_id not in raw or not np.array_equal(a.t_wc, raw[a.source_frame_id].t_wc):
            raise ValueError('input keyframe differs from raw frame cache')
        f = raw[a.source_frame_id]
        if f.timestamp != a.timestamp or f.anchor_id != a.id:
            raise ValueError('keyframe/frame ownership mismatch')
        inverse = np.eye(4); inverse[:3, :3] = a.t_wc[:3, :3].T
        inverse[:3, 3] = -inverse[:3, :3]@a.t_wc[:3, 3]
        corrections.append(b.t_wc@inverse)
    corrections = np.array(corrections); times = np.array([k.timestamp for k in before.keyframes])
    maximum = 0.
    for frame, observed in zip(valid, trajectory):
        if (frame.anchor_id not in range(len(before.keyframes))
                or before.keyframes[frame.anchor_id].source_frame_id > frame.id):
            raise ValueError('invalid/future raw anchor')
        expected = frame.t_wc
        if accepted and frame.anchor_id in active:
            stamp = frame.timestamp
            if stamp >= times[-1]: correction = corrections[-1]
            elif stamp < times[0]: correction = np.eye(4)
            else:
                i = int(np.searchsorted(times, stamp, side='right'))-1
                alpha = (stamp-times[i])/(times[i+1]-times[i])
                a, b = corrections[i], corrections[i+1]
                correction = np.eye(4)
                # Geodesic matrix interpolation instead of production Slerp.
                rotvec = Rotation.from_matrix(a[:3, :3].T@b[:3, :3]).as_rotvec()
                correction[:3, :3] = a[:3, :3]@Rotation.from_rotvec(alpha*rotvec).as_matrix()
                correction[:3, 3] = (1-alpha)*a[:3, 3]+alpha*b[:3, 3]
            expected = correction@frame.t_wc
        error = float(np.max(np.abs(expected-observed)))
        maximum = max(maximum, error)
        if not np.isfinite(error) or error > 1e-8:
            raise ValueError('trajectory/state inconsistency at frame '+str(frame.id))
    return dict(valid_poses=len(valid), input_frames=len(frames), lost_frames_preserved=len(frames)-len(valid),
                maximum_pose_matrix_difference=maximum, interpretation='offline history deformation only')
