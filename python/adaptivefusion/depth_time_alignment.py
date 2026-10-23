"""Source depth registration using only the supplied estimated T_wc history."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .geometry import check_pose, inverse


def estimated_pose(frames, timestamp, maximum_gap_s=None):
    times = np.array([f.timestamp for f in frames])
    j = int(np.searchsorted(times, timestamp))
    if j < len(frames) and abs(times[j]-timestamp) <= 1e-9:
        if not frames[j].valid:
            raise ValueError('estimated frame is lost')
        return frames[j].t_wc.copy(), [frames[j].id]
    if j == 0 or j == len(frames):
        raise ValueError('estimated trajectory does not bracket sensor timestamp')
    a, b = frames[j-1], frames[j]
    if not a.valid or not b.valid:
        raise ValueError('lost frame in estimated motion bracket')
    if maximum_gap_s is not None and b.timestamp-a.timestamp > maximum_gap_s:
        raise ValueError('estimated motion bracket exceeds configured gap')
    alpha = (timestamp-a.timestamp)/(b.timestamp-a.timestamp)
    pose = np.eye(4)
    pose[:3, :3] = Slerp([a.timestamp, b.timestamp], Rotation.from_matrix(
        np.array([a.t_wc[:3, :3], b.t_wc[:3, :3]])))([timestamp]).as_matrix()[0]
    pose[:3, 3] = (1-alpha)*a.t_wc[:3, 3]+alpha*b.t_wc[:3, 3]
    return check_pose(pose), [a.id, b.id]


def sensor_motion(frames, rgb_timestamp, depth_timestamp):
    rgb, rgb_ids = estimated_pose(frames, rgb_timestamp)
    depth, depth_ids = estimated_pose(frames, depth_timestamp)
    return inverse(rgb)@depth, dict(rgb_frame_ids=rgb_ids, depth_frame_ids=depth_ids,
        rgb_depth_dt_s=depth_timestamp-rgb_timestamp,
        rgb_bracket_span_s=frames[rgb_ids[-1]].timestamp-frames[rgb_ids[0]].timestamp,
        depth_bracket_span_s=frames[depth_ids[-1]].timestamp-frames[depth_ids[0]].timestamp)


def register_source_depth(depth, camera, t_rgb_depth, scale=5000.):
    """Nearest-pixel forward projection, nearest-Z buffer, no dilation or filling.

    Input/output uint16 use the original 1/5000m units. All geometry uses metres;
    output rounding is the only depth quantization. Existing spatial registration
    is retained: t_rgb_depth is motion between capture times, not IR extrinsics.
    """
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError('native uint16 depth required')
    T = check_pose(t_rgb_depth); fx, fy, cx, cy = camera
    y, x = np.nonzero(depth)
    z = depth[y, x].astype(float)/scale
    p = np.c_[(x-cx)/fx*z, (y-cy)/fy*z, z]
    q = p@T[:3, :3].T+T[:3, 3]
    positive = np.isfinite(q).all(1) & (q[:, 2] > 0)
    q = q[positive]
    ij = np.floor(q[:, :2]/q[:, 2, None]*[fx, fy]+[cx, cy]+.5).astype(np.int64)
    height, width = depth.shape
    inside = (ij[:, 0] >= 0)&(ij[:, 0] < width)&(ij[:, 1] >= 0)&(ij[:, 1] < height)
    ij, q = ij[inside], q[inside]
    flat = np.full(depth.size, np.inf)
    np.minimum.at(flat, ij[:, 1]*width+ij[:, 0], q[:, 2])
    valid = np.isfinite(flat)
    quantized = np.zeros(depth.size, np.float64)
    quantized[valid] = np.floor(flat[valid]*scale+.5)
    valid &= (quantized >= 1)&(quantized <= 65535)
    result = np.zeros(depth.size, np.uint16)
    result[valid] = quantized[valid].astype(np.uint16)
    return result.reshape(depth.shape), dict(nonzero_input=len(z), positive_transformed=int(positive.sum()),
        inside_projections=len(q), output_nonzero=int(valid.sum()),
        collision_count=len(q)-int(np.isfinite(flat).sum()))
