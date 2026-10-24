"""Offline replacement diagnostics; no truth, fitting or decision thresholds."""
import numpy as np
from scipy.spatial.transform import Rotation


def fit_partition(pixels):
    """Fixed source-image 4x4 checkerboard, selected before F and depth tests."""
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    cells = np.clip((pixels/[640, 480]*4).astype(int), 0, 3)
    return cells.sum(axis=1) % 2 == 0


def closure(forward, reverse):
    cycle = np.asarray(reverse) @ np.asarray(forward)
    return dict(translation_m=float(np.linalg.norm(cycle[:3, 3])),
                rotation_deg=float(np.rad2deg(Rotation.from_matrix(cycle[:3, :3]).magnitude())))


def common_check(raw, candidate, xyz, pixels, camera):
    """Score identical unused source observations with positive depth in both poses."""
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    moved = [xyz @ t[:3, :3].T+t[:3, 3] for t in [np.asarray(raw), np.asarray(candidate)]]
    common = np.ones(len(xyz), bool)
    for p in moved:
        common &= np.isfinite(p).all(axis=1) & (p[:, 2] > 1e-6)
    out = dict(count=len(xyz), positive_count=int(common.sum()))
    for name, p in zip(['raw', 'candidate'], moved):
        e = np.linalg.norm(p[common, :2]/p[common, 2, None]*camera[:2]+camera[2:]-pixels[common], axis=1)
        out[name] = dict(median_px=float(np.median(e)) if len(e) else None,
                         p95_px=float(np.quantile(e, .95)) if len(e) else None)
    # Included only to reuse the existing sealed evaluator's output schema.
    out.update(out['candidate'])
    return out
