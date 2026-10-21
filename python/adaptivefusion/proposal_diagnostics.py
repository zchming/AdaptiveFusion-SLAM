"""Descriptive diagnostics for immutable RGB-D proposals; no pose fitting.

Truth is used only to describe disagreement under the existing pinhole model.
Neither epipolar error nor the support spectrum certifies match identity or rank.
"""
import numpy as np
from .geometry import check_pose, inverse, skew


def summary(values):
    a = np.asarray(values, float)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError('finite scalar measurements required')
    return dict(count=len(a), median=float(np.median(a)) if len(a) else None,
                p95=float(np.quantile(a, .95)) if len(a) else None)


def support_geometry(xyz, uv, width, height):
    xyz, uv = np.asarray(xyz, float), np.asarray(uv, float)
    if (xyz.ndim != 2 or xyz.shape[1:] != (3,) or uv.shape != (len(xyz), 2)
            or not np.isfinite(xyz).all() or not np.isfinite(uv).all()
            or min(width, height) <= 0 or np.any(uv < 0)
            or np.any(uv >= [width, height])):
        raise ValueError('invalid support')
    if not len(xyz):
        return {'count': 0}
    center = xyz.mean(0)
    singular = np.linalg.svd(xyz-center, compute_uv=False) / np.sqrt(len(xyz))
    cells = np.floor(uv/[width, height]*4).astype(int)
    return dict(count=len(xyz), centroid_m=center.tolist(),
                centered_rms_axes_m=singular.tolist(),
                pixel_span=np.ptp(uv, axis=0).tolist(),
                occupied_4x4_cells=len(set(map(tuple, cells))),
                depth_m=summary(xyz[:, 2]))


def fixed_model_errors(transform, xyz, target_xyz, target_uv, camera):
    transform = check_pose(transform)
    xyz, target_xyz, target_uv = map(lambda a: np.asarray(a, float),
                                     [xyz, target_xyz, target_uv])
    if (xyz.ndim != 2 or xyz.shape[1:] != (3,) or target_xyz.shape != xyz.shape
            or target_uv.shape != (len(xyz), 2)
            or not all(np.isfinite(a).all() for a in [xyz, target_xyz, target_uv])):
        raise ValueError('invalid correspondences')
    fx, fy, cx, cy = camera
    if not np.isfinite(camera).all() or min(fx, fy) <= 0:
        raise ValueError('invalid camera')
    predicted = xyz@transform[:3, :3].T+transform[:3, 3]
    positive = predicted[:, 2] > 1e-8
    pixels = np.full(len(xyz), np.nan)
    pixels[positive] = np.linalg.norm(
        predicted[positive, :2]/predicted[positive, 2, None]*[fx, fy]+[cx, cy]
        - target_uv[positive], axis=1)
    metric = np.linalg.norm(predicted-target_xyz, axis=1)
    return dict(metric_m=summary(metric), reprojection_px=summary(pixels[positive]),
                nonpositive_predictions=int((~positive).sum()),
                metric_per_record_m=metric.tolist(),
                reprojection_per_record_px=[float(v) if np.isfinite(v) else None for v in pixels])


def epipolar_errors(transform, source_uv, target_uv, camera):
    """Symmetric point-to-epipolar-line distance, existing zero-distortion model."""
    t = check_pose(transform)
    a, b = np.asarray(source_uv, float), np.asarray(target_uv, float)
    if a.ndim != 2 or a.shape[1:] != (2,) or b.shape != a.shape or not np.isfinite([a, b]).all():
        raise ValueError('invalid pixels')
    fx, fy, cx, cy = camera
    if not np.isfinite(camera).all() or min(fx, fy) <= 0:
        raise ValueError('invalid camera')
    if np.linalg.norm(t[:3, 3]) < 1e-10:
        return dict(status='zero_translation_unassessed', errors_px=[], measured=summary([]))
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    ki = np.linalg.inv(k)
    f = ki.T@skew(t[:3, 3])@t[:3, :3]@ki
    aa, bb = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
    target_lines, source_lines = aa@f.T, bb@f
    da, db = (target_lines[:, :2]**2).sum(1), (source_lines[:, :2]**2).sum(1)
    good = (da > 1e-30) & (db > 1e-30)
    errors = np.full(len(a), np.nan)
    numerator = (bb*target_lines).sum(1)**2
    errors[good] = np.sqrt(.5*numerator[good]*(1/da[good]+1/db[good]))
    return dict(status='evaluated', unassessed=int((~good).sum()),
                errors_px=[float(v) if np.isfinite(v) else None for v in errors],
                measured=summary(errors[good]))


def centroid_decomposition(candidate, reference, xyz):
    """Exact delta action at support centroid = rotation lever arm + delta t.

    A large camera-origin translation difference can compensate angular error
    over a narrow distant support. This is descriptive, not a causal proof.
    """
    c, r = check_pose(candidate), check_pose(reference)
    xyz = np.asarray(xyz, float)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('nonempty support required')
    lever = (c[:3, :3]-r[:3, :3])@xyz.mean(0)
    delta = c[:3, 3]-r[:3, 3]
    return dict(rotation_lever_vector_m=lever.tolist(), translation_delta_vector_m=delta.tolist(),
                centroid_action_delta_vector_m=(lever+delta).tolist(),
                rotation_lever_m=float(np.linalg.norm(lever)),
                translation_delta_m=float(np.linalg.norm(delta)),
                centroid_action_delta_m=float(np.linalg.norm(lever+delta)))
