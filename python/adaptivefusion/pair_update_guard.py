"""Development-only candidate update gate; uses validation pixels, never truth.

These pixels become decision inputs and are no longer independent validation.
A rejected update returns the original transform exactly; this is not tracking.
"""
import numpy as np
from .geometry import check_pose

# Derived from frozen V12 pixel sigma (2 px) and Huber delta (2).
CONFIG = dict(minimum_check_count=6, maximum_median_px=4.,
              allowed_median_increase_px=1e-9)


def _pixel_check(t, source, pixels, camera):
    predicted = source@t[:3, :3].T+t[:3, 3]
    positive = np.isfinite(predicted).all(1) & (predicted[:, 2] > 1e-6)
    if not positive.all():
        return dict(count=len(source), status='nonpositive_prediction', median_px=None)
    uv = predicted[:, :2]/predicted[:, 2, None]*camera[:2]+camera[2:]
    return dict(count=len(source), status='evaluated',
                median_px=float(np.median(np.linalg.norm(uv-pixels, axis=1))))


def guard_update(initial, candidate, source, pixels, camera, *, solver_applied):
    initial = check_pose(initial).copy()
    result = dict(accepted=False, reason='solver did not apply',
                  selected_t_target_source=initial.tolist(), candidate_check=None,
                  initial_check=None)
    if not solver_applied:
        return result
    try:
        candidate = check_pose(candidate)
        source, pixels, camera = map(lambda a: np.asarray(a, float), [source, pixels, camera])
        if (source.ndim != 2 or source.shape[1:] != (3,) or pixels.shape != (len(source), 2)
                or camera.shape != (4,) or not all(np.isfinite(a).all() for a in [source, pixels, camera])
                or min(camera[:2]) <= 0 or np.any(source[:, 2] <= 0)):
            raise ValueError('invalid check observations')
        if len(source) < CONFIG['minimum_check_count']:
            raise ValueError('insufficient check observations')
        base = _pixel_check(initial, source, pixels, camera)
        new = _pixel_check(candidate, source, pixels, camera)
        result.update(initial_check=base, candidate_check=new)
        if base['status'] != 'evaluated' or new['status'] != 'evaluated':
            raise ValueError('nonpositive projection; cannot assess update')
        if new['median_px'] > CONFIG['maximum_median_px']:
            raise ValueError('candidate pixel median exceeds frozen bound')
        if new['median_px'] > base['median_px']+CONFIG['allowed_median_increase_px']:
            raise ValueError('candidate worsens check-pixel median')
        result.update(accepted=True, reason='check passed; not an accuracy certificate',
                      selected_t_target_source=candidate.tolist())
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        result['reason'] = str(error)
    return result
