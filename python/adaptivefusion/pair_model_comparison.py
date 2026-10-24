"""Matched six-DoF pair objectives, development only; no GT or map updates."""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .geometry import check_pose, block_huber

CONFIG = dict(minimum_fit=6, pixel_sigma=2., metric_sigma_m=.01,
              huber_delta=2., maximum_nfev=100, tolerance=1e-10)


def refine_pair(source, target_pixels, camera, initial, *, model, target_points=None):
    """Fit the same supplied source rows; no sampling, filtering or target search.

    3D-2D never reads target_points. Common initialization and a common robust
    block policy control sampling/initialization differences; objective units and
    noise assumptions still differ. Any failed solve returns the initial pose.
    """
    start = time.perf_counter()
    initial = check_pose(initial).copy()
    result = dict(applied=False, reason='invalid input', t_target_source=initial.tolist(),
                  nfev=0, initial_cost=None, final_cost=None)
    try:
        a, pixels, k = np.asarray(source, float), np.asarray(target_pixels, float), np.asarray(camera, float)
        if (model not in ['3d3d', '3d2d'] or a.ndim != 2 or a.shape[1:] != (3,)
                or pixels.shape != (len(a), 2) or k.shape != (4,)
                or not all(np.isfinite(v).all() for v in [a, pixels, k])
                or min(k[:2]) <= 0 or np.any(a[:, 2] <= 0)):
            raise ValueError('invalid input')
        if len(a) < CONFIG['minimum_fit']:
            raise ValueError('insufficient fit support')
        singular = np.linalg.svd(a-a.mean(0), compute_uv=False)
        if singular[0] <= 1e-10 or singular[1] <= 1e-8*singular[0]:
            raise ValueError('collinear source support')
        b = None
        if model == '3d3d':
            b = np.asarray(target_points, float)
            if b.shape != a.shape or not np.isfinite(b).all() or np.any(b[:, 2] <= 0):
                raise ValueError('invalid target depth')

        def pose(x):
            t = initial.copy()
            t[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()@initial[:3, :3]
            t[:3, 3] = initial[:3, 3]+x[3:]
            return t

        def residual(x):
            t = pose(x)
            p = a@t[:3, :3].T+t[:3, 3]
            if not np.isfinite(p).all() or np.any(p[:, 2] <= 1e-6):
                raise ValueError('nonpositive prediction; roll back')
            if model == '3d3d':
                r = (p-b)/CONFIG['metric_sigma_m']
            else:
                r = (p[:, :2]/p[:, 2, None]*k[:2]+k[2:]-pixels)/CONFIG['pixel_sigma']
            return block_huber(r, CONFIG['huber_delta']).ravel()

        zero = np.zeros(6)
        r0 = residual(zero)
        result['initial_cost'] = float(.5*r0@r0)
        fit = least_squares(residual, zero, method='trf', jac='3-point',
                            max_nfev=CONFIG['maximum_nfev'],
                            ftol=CONFIG['tolerance'], xtol=CONFIG['tolerance'],
                            gtol=CONFIG['tolerance'])
        result.update(nfev=int(fit.nfev), final_cost=float(fit.cost),
                      solver_status=int(fit.status), solver_message=str(fit.message))
        if not fit.success:
            raise ValueError('solver did not converge')
        if not np.isfinite(fit.x).all() or not np.isfinite(fit.cost) or fit.cost > result['initial_cost']+1e-10:
            raise ValueError('nonfinite or increased cost')
        singular = np.linalg.svd(fit.jac, compute_uv=False)
        if len(singular) < 6 or singular[-1] <= 1e-10*singular[0]:
            raise ValueError('rank-deficient local Jacobian')
        transform = check_pose(pose(fit.x))
        result.update(applied=True, reason='converged; not accuracy certificate',
                      t_target_source=transform.tolist(), jacobian_singular_values=singular.tolist())
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        result['reason'] = str(error)
    result['wall_s'] = time.perf_counter()-start
    return result
