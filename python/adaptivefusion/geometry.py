"""Metric camera geometry. Stored poses are T_wc; BA vectors describe T_cw."""
import numpy as np
from scipy.spatial.transform import Rotation


def check_pose(pose):
    p = np.asarray(pose, dtype=float)
    if (p.shape != (4, 4) or not np.isfinite(p).all()
            or not np.allclose(p[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or abs(np.linalg.det(p[:3, :3]) - 1) > 1e-8):
        raise ValueError('invalid SE(3) pose')
    return p


def inverse(pose):
    p = check_pose(pose)
    q = np.eye(4)
    q[:3, :3] = p[:3, :3].T
    q[:3, 3] = -q[:3, :3] @ p[:3, 3]
    return q


def to_cw_vector(t_wc):
    t = inverse(t_wc)
    return np.r_[Rotation.from_matrix(t[:3, :3]).as_rotvec(), t[:3, 3]]


def from_cw_vector(vector):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    t[:3, 3] = vector[3:]
    return inverse(t)


def project(camera, points_camera):
    p = np.asarray(points_camera, float)
    if not np.isfinite(p).all() or np.any(p[..., 2] <= 1e-8):
        raise ValueError('non-finite or non-positive camera depth')
    fx, fy, cx, cy = camera
    return p[..., :2] / p[..., 2, None] * [fx, fy] + [cx, cy]


def unproject(camera, uv, z):
    fx, fy, cx, cy = camera
    if not np.isfinite(z) or z <= 0:
        raise ValueError('invalid axial depth')
    return np.r_[(np.asarray(uv) - [cx, cy]) / [fx, fy] * z, z]


def block_huber(residuals, delta):
    """Transform each row so squared norm equals Ceres rho(||r||²).

    SciPy is called with linear loss. A 2D pixel block stays a 2D block;
    a scalar depth block is transformed independently. This matches costs,
    not Ceres' linearization or trust-region steps.
    """
    r = np.asarray(residuals, float)
    norm = np.linalg.norm(r, axis=-1)
    rho = np.where(norm <= delta, norm**2, 2 * delta * norm - delta**2)
    scale = np.sqrt(np.divide(rho, norm**2, out=np.ones_like(norm), where=norm > 0))
    return r * scale[..., None]


def skew(vectors):
    """Batched cross-product matrices."""
    vectors = np.asarray(vectors, float)
    result = np.zeros(vectors.shape[:-1] + (3, 3))
    x, y, z = np.moveaxis(vectors, -1, 0)
    result[..., 0, 1], result[..., 0, 2] = -z, y
    result[..., 1, 0], result[..., 1, 2] = z, -x
    result[..., 2, 0], result[..., 2, 1] = -y, x
    return result


def so3_right_jacobian(rotvec):
    """Derivative used by additive angle-axis coordinates, stable at zero."""
    w = np.asarray(rotvec, float)
    theta2 = np.sum(w*w, axis=-1)
    theta = np.sqrt(theta2)
    small = theta < 1e-5
    safe2 = np.where(small, 1., theta2)
    a = np.where(small, .5-theta2/24+theta2**2/720, (1-np.cos(theta))/safe2)
    b = np.where(small, 1/6-theta2/120+theta2**2/5040,
                 (theta-np.sin(theta))/(safe2*np.where(small, 1., theta)))
    k = skew(w)
    return np.eye(3) - a[...,None,None]*k + b[...,None,None]*(k@k)


def block_huber_jacobian(residuals, delta):
    """Jacobian of block_huber, including the derivative of its scale."""
    r = np.asarray(residuals, float)
    s = np.sum(r*r, axis=-1)
    norm = np.sqrt(s)
    outside = norm > delta
    rho = np.where(outside, 2*delta*norm-delta**2, s)
    safe = np.where(outside, s, 1.)
    scale = np.where(outside, np.sqrt(np.maximum(rho, 0.)/safe), 1.)
    rho_prime = np.where(outside, delta/np.maximum(norm, 1e-30), 1.)
    twice_derivative = np.where(outside, (rho_prime*safe-rho)/(scale*safe**2), 0.)
    return scale[...,None,None]*np.eye(r.shape[-1]) + twice_derivative[...,None,None]*(r[..., :,None]*r[...,None,:])
