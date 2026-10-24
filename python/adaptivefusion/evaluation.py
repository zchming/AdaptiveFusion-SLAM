"""Independent trajectory evaluation. Never imported by inference or optimization."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import inverse


def read_tum(path):
    times, poses = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        values = np.array(list(map(float, line.split())))
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError('invalid TUM trajectory row')
        stamp, xyz, q = values[0], values[1:4], values[4:]
        if (times and stamp <= times[-1]) or np.linalg.norm(q) < 1e-12:
            raise ValueError('invalid TUM timestamps or quaternion')
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(q / np.linalg.norm(q)).as_matrix()
        pose[:3, 3] = xyz
        times.append(stamp); poses.append(pose)
    if not times:
        raise ValueError('empty trajectory')
    return np.array(times), np.array(poses)


def associate(truth_times, estimate_times, tolerance=.02):
    """Match the existing C++ evaluator's monotone nearest-unused association."""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('invalid association tolerance')
    cursor, matches = 0, []
    for index, timestamp in enumerate(estimate_times):
        if cursor >= len(truth_times): break
        while (cursor + 1 < len(truth_times) and
               abs(truth_times[cursor + 1] - timestamp) <= abs(truth_times[cursor] - timestamp)):
            cursor += 1
        if abs(truth_times[cursor] - timestamp) <= tolerance:
            matches.append((cursor, index)); cursor += 1
    return matches


def rigid_alignment(estimated_xyz, truth_xyz):
    """Least-squares SE(3) alignment, with reflection correction and no scale fit."""
    emean, gmean = estimated_xyz.mean(axis=0), truth_xyz.mean(axis=0)
    u, _, vt = np.linalg.svd((estimated_xyz-emean).T @ (truth_xyz-gmean))
    correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    translation = gmean - rotation @ emean
    result = np.eye(4); result[:3,:3] = rotation; result[:3,3] = translation
    return result


def fixed_interval_pairs(times, delta=1., tolerance=.02):
    if not np.isfinite([delta,tolerance]).all() or min(delta,tolerance) <= 0:
        raise ValueError('invalid RPE interval/tolerance')
    pairs = []
    for i, timestamp in enumerate(times):
        target = timestamp + delta
        position = int(np.searchsorted(times, target))
        options = [j for j in [position-1,position] if i < j < len(times)]
        if not options: continue
        j = min(options,key=lambda j:(abs(times[j]-target),j))
        if abs(times[j]-target) <= tolerance: pairs.append((i,j))
    return pairs


def rpe(truth, estimate, pairs):
    translation, rotation = [], []
    for i,j in pairs:
        truth_relative = inverse(truth[i]) @ truth[j]
        estimate_relative = inverse(estimate[i]) @ estimate[j]
        error = inverse(truth_relative) @ estimate_relative
        translation.append(float(error[:3,3] @ error[:3,3]))
        rotation.append(float(Rotation.from_matrix(error[:3,:3]).magnitude()**2))
    return {'pairs':len(pairs),
            'translation_rmse_m':float(np.sqrt(np.mean(translation))) if pairs else None,
            'rotation_rmse_rad':float(np.sqrt(np.mean(rotation))) if pairs else None}


def evaluate(truth_file, estimate_file, *, time_tolerance=.02, rpe_delta=1., rpe_tolerance=.02):
    truth_times, truth = read_tum(truth_file)
    estimate_times, estimate = read_tum(estimate_file)
    matches = associate(truth_times,estimate_times,time_tolerance)
    if len(matches) < 3: raise ValueError('fewer than three truth matches')
    gi,ei = np.array(matches).T
    g,e = truth[gi],estimate[ei]
    alignment = rigid_alignment(e[:,:3,3],g[:,:3,3])
    aligned_xyz = (alignment[:3,:3] @ e[:,:3,3].T).T + alignment[:3,3]
    ate = float(np.sqrt(np.mean(np.sum((aligned_xyz-g[:,:3,3])**2,axis=1))))
    pairs = fixed_interval_pairs(estimate_times[ei],rpe_delta,rpe_tolerance)
    # Both actual sensor time spans must meet the frozen interval tolerance.
    pairs = [(i,j) for i,j in pairs if abs(truth_times[gi[j]]-truth_times[gi[i]]-rpe_delta)<=rpe_tolerance]
    jumps = []
    for i in range(1,len(estimate)):
        dt = estimate_times[i]-estimate_times[i-1]
        if dt > .1: continue
        relative = inverse(estimate[i-1]) @ estimate[i]
        if np.linalg.norm(relative[:3,3]) > .2 or Rotation.from_matrix(relative[:3,:3]).magnitude() > np.deg2rad(15):
            jumps.append(i)
    duration = truth_times[-1]-truth_times[0]
    return {'alignment':'SE3 without scale', 'ground_truth_poses':len(truth),
            'estimated_poses':len(estimate),'matched_poses':len(matches),
            'estimated_match_ratio':len(matches)/len(estimate),
            'duration_coverage_ratio':float((truth_times[gi[-1]]-truth_times[gi[0]])/duration) if duration>0 else None,
            'ate_rmse_m':ate,'adjacent_matched_rpe':rpe(g,e,[(i-1,i) for i in range(1,len(matches))]),
            'fixed_interval_rpe':rpe(g,e,pairs),'rpe_delta_s':rpe_delta,'rpe_tolerance_s':rpe_tolerance,
            'max_gt_association_difference_s':float(np.max(np.abs(truth_times[gi]-estimate_times[ei]))),
            'internal_gaps_over_0_1s':int(np.sum(np.diff(estimate_times)>.1)),
            'jump_indices':jumps,'first_timestamp':float(estimate_times[0]),'last_timestamp':float(estimate_times[-1])}
