"""Ceres-compatible window objective with transactional offline SciPy solving."""
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, coo_matrix
from scipy.spatial.transform import Rotation
from .geometry import (inverse, project, to_cw_vector, from_cw_vector, block_huber,
                       skew, so3_right_jacobian, block_huber_jacobian)
from .components import keyframe_components


class Problem:
    def __init__(self, snapshot, boundary='window', component_policy='all'):
        self.s = snapshot.validate()
        if boundary not in ['window','fixed-external']:raise ValueError('invalid boundary model')
        self.boundary=boundary
        c = snapshot.config
        self.begin = max(0, len(snapshot.keyframes) - c.window_size)
        self.kids = list(range(self.begin, len(snapshot.keyframes)))
        if component_policy not in ['all', 'anchor']:
            raise ValueError('invalid component policy')
        if component_policy == 'anchor' and (boundary != 'window' or self.begin != 0):
            raise ValueError('anchor component policy requires a full batch window')
        self.component_policy = component_policy
        counts = {}
        total_counts={}
        for o in snapshot.observations:
            total_counts[o.point_id]=total_counts.get(o.point_id,0)+1
            if o.keyframe_id >= self.begin:
                counts[o.point_id] = counts.get(o.point_id, 0) + 1
        self.pids = [pid for pid in snapshot.points if
                     (counts.get(pid,0)>=c.minimum_point_observations if boundary=='window' else
                      counts.get(pid,0)>0 and total_counts.get(pid,0)>=c.minimum_point_observations)]
        if component_policy == 'anchor':
            components = keyframe_components(snapshot, self.kids, self.pids)
            self.kids = next((group for group in components if self.begin in group), [])
            active = set(self.kids)
            self.pids = [pid for pid in self.pids if snapshot.points[pid].host in active]
        point_ids = set(self.pids)
        self.obs = [o for o in snapshot.observations if o.point_id in point_ids
                    and (o.keyframe_id >= self.begin or boundary=='fixed-external')]
        self.free = self.kids[1:]
        self.ki = {kid: i for i, kid in enumerate(self.free)}
        self.pi = {pid: i for i, pid in enumerate(self.pids)}
        self.npose = len(self.free) * 6
        self.x0 = np.r_[*[to_cw_vector(snapshot.keyframes[k].t_wc) for k in self.free],
                        *[snapshot.points[p].xyz for p in self.pids]] if self.free or self.pids else np.empty(0)
        self.ok = np.array([o.keyframe_id for o in self.obs], int)
        self.free_index=np.array([self.ki.get(o.keyframe_id,-1) for o in self.obs],int)
        self.op = np.array([self.pi[o.point_id] for o in self.obs], int)
        self.uv = np.array([[o.u, o.v] for o in self.obs], float).reshape(-1, 2)
        self.depth_mask = np.array([o.raw_depth > 0 for o in self.obs], bool)
        self.depths = np.array([o.raw_depth / c.depth_scale for o in self.obs])
        self.fixed=np.array([to_cw_vector(k.t_wc) for k in snapshot.keyframes])

    def unpack(self, x):
        poses=self.fixed.copy()
        if self.free:poses[self.free]=x[:self.npose].reshape(-1,6)
        points = x[self.npose:].reshape(-1, 3)
        return poses, points

    def camera_points(self, x):
        poses, points = self.unpack(x)
        return (Rotation.from_rotvec(poses[self.ok, :3]).apply(points[self.op])
                + poses[self.ok, 3:])

    def raw(self, x):
        pc = self.camera_points(x)
        pixel = project(self.s.camera, pc) - self.uv
        depth = (pc[self.depth_mask, 2] - self.depths[self.depth_mask]) * self.s.config.depth_residual_weight
        return pixel, depth

    def residual(self, x):
        pixel, depth = self.raw(x)
        delta = self.s.config.huber_delta_pixels
        return np.r_[block_huber(pixel, delta).ravel(), block_huber(depth[:, None], delta).ravel()]

    def metrics(self, x):
        pixel, depth = self.raw(x)
        r = self.residual(x)
        return {'robust_cost': float(r @ r / 2),
                'pixel_rmse': float(np.sqrt(np.mean(np.sum(pixel**2, axis=1)))),
                'weighted_depth_rmse': float(np.sqrt(np.mean(depth**2))) if depth.size else None}

    def sparsity(self):
        m, n = 2 * len(self.obs) + int(self.depth_mask.sum()), len(self.x0)
        a = lil_matrix((m, n), dtype=int)
        di = 2 * len(self.obs)
        for i, o in enumerate(self.obs):
            cols = list(range(self.npose + 3 * self.pi[o.point_id], self.npose + 3 * self.pi[o.point_id] + 3))
            if o.keyframe_id in self.ki:
                j = 6 * self.ki[o.keyframe_id]; cols += list(range(j, j + 6))
            for row in [2*i, 2*i+1]: a[row, cols] = 1
            if o.raw_depth:
                a[di, cols] = 1; di += 1
        return a.tocsr()

    def jacobian(self, x):
        """Exact sparse derivatives of the same grouped robust objective."""
        poses, points = self.unpack(x)
        rotations = Rotation.from_rotvec(poses[:,:3]).as_matrix()
        rotation = rotations[self.ok]
        pc = np.einsum('nij,nj->ni',rotation,points[self.op])+poses[self.ok,3:]
        pixel = project(self.s.camera,pc)-self.uv
        fx,fy,_,_ = self.s.camera
        projection = np.zeros((len(pc),2,3))
        projection[:,0,0] = fx/pc[:,2]
        projection[:,1,1] = fy/pc[:,2]
        projection[:,0,2] = -fx*pc[:,0]/pc[:,2]**2
        projection[:,1,2] = -fy*pc[:,1]/pc[:,2]**2
        angular = -rotation @ skew(points[self.op]) @ so3_right_jacobian(poses[self.ok,:3])
        dpc_pose = np.concatenate([angular,np.broadcast_to(np.eye(3),angular.shape)],axis=2)
        projection = block_huber_jacobian(pixel,self.s.config.huber_delta_pixels) @ projection
        pixel_point, pixel_pose = projection @ rotation, projection @ dpc_pose
        rows,cols,values = [],[],[]
        def block(rowbase,colbase,array):
            nr,nc=array.shape[1:]
            rows.append(np.broadcast_to(rowbase[:,None,None]+np.arange(nr)[None,:,None],array.shape).ravel())
            cols.append(np.broadcast_to(colbase[:,None,None]+np.arange(nc)[None,None,:],array.shape).ravel())
            values.append(array.ravel())
        free = self.free_index>=0
        block(2*np.arange(len(pc)),self.npose+3*self.op,pixel_point)
        block(2*np.flatnonzero(free),6*self.free_index[free],pixel_pose[free])
        dm = self.depth_mask
        depth_error = (pc[dm,2]-self.depths[dm])*self.s.config.depth_residual_weight
        depth_scale = block_huber_jacobian(depth_error[:,None],self.s.config.huber_delta_pixels)[:,0,0]*self.s.config.depth_residual_weight
        depth_point = rotation[dm,2,:]*depth_scale[:,None]
        depth_pose = dpc_pose[dm,2,:]*depth_scale[:,None]
        depth_rows = 2*len(pc)+np.arange(int(dm.sum()))
        block(depth_rows,self.npose+3*self.op[dm],depth_point[:,None,:])
        dfree = free[dm]
        block(depth_rows[dfree],6*self.free_index[dm][dfree],depth_pose[dfree,None,:])
        return coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
                          shape=(2*len(pc)+int(dm.sum()),len(x))).tocsr()

    def preflight(self):
        if len(self.kids) < 2 or not self.pids:
            raise ValueError('insufficient shared geometry')
        if not self.depth_mask.any():
            raise ValueError('metric scale unobservable: no active depth')
        # Each camera must have spatial support; structural connectivity must
        # connect every camera to the fixed anchor through shared landmarks.
        reached = set(range(self.begin+1)) if self.boundary=='fixed-external' else {self.begin}
        adjacency = {k: set() for k in range(len(self.s.keyframes))}
        cameras_by_point = {pid: set() for pid in self.pids}
        for o in self.obs: cameras_by_point[o.point_id].add(o.keyframe_id)
        for cameras in cameras_by_point.values():
            for k in cameras: adjacency[k].update(cameras)
        while True:
            updated = reached | set().union(*(adjacency[k] for k in reached))
            if updated == reached: break
            reached = updated
        if not set(self.kids)<=reached: raise ValueError('disconnected window')
        for k in self.kids:
            xyz = np.array([self.s.points[o.point_id].xyz for o in self.obs if o.keyframe_id == k])
            if len(xyz) < 4 or np.linalg.matrix_rank(xyz - xyz.mean(axis=0), tol=1e-8) < 2:
                raise ValueError('insufficient or collinear camera support')
        self.raw(self.x0)  # Positive depth and finite projection.


@dataclass
class SolveResult:
    accepted: bool
    reason: str
    snapshot: object
    initial: dict
    final: dict
    solve_ms: float
    nfev: int = 0
    variables: int = 0
    observations: int = 0
    diagnostics: dict = None

    def report(self):
        return {k: v for k, v in vars(self).items() if k != 'snapshot'}


def solve(snapshot, *, max_nfev=200, time_limit_s=30., jacobian='sparse-fd', lsmr_tolerance=None,
          termination_profile='strict', method='scipy-trf', acceptance_policy='converged',boundary='window',
          component_policy='all'):
    """Input is never mutated; every failure returns an independent full original.

    max_nfev is deliberately distinct from Ceres maximum_iterations. Both
    limits are recorded by the CLI, not presented as equivalent iteration counts.
    """
    start = time.perf_counter()
    if method=='schur-lm':jacobian='analytic'
    original = snapshot.clone()
    initial, final, nfev, nvars, nobs = {}, {}, 0, 0, 0
    diagnostics = {'jacobian':jacobian,'lsmr_tolerance':lsmr_tolerance,
                   'termination_profile':termination_profile,'method':method,'acceptance_policy':acceptance_policy,
                   'boundary':boundary,'component_policy':component_policy}
    try:
        if max_nfev <= 0 or not np.isfinite(time_limit_s) or time_limit_s <= 0:
            raise ValueError('invalid solver budget')
        problem = Problem(original,boundary=boundary,component_policy=component_policy)
        diagnostics['active_keyframe_ids'] = problem.kids
        diagnostics['partial_scope'] = len(problem.kids) < len(original.keyframes) - problem.begin
        nvars, nobs = len(problem.x0), len(problem.obs)
        problem.preflight()
        initial = problem.metrics(problem.x0)
        def residual(x):
            if time.perf_counter() - start > time_limit_s:
                raise TimeoutError('solver time budget exceeded')
            return problem.residual(x)
        if jacobian not in ['sparse-fd','analytic']:raise ValueError('invalid Jacobian method')
        profiles={'strict':(1e-10,1e-10,1e-10),'ceres-cost':(1e-6,1e-8,1e-10)}
        if termination_profile not in profiles:raise ValueError('invalid termination profile')
        ftol,xtol,gtol=profiles[termination_profile]
        diagnostics.update({'ftol':ftol,'xtol':xtol,'gtol':gtol})
        options = ({'jac':problem.jacobian} if jacobian=='analytic' else {'jac_sparsity':problem.sparsity()}) if method=='scipy-trf' else {}
        if lsmr_tolerance is not None:
            if not np.isfinite(lsmr_tolerance) or lsmr_tolerance<=0:raise ValueError('invalid LSMR tolerance')
            options['tr_options']={'atol':lsmr_tolerance,'btol':lsmr_tolerance}
        if method=='schur-lm':
            from .schur import optimize
            fit=optimize(problem,max_nfev,start+time_limit_s,ftol,xtol,gtol)
            diagnostics.update({'iterations':fit.iterations,'accepted_steps':fit.accepted_steps})
        elif method=='scipy-trf':
            fit = least_squares(residual, problem.x0, method='trf', loss='linear',
                            **options, tr_solver='lsmr',
                            x_scale='jac', max_nfev=max_nfev,
                            ftol=ftol, xtol=xtol, gtol=gtol)
        else:raise ValueError('invalid solver method')
        nfev = int(fit.nfev)
        if np.isfinite(fit.x).all():final = problem.metrics(fit.x)
        diagnostics.update({'termination_status':int(fit.status),'optimality':float(fit.optimality),
                            'last_candidate_recorded':bool(final),'returned_state':'original'})
        if acceptance_policy not in ['converged','usable']:raise ValueError('invalid acceptance policy')
        if not fit.success and acceptance_policy=='converged':
            raise ValueError('solver did not converge: ' + fit.message)
        if fit.status<0:raise ValueError('numerical solver failure: '+fit.message)
        if not fit.success and (fit.nfev<=1 or not final or final['robust_cost']>=initial['robust_cost']):
            raise ValueError('no usable cost-decreasing step: '+fit.message)
        if not np.isfinite(fit.x).all(): raise ValueError('non-finite solution')
        final = problem.metrics(fit.x)
        if final['robust_cost'] > initial['robust_cost'] + max(1e-10, initial['robust_cost'] * 1e-9):
            raise ValueError('robust cost increased')
        # A small exact rank audit catches geometrically unobservable examples.
        # Large windows use structural checks only; this is reported as a limit.
        if len(fit.x) <= 200:
            jac = fit.jac.toarray() if hasattr(fit.jac, 'toarray') else fit.jac
            if np.linalg.matrix_rank(jac, tol=1e-7) < len(fit.x):
                raise ValueError('rank-deficient local problem')
        candidate = original.clone()
        poses, points = problem.unpack(fit.x)
        for kid in problem.free:
            candidate.keyframes[kid].t_wc = from_cw_vector(poses[kid])
        for pid, j in problem.pi.items(): candidate.points[pid].xyz = points[j].copy()
        candidate.validate()
        if not np.array_equal(candidate.keyframes[problem.begin].t_wc,
                              original.keyframes[problem.begin].t_wc):
            raise ValueError('fixed anchor changed')
        if component_policy == 'anchor':
            active = set(problem.kids)
            if (any(not np.array_equal(k.t_wc, candidate.keyframes[k.id].t_wc)
                    for k in original.keyframes if k.id not in active)
                    or any(not np.array_equal(p.xyz, candidate.points[pid].xyz)
                           for pid, p in original.points.items() if pid not in problem.pi)):
                raise ValueError('excluded geometry changed')
        reason = ('accepted anchor-connected component only' if diagnostics['partial_scope']
                  and component_policy == 'anchor' else 'accepted offline parity solution')
        return SolveResult(True, reason, candidate, initial, final,
                           (time.perf_counter()-start)*1000, nfev, nvars, nobs,
                           {**diagnostics,'returned_state':'accepted_solution'})
    except (ValueError, TimeoutError, FloatingPointError, np.linalg.LinAlgError) as error:
        return SolveResult(False, str(error), original, initial, final,
                           (time.perf_counter()-start)*1000, nfev, nvars, nobs,diagnostics)


def reanchor_singletons(before, after):
    """Separate ablation: preserve each singleton's original host-camera point.

    Multi-view points are never rigidly dragged by a host. No live writeback.
    """
    before.validate(); after.validate()
    if (before.state_version != after.state_version or before.source_revision != after.source_revision
            or before.config != after.config or not np.array_equal(before.camera, after.camera)
            or set(before.points) != set(after.points) or set(before.observations) != set(after.observations)
            or len(before.keyframes) != len(after.keyframes)
            or any((a.id, a.source_frame_id, a.timestamp, a.width, a.height) !=
                   (b.id, b.source_frame_id, b.timestamp, b.width, b.height)
                   for a,b in zip(before.keyframes, after.keyframes))
            or any(before.points[p].host != after.points[p].host for p in before.points)):
        raise ValueError('snapshot state mismatch')
    output, changed = after.clone(), []
    counts = {pid: 0 for pid in before.points}
    for o in before.observations: counts[o.point_id] += 1
    for pid, p in before.points.items():
        if counts[pid] != 1: continue
        old, new = before.keyframes[p.host].t_wc, after.keyframes[p.host].t_wc
        if not np.array_equal(old, new):
            output.points[pid].xyz = (new @ inverse(old) @ np.r_[p.xyz, 1])[:3]
            changed.append(pid)
    return output.validate(), changed
