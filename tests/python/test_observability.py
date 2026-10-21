import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation
from fixtures import scene
from test_components import records
from adaptivefusion.geometry import inverse, project
from adaptivefusion.local_ba import Problem, solve
from adaptivefusion.observability import bridge_audit, single_landmark_separators
from adaptivefusion.state import Point, Observation, write_snapshot, load_snapshot
from adaptivefusion.evaluation import read_tum


def bridge_scene(count, *, collinear=False, interleaved=False):
    s, _, xyz = scene(cameras=4, exact=True, quantized=False)
    s.config.window_size = 4
    left = {0, 2} if interleaved else {0, 1}
    right = sorted(set(range(4))-left)
    for pid, point in list(s.points.items()):
        s.points[pid+1000] = Point(pid+1000, right[0], point.xyz.copy())
    s.observations = [Observation(
        o.point_id if o.keyframe_id in left else o.point_id+1000,
        o.keyframe_id, o.feature_index, o.u, o.v, o.raw_depth)
        for o in s.observations]
    bridge_xyz = [np.array([.05,.05,5.])+t*np.array([.2,.08,.1])
                  for t in np.linspace(-1., 1., max(count, 1))]
    if count >= 3 and not collinear:
        bridge_xyz[-1] = bridge_xyz[-1] + [0., .12, 0.]
    for j, position in enumerate(bridge_xyz[:count]):
        pid = 5000+j; s.points[pid] = Point(pid, 0, position)
        for k in s.keyframes:
            pc = (inverse(k.t_wc) @ np.r_[position,1])[:3]
            pixel = project(s.camera,pc).astype(np.float32).astype(float)
            s.observations.append(Observation(pid,k.id,len(xyz)+j,*pixel,int(round(pc[2]*5000))))
    return s.validate(), right


def rotated_right(s, right, count):
    center = s.points[5000].xyz
    axis = (s.points[5001].xyz-center) if count >= 2 else np.array([0.,1.,0.])
    rotation = Rotation.from_rotvec(.1*axis/np.linalg.norm(axis)).as_matrix()
    g = np.eye(4); g[:3,:3] = rotation; g[:3,3] = center-rotation@center
    output = s.clone()
    for kid in right: output.keyframes[kid].t_wc = g @ output.keyframes[kid].t_wc
    for pid, point in output.points.items():
        if point.host in right: point.xyz = (g @ np.r_[point.xyz,1])[:3]
    return output.validate()


class ObservabilityTests(unittest.TestCase):
    def test_single_bridge_has_rotational_invariance_and_is_rejected(self):
        s, right = bridge_scene(1)
        p = Problem(s); p.preflight()  # The old local-support checks pass.
        q = Problem(rotated_right(s,right,1))
        np.testing.assert_allclose(p.camera_points(p.x0),q.camera_points(q.x0),atol=2e-13,rtol=0)
        self.assertGreater(np.linalg.norm(s.keyframes[2].t_wc-q.s.keyframes[2].t_wc),.05)
        audit = bridge_audit(p)
        self.assertEqual(audit['decision'],'reject')
        self.assertIn(5000,[w['point_id'] for w in audit['single_landmark_separators']])

    def test_two_bridge_points_leave_rotation_about_their_axis(self):
        s, right = bridge_scene(2); p = Problem(s)
        self.assertFalse(single_landmark_separators(p.kids,p.pids,p.obs))
        q = Problem(rotated_right(s,right,2))
        np.testing.assert_allclose(p.residual(p.x0),q.residual(q.x0),atol=3e-11,rtol=0)
        audit = bridge_audit(p)
        self.assertEqual(audit['decision'],'reject')
        self.assertTrue(any(c['crossing_landmarks']==2 for c in audit['temporal_cuts']))

    def test_three_collinear_points_remain_degenerate(self):
        s, right = bridge_scene(3,collinear=True); p = Problem(s)
        q = Problem(rotated_right(s,right,3))
        np.testing.assert_allclose(p.residual(p.x0),q.residual(q.x0),atol=3e-11,rtol=0)
        self.assertTrue(any(c['rejection_kind']=='numerically_collinear_at_input'
                            for c in bridge_audit(p)['temporal_cuts']))

    def test_three_noncollinear_planar_points_are_not_falsely_rejected(self):
        s, right = bridge_scene(3); p = Problem(s)
        audit = bridge_audit(p)
        self.assertEqual(audit['decision'],'not_rejected_not_certified')
        q = Problem(rotated_right(s,right,3))
        self.assertGreater(np.max(np.abs(p.residual(p.x0)-q.residual(q.x0))),.01)
        cut = next(c for c in audit['temporal_cuts'] if c['crossing_landmarks']==3)
        self.assertLess(cut['singular_values_m'][2],1e-12)  # Planarity is allowed.

    def test_interleaved_single_point_separator_is_found(self):
        s, _ = bridge_scene(1,interleaved=True); p = Problem(s)
        audit = bridge_audit(p)
        self.assertTrue(all(c['rejection_kind'] is None for c in audit['temporal_cuts']))
        self.assertEqual(audit['decision'],'reject')
        self.assertIn(5000,[w['point_id'] for w in audit['single_landmark_separators']])

    def test_non_temporal_two_point_separator_is_an_explicit_limit(self):
        s, right = bridge_scene(2,interleaved=True); p = Problem(s)
        q = Problem(rotated_right(s,right,2))
        np.testing.assert_allclose(p.residual(p.x0),q.residual(q.x0),atol=3e-11,rtol=0)
        self.assertEqual(bridge_audit(p)['decision'],'not_rejected_not_certified')
        self.assertIn('non-temporal',bridge_audit(p)['limitations'])

    def test_rejection_happens_before_solver_and_preserves_all_state(self):
        for count, collinear in [(1,False),(2,False),(3,True)]:
            s, _ = bridge_scene(count,collinear=collinear); original = s.clone()
            with patch('adaptivefusion.schur.optimize') as optimizer:
                fit = solve(s,method='schur-lm',observability_check='bridge-audit')
                optimizer.assert_not_called()
            self.assertFalse(fit.accepted); self.assertEqual(fit.nfev,0)
            self.assertEqual(fit.reason,'bridge observability check failed')
            self.assertEqual(fit.snapshot.observations,original.observations)
            self.assertEqual(set(fit.snapshot.points),set(original.points))
            for a,b in zip(fit.snapshot.keyframes,original.keyframes):
                np.testing.assert_array_equal(a.t_wc,b.t_wc)
            for pid,point in original.points.items():
                np.testing.assert_array_equal(point.xyz,fit.snapshot.points[pid].xyz)

    def test_supported_input_matches_default_numerically(self):
        s, _, _ = scene(quantized=False)
        a = solve(s,method='schur-lm'); b = solve(s,method='schur-lm',observability_check='bridge-audit')
        self.assertTrue(a.accepted and b.accepted)
        self.assertEqual(b.diagnostics['observability_audit']['decision'],'not_rejected_not_certified')
        self.assertEqual(a.final,b.final); self.assertEqual(a.nfev,b.nfev)
        for x,y in zip(a.snapshot.keyframes,b.snapshot.keyframes):np.testing.assert_array_equal(x.t_wc,y.t_wc)
        for pid in a.snapshot.points:np.testing.assert_array_equal(a.snapshot.points[pid].xyz,b.snapshot.points[pid].xyz)

    def test_audit_is_coordinate_frame_and_metric_scale_consistent(self):
        for collinear in [False,True]:
            s, _ = bridge_scene(3,collinear=collinear)
            expected = bridge_audit(Problem(s))['decision']
            for scale in [.001,1000.]:
                changed = s.clone(); rotation = Rotation.from_rotvec([.2,-.1,.3]).as_matrix()
                for point in changed.points.values():point.xyz=rotation@(scale*point.xyz)+[3.,-5.,7.]
                self.assertEqual(bridge_audit(Problem(changed))['decision'],expected)

    def test_iterative_articulations_match_bruteforce_camera_connectivity(self):
        rng = np.random.default_rng(7)
        for _ in range(40):
            kids=list(range(6)); pids=list(range(100,108)); obs=[]
            for pid in pids:
                for kid in rng.choice(kids,size=int(rng.integers(2,5)),replace=False):
                    obs.append(Observation(pid,int(kid),pid,0.,0.,1))
            def components(excluded):
                groups={k:{k} for k in kids}
                tracks=defaultdict(set)
                for o in obs:
                    if o.point_id!=excluded:tracks[o.point_id].add(o.keyframe_id)
                for cameras in tracks.values():
                    union=set().union(*(groups[k] for k in cameras))
                    for k in union:groups[k]=union
                return len({tuple(sorted(v)) for v in groups.values()})
            expected={pid for pid in pids if components(pid)>components(None)}
            self.assertEqual({w['point_id'] for w in single_landmark_separators(kids,pids,obs)},expected)

    def test_long_chain_does_not_require_python_recursion(self):
        kids=list(range(1500));pids=list(range(10000,11499));observations=[]
        for k,pid in enumerate(pids):
            observations.extend([Observation(pid,k,0,0.,0.,1),Observation(pid,k+1,0,0.,0.,1)])
        result=single_landmark_separators(kids,pids,observations)
        self.assertEqual({w['point_id'] for w in result},set(pids))

    def test_near_collinearity_is_labeled_as_input_state_degeneracy(self):
        s,_=bridge_scene(3)
        # Measurements still describe a triangle; only its initialization is
        # almost collinear. This rejection is conservative, not a theorem
        # about all measurement-consistent solutions.
        s.points[5002].xyz[1]-=.12-1e-12
        audit=bridge_audit(Problem(s))
        self.assertEqual(audit['decision'],'reject')
        self.assertTrue(any(c['rejection_kind']=='numerically_collinear_at_input'
                            for c in audit['temporal_cuts']))
        self.assertIn('input-state dependent',audit['limitations'])

    def test_guard_is_restricted_to_full_batch_and_rejects_invalid_mode(self):
        s, _, _ = scene(cameras=6); s.config.window_size=2
        for mode in ['bridge-audit','invalid']:
            fit=solve(s,observability_check=mode)
            self.assertFalse(fit.accepted);self.assertEqual(fit.nfev,0)

    def test_guarded_cli_outputs_full_rollback_not_empty_trajectory(self):
        s, _ = bridge_scene(1); frames=records(s)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);cache=base/'cache';cache.mkdir()
            write_snapshot(cache/'42.before.snapshot',s)
            (cache/'complete.json').write_text(json.dumps({'processed_frames':len(frames),'risk_enabled':False,'online_writeback':False}))
            with (cache/'frames.txt').open('x') as f:
                f.write('frame_id timestamp valid reference_id anchor_id '+' '.join(f't{i}' for i in range(16))+'\n')
                for frame in frames:f.write(f'{frame.id} {frame.timestamp:.17g} 1 {frame.reference_id} {frame.anchor_id} '+' '.join(f'{v:.17g}' for v in frame.t_wc.ravel())+'\n')
            root=Path(__file__).resolve().parents[2]
            command=[sys.executable,str(root/'python/run_batch_reference.py'),'--cache',str(cache),'--output',str(base/'result'),'--component-policy','anchor','--observability-check','bridge-audit']
            result=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,2,result.stdout+result.stderr)
            report=json.loads((base/'result/report.json').read_text())
            self.assertEqual(report['status'],'rejected');self.assertEqual(report['nfev'],0)
            times,poses=read_tum(base/'result/corrected.tum');self.assertEqual(len(times),len(frames))
            for frame,t,pose in zip(frames,times,poses):
                self.assertEqual(frame.timestamp,t);np.testing.assert_allclose(frame.t_wc,pose,atol=1e-12,rtol=0)
            after=load_snapshot(base/'result/output.snapshot')
            for k in s.keyframes:np.testing.assert_array_equal(k.t_wc,after.keyframes[k.id].t_wc)
            for pid,point in s.points.items():np.testing.assert_array_equal(point.xyz,after.points[pid].xyz)


if __name__=='__main__':unittest.main()
