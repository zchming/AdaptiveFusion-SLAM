import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from fixtures import scene
from adaptivefusion.geometry import inverse, project, unproject, to_cw_vector, from_cw_vector, block_huber
from adaptivefusion.state import load_snapshot, write_snapshot, Point, Observation
from adaptivefusion.local_ba import Problem, solve, reanchor_singletons
from adaptivefusion.trajectory import FrameRecord, correct_history, load_frames


class GeometryTests(unittest.TestCase):
    def test_projection_pose_direction_and_scale(self):
        t=np.eye(4);t[:3,:3]=Rotation.from_rotvec([.2,-.1,.3]).as_matrix();t[:3,3]=[1,2,3]
        np.testing.assert_allclose(from_cw_vector(to_cw_vector(t)),t,atol=1e-14)
        point=np.array([.4,-.3,4.])
        camera=[500,510,320,240]
        np.testing.assert_allclose(unproject(camera,project(camera,point),point[2]),point)
        np.testing.assert_allclose(inverse(t) @ t,np.eye(4),atol=1e-14)
        current_from_ref=np.eye(4);current_from_ref[0,3]=-.2
        self.assertAlmostEqual((t @ inverse(current_from_ref))[0,3],(t @ np.array([.2,0,0,1]))[0])

    def test_block_huber_matches_ceres_not_scalar_huber(self):
        r=np.array([[3.,4.],[.3,.4],[0,0]])
        transformed=block_huber(r,2.)
        np.testing.assert_allclose(np.sum(transformed**2,axis=1),[16,.25,0])
        scalar=np.sum(block_huber(r.reshape(-1,1),2.)**2)
        self.assertNotAlmostEqual(float(np.sum(transformed**2)),float(scalar))

    def test_snapshot_round_trip_and_no_overwrite(self):
        s,_,_=scene()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'s.snapshot';write_snapshot(path,s);other=load_snapshot(path)
            self.assertEqual(set(other.observations),set(s.observations))
            np.testing.assert_array_equal(other.keyframes[1].t_wc,s.keyframes[1].t_wc)
            self.assertEqual(set(other.points),set(s.points))
            with self.assertRaises(FileExistsError):write_snapshot(path,s)
            path.write_text(path.read_text().replace('END','END extra'))
            with self.assertRaises(ValueError):load_snapshot(path)

    def test_noise_free_metric_recovery_and_fixed_anchor(self):
        s,truth,xyz=scene(quantized=False)
        original=s.clone();result=solve(s,max_nfev=800)
        self.assertTrue(result.accepted,result.reason)
        self.assertLess(result.final['robust_cost'],result.initial['robust_cost']*.001)
        np.testing.assert_array_equal(result.snapshot.keyframes[0].t_wc,s.keyframes[0].t_wc)
        for k in range(1,len(truth)):
            self.assertLess(np.linalg.norm(result.snapshot.keyframes[k].t_wc[:3,3]-truth[k][:3,3]),1e-4)
            self.assertLess(Rotation.from_matrix(truth[k][:3,:3].T @ result.snapshot.keyframes[k].t_wc[:3,:3]).magnitude(),1e-5)
        for a,b in zip(s.keyframes,original.keyframes):np.testing.assert_array_equal(a.t_wc,b.t_wc)
        for pid in s.points:np.testing.assert_array_equal(s.points[pid].xyz,original.points[pid].xyz)

    def test_noisy_optimization_and_geometry_error(self):
        s,truth,_=scene(noisy=True)
        result=solve(s,max_nfev=800)
        self.assertTrue(result.accepted,result.reason)
        self.assertLess(result.final['robust_cost'],result.initial['robust_cost'])
        before=sum(np.linalg.norm(k.t_wc[:3,3]-t[:3,3]) for k,t in zip(s.keyframes,truth))
        after=sum(np.linalg.norm(k.t_wc[:3,3]-t[:3,3]) for k,t in zip(result.snapshot.keyframes,truth))
        self.assertLess(after,before)

    def test_exact_start_remains_exact_with_quantization_tolerance(self):
        s,truth,_=scene(exact=True)
        r=solve(s,max_nfev=800)
        self.assertTrue(r.accepted,r.reason)
        self.assertLess(np.linalg.norm(r.snapshot.keyframes[2].t_wc[:3,3]-truth[2][:3,3]),3e-4)

    def assertRollback(self,s,**kwargs):
        original=s.clone();r=solve(s,**kwargs)
        self.assertFalse(r.accepted)
        for a,b in zip(r.snapshot.keyframes,original.keyframes):np.testing.assert_array_equal(a.t_wc,b.t_wc)
        for pid in s.points:np.testing.assert_array_equal(r.snapshot.points[pid].xyz,original.points[pid].xyz)
        self.assertEqual(r.snapshot.observations,s.observations)
        return r

    def test_timeout_and_nonconvergence_rollback(self):
        s,_,_=scene()
        self.assertIn('budget',self.assertRollback(s,time_limit_s=1e-12).reason)
        self.assertIn('converge',self.assertRollback(s,max_nfev=1).reason)

    def test_invalid_negative_depth_and_missing_scale_rollback(self):
        s,_,_=scene();s.points[100].xyz[2]=-1
        self.assertIn('depth',self.assertRollback(s).reason)
        s,_,_=scene();s.observations=[Observation(o.point_id,o.keyframe_id,o.feature_index,o.u,o.v,0) for o in s.observations]
        self.assertIn('scale',self.assertRollback(s).reason)
        s,_,_=scene();s.points[100].xyz[0]=np.nan
        self.assertIn('invalid',self.assertRollback(s).reason)

    def test_disconnected_and_collinear_rejected(self):
        s,_,_=scene();s.observations=[o for o in s.observations if o.keyframe_id!=2]
        self.assertIn('disconnected',self.assertRollback(s).reason)
        s,_,_=scene()
        for i,p in enumerate(s.points.values()):p.xyz=np.array([i*.1,0,4.])
        self.assertIn('collinear',self.assertRollback(s).reason)

    def test_window_boundary_and_global_scale(self):
        s,_,_=scene(cameras=6)
        problem=Problem(s)
        self.assertEqual(problem.begin,1)
        self.assertTrue(all(o.keyframe_id>=1 for o in problem.obs))
        scaled=s.clone()
        for k in scaled.keyframes:k.t_wc[:3,3]*=2
        for p in scaled.points.values():p.xyz*=2
        a,b=problem.raw(problem.x0),Problem(scaled).raw(Problem(scaled).x0)
        np.testing.assert_allclose(a[0],b[0],atol=1e-12)
        self.assertGreater(np.linalg.norm(b[1]),np.linalg.norm(a[1]))

    def test_singleton_reanchor_preserves_host_camera_geometry(self):
        s,_,_=scene();s.points[999]=Point(999,2,np.array([.1,.1,4.]))
        s.observations.append(Observation(999,2,99,330.,250.,20000))
        after=s.clone();after.keyframes[2].t_wc[0,3]+=.2
        repaired,ids=reanchor_singletons(s,after)
        self.assertEqual(ids,[999])
        np.testing.assert_allclose(inverse(s.keyframes[2].t_wc) @ np.r_[s.points[999].xyz,1],
                                   inverse(after.keyframes[2].t_wc) @ np.r_[repaired.points[999].xyz,1])
        np.testing.assert_array_equal(repaired.points[100].xyz,after.points[100].xyz)
        np.testing.assert_array_equal(after.points[999].xyz,s.points[999].xyz)

    def test_history_keeps_relative_pose_and_lost_gap(self):
        s,_,_=scene()
        records=[]
        for k in s.keyframes:
            follow=k.t_wc.copy();follow[0,3]+=.03
            records += [FrameRecord(2*k.id,k.timestamp,True,2*k.id-1,k.id,k.t_wc),
                        FrameRecord(2*k.id+1,k.timestamp+.2,k.id!=1,2*k.id,k.id,follow)]
        corrected=s.keyframes[2].t_wc.copy();corrected[:3,:3]=Rotation.from_rotvec([.1,.2,0]).as_matrix();corrected[:3,3]+=.3
        output=correct_history(records,s.keyframes,{2:corrected})
        self.assertEqual(len(output),5)
        np.testing.assert_allclose(output[-2][1],corrected,atol=1e-14)
        np.testing.assert_allclose(inverse(output[-2][1]) @ output[-1][1],
                                   inverse(records[-2].t_wc) @ records[-1].t_wc,atol=1e-14)
        np.testing.assert_array_equal(output[0][1],records[0].t_wc)

    def test_wrong_state_and_unrepresentable_keypoint_rejected(self):
        s,_,_=scene();after=s.clone();after.state_version+=1
        with self.assertRaises(ValueError):reanchor_singletons(s,after)
        after=s.clone();after.points[100].host=1
        with self.assertRaises(ValueError):reanchor_singletons(s,after)
        o=s.observations[0]
        s.observations[0]=Observation(o.point_id,o.keyframe_id,o.feature_index,
                                     1.49999999,o.v,o.raw_depth)
        with self.assertRaises(ValueError):s.validate()


if __name__=='__main__':unittest.main()
