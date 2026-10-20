import unittest
import numpy as np
from fixtures import scene
from adaptivefusion.offline import CachedState,full_history,merge_cached_problem,split_landmark_holdout,heldout_geometry
from adaptivefusion.trajectory import FrameRecord
from adaptivefusion.geometry import inverse


class OfflineStateTests(unittest.TestCase):
    def test_new_pose_uses_corrected_reference_and_new_points_follow_host(self):
        s,_,_=scene()
        records=[FrameRecord(k.source_frame_id,k.timestamp,True,-1,k.id,k.t_wc.copy()) for k in s.keyframes]
        state=CachedState(records)
        state.poses[0]=s.keyframes[0].t_wc.copy();state.poses[0][0,3]+=.5
        prepared=state.prepare(s)
        for k in prepared.keyframes:
            np.testing.assert_allclose(k.t_wc[:3,3],s.keyframes[k.id].t_wc[:3,3]+[.5,0,0],atol=1e-12)
        for pid,p in prepared.points.items():np.testing.assert_allclose(p.xyz,s.points[pid].xyz+[.5,0,0],atol=1e-12)
        state.commit(prepared,prepared)
        changed=s.clone();changed.points[100].xyz+=3
        next_input=state.prepare(changed)
        np.testing.assert_array_equal(next_input.points[100].xyz,state.points[100])
        self.assertFalse(np.array_equal(next_input.points[100].xyz,changed.points[100].xyz))
        np.testing.assert_array_equal(s.keyframes[0].t_wc,np.eye(4))

    def test_identity_backend_preserves_original_history(self):
        s,_,_=scene()
        records=[FrameRecord(k.source_frame_id,k.timestamp,True,-1,k.id,k.t_wc.copy()) for k in s.keyframes]
        result=full_history(records,s,{k.id:k.t_wc for k in s.keyframes})
        for f,(_,p) in zip(records,result):np.testing.assert_allclose(f.t_wc,p,atol=1e-12)

    def test_union_preserves_observations_and_birth_geometry(self):
        first,_,_=scene(cameras=2);last,_,_=scene(cameras=3)
        records=[FrameRecord(k.source_frame_id,k.timestamp,True,-1,k.id,k.t_wc.copy()) for k in last.keyframes]
        original=first.points[100].xyz.copy()
        last.points[100].xyz+=4  # A later C++ state must not overwrite birth geometry.
        merged=merge_cached_problem([first,last],records)
        self.assertEqual(len(merged.observations),24)
        np.testing.assert_array_equal(merged.points[100].xyz,original)
        self.assertEqual(merged.config.window_size,3)
        poses={k.id:k.t_wc.copy() for k in last.keyframes}
        poses[0][0,3]+=.5
        moved=merge_cached_problem([first,last],records,initialization='provided',initial_poses=poses)
        np.testing.assert_array_equal(first.points[100].xyz,original)
        np.testing.assert_allclose(moved.points[100].xyz,original+[.5,0,0])

    def test_whole_landmark_holdout_has_no_optimization_overlap(self):
        s,_,_=scene(exact=True,quantized=False)
        train,held=split_landmark_holdout(s)
        self.assertFalse(held&set(train.points))
        self.assertTrue(all(o.point_id not in held for o in train.observations))
        metric=heldout_geometry(s,{k.id:k.t_wc for k in s.keyframes},held)
        self.assertEqual(metric['cross_camera_observations'],2*len(held))
        self.assertEqual(metric['invalid_predictions'],0)
        self.assertLess(metric['pixel_rmse'],2e-5)


if __name__=='__main__':unittest.main()
