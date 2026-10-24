import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from adaptivefusion.observation_pool import observation_pool
from adaptivefusion.state import Observation,Point,write_snapshot
from test_landmark_identity import duplicate_scene


def fixture(root):
    full,_,_=duplicate_scene();full.config.window_size=4
    full.points[5000]=Point(5000,1,np.array([.1,.1,2.]))
    full.observations.append(Observation(5000,1,100,185.,145.,10000));full.validate()
    baseline=full.clone();del baseline.points[5000]
    baseline.observations=[o for o in baseline.observations if o.point_id!=5000]
    (root/'complete.json').write_text(json.dumps({'processed_frames':4,'risk_enabled':False,'online_writeback':False}))
    write_snapshot(root/'1.before.snapshot',full)
    return full,baseline


class ObservationPoolTests(unittest.TestCase):
    def test_recover_singletons_preserve_original_state_and_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);full,base=fixture(root);before=copy.deepcopy(base)
            view,report=observation_pool(root,base)
            self.assertEqual(set(view.points),set(full.points));self.assertEqual(set(view.observations),set(full.observations))
            self.assertEqual(report['singleton_points_recovered'],1)
            self.assertEqual(report['candidate_observations'],report['baseline_observations']+1)
            for k,p in zip(view.keyframes,base.keyframes):np.testing.assert_array_equal(k.t_wc,p.t_wc)
            for pid in base.points:np.testing.assert_array_equal(view.points[pid].xyz,base.points[pid].xyz)
            view.keyframes[0].t_wc[0,3]=100
            view.points[0].xyz[0]=100
            self.assertEqual(base.observations,before.observations)
            np.testing.assert_array_equal(base.keyframes[0].t_wc,before.keyframes[0].t_wc)
            np.testing.assert_array_equal(base.points[0].xyz,before.points[0].xyz)

    def test_duplicate_snapshots_do_not_duplicate_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);full,base=fixture(root);write_snapshot(root/'2.before.snapshot',full)
            view,report=observation_pool(root,base)
            self.assertEqual(len(view.observations),len(full.observations))
            self.assertEqual(len(report['cached_before_snapshot_sha256']),2)

    def test_changed_cached_observation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);full,base=fixture(root)
            full.observations[-1]=Observation(5000,1,100,185.,145.,10001)
            write_snapshot(root/'2.before.snapshot',full)
            with self.assertRaisesRegex(ValueError,'observation changed'):observation_pool(root,base)

    def test_wrong_baseline_subset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);full,base=fixture(root)
            del base.points[0];base.observations=[o for o in base.observations if o.point_id!=0]
            with self.assertRaisesRegex(ValueError,'shared cached points'):observation_pool(root,base)

    def test_risk_online_and_incomplete_cache_are_rejected(self):
        for key,value in [('risk_enabled',True),('online_writeback',True),('processed_frames',3)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);full,base=fixture(root)
                marker=json.loads((root/'complete.json').read_text());marker[key]=value
                (root/'complete.json').write_text(json.dumps(marker))
                with self.assertRaises(ValueError):observation_pool(root,base)

    def test_camera_source_and_frame_metadata_must_match(self):
        for change in ['camera','source','frame']:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);full,base=fixture(root)
                if change=='camera':base.camera[0]+=1
                elif change=='source':base.source_revision='different-source'
                else:base.keyframes[1].timestamp+=.01
                with self.assertRaises(ValueError):observation_pool(root,base)

    def test_baseline_poses_are_preserved_despite_cached_ba_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);full,base=fixture(root)
            for k in base.keyframes:k.t_wc[0,3]=5*k.id
            view,_=observation_pool(root,base)
            for k,p in zip(view.keyframes,base.keyframes):np.testing.assert_array_equal(k.t_wc,p.t_wc)
