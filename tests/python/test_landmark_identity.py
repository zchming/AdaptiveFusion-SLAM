import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from adaptivefusion.landmark_identity import audit_pair, audit_cycles, identity_evidence, reciprocal_matches
from adaptivefusion.state import Snapshot, Keyframe, Point, Observation


def duplicate_scene(same_ids=False):
    import cv2
    g=np.random.default_rng(2).integers(0,256,(160,200),dtype=np.uint8)
    images=[cv2.warpAffine(g,np.array([[1,0,2*k],[0,1,0]],float),(200,160)) for k in range(4)]
    d=np.full(g.shape,10000,np.uint16);points={};obs=[]
    pixels=[(y,x) for y in range(25,140,15) for x in range(25,175,15)]
    for kid in range(4):
        offset=0 if same_ids or kid<2 else 1000
        for i,(y,x) in enumerate(pixels):
            pid=offset+i
            if pid not in points:points[pid]=Point(pid,kid,np.array([x/100,y/100,2.]))
            obs.append(Observation(pid,kid,i,float(x+2*kid),float(y),10000))
    s=Snapshot(np.array([100.,100.,100.,80.]),[Keyframe(i,i,float(i+1),200,160,np.eye(4)) for i in range(4)],points,obs).validate()
    return s,images,d


def fake_report(a,b,first=0,second=1,transform=None):
    return {'source_keyframe':first,'target_keyframe':second,
            'verification':{'accepted':True,'t_target_source':np.eye(4).tolist() if transform is None else transform.tolist()},
            'records':[{'source_point_id':a,'target_point_id':b,'source_keyframe':first,'target_keyframe':second,'pair_geometric_inlier':True}]}


def topology(tracks):
    return SimpleNamespace(points={p:None for p in tracks},observations=[SimpleNamespace(point_id=p,keyframe_id=k) for p,frames in tracks.items() for k in frames])


class LandmarkIdentityTests(unittest.TestCase):
    def test_reciprocal_unique_neighbors_and_many_to_one_rejection(self):
        a=np.array([[0,0],[10,0]],float);b=a+3
        self.assertEqual(reciprocal_matches(a,b,b,a,[True,True],[True,True]),[(0,0),(1,1)])
        self.assertEqual(reciprocal_matches(a,np.array([[3,0],[3.5,0]]),b,a,[True,True],[True,True]),[])
        self.assertEqual(reciprocal_matches(a,b,b,a,[False,True],[True,True]),[(1,1)])
        self.assertEqual(reciprocal_matches([],b,[],a,[],[True,True]),[])

    def test_synthetic_duplicate_ids_have_metric_and_multiframe_support(self):
        s,images,d=duplicate_scene();reports=[]
        for a,b in [(0,1),(0,2),(1,2)]:
            reports.append(audit_pair(s,a,b,images[a],images[b],d,d))
        self.assertTrue(all(p['verification']['accepted'] for p in reports))
        self.assertGreater(reports[1]['different_id_matches'],50)
        np.testing.assert_allclose(np.array(reports[1]['verification']['t_target_source'])[:3,3],[.08,0,0],atol=1e-10)
        cycles=audit_cycles(reports,[0,1,2]);self.assertEqual(len(cycles['qualified_pairs']),3)
        identity=identity_evidence(s,reports[1:],[0,1,2])
        self.assertGreater(identity['candidate_edges'],50)

    def test_cached_poses_do_not_drive_identity_and_input_is_unchanged(self):
        s,images,d=duplicate_scene();first=audit_pair(s,0,2,images[0],images[2],d,d)
        for k in s.keyframes:k.t_wc[:3,3]=[10*k.id,3*k.id,-4*k.id]
        before=copy.deepcopy(s)
        second=audit_pair(s,0,2,images[0],images[2],d,d)
        self.assertEqual(first,second)
        for a,b in zip(s.keyframes,before.keyframes):np.testing.assert_array_equal(a.t_wc,b.t_wc)
        self.assertEqual(s.observations,before.observations)
        for p in s.points:np.testing.assert_array_equal(s.points[p].xyz,before.points[p].xyz)

    def test_existing_same_id_tracks_are_controls_not_new_identity(self):
        s,images,d=duplicate_scene(same_ids=True);p=audit_pair(s,0,2,images[0],images[2],d,d)
        self.assertTrue(p['verification']['accepted']);self.assertEqual(p['different_id_matches'],0)
        self.assertEqual(identity_evidence(s,[p],[0,1,2])['candidate_edges'],0)

    def test_failed_depth_provenance_and_invalid_scope_reject(self):
        s,images,d=duplicate_scene();wrong=d.copy();wrong[25,25]=0
        with self.assertRaisesRegex(ValueError,'provenance'):audit_pair(s,0,2,images[0],images[2],wrong,d)
        with self.assertRaises(ValueError):audit_pair(s,2,0,images[2],images[0],d,d)
        with self.assertRaisesRegex(ValueError,'type/size'):audit_pair(s,0,2,images[0].astype(float),images[2],d,d)

    def test_one_pair_and_duplicate_reports_cannot_inflate_support(self):
        s=topology({1:[0,1],2:[2,3]});p=fake_report(1,2,0,2)
        j=identity_evidence(s,[p,p,p],[0,1,2,3]);self.assertEqual(j['candidate_edges'],0)
        self.assertEqual(len(j['edges'][0]['view_pairs']),1)

    def test_three_view_support_and_pairwise_ownership_conflict(self):
        s=topology({1:[0,1],2:[2,3]});reports=[fake_report(1,2,0,2),fake_report(1,2,1,2)]
        self.assertEqual(identity_evidence(s,reports,[0,1,2,3])['candidate_edges'],1)
        s=topology({1:[0,1],2:[1,2,3]});j=identity_evidence(s,reports,[0,1,2,3])
        self.assertEqual(j['candidate_edges'],0);self.assertFalse(j['edges'][0]['pairwise_ownership_compatible'])

    def test_transitive_cluster_conflict_is_not_silently_merged(self):
        s=topology({1:[0,1],2:[2,3],3:[0,4]})
        reports=[fake_report(1,2,0,2),fake_report(1,2,1,2),fake_report(2,3,2,4),fake_report(2,3,3,4)]
        j=identity_evidence(s,reports,[0,1,2,3,4]);self.assertEqual(j['candidate_edges'],0)
        self.assertEqual(j['clusters'][0]['ownership_conflict_keyframes'],[0])

    def test_rejected_pairs_and_out_of_window_evidence_do_not_vote(self):
        s=topology({1:[0,1],2:[2,3]});a=fake_report(1,2,0,2);b=fake_report(1,2,1,2)
        b['verification']['accepted']=False
        self.assertEqual(identity_evidence(s,[a,b],[0,1,2,3])['candidate_edges'],0)
        b['verification']['accepted']=True
        self.assertEqual(identity_evidence(s,[a,b],[0,2])['candidate_edges'],0)

    def test_triangle_composition_order_and_failed_cycle(self):
        a=np.eye(4);a[:3,:3]=Rotation.from_rotvec([.1,0,0]).as_matrix();a[:3,3]=[.1,0,.2]
        b=np.eye(4);b[:3,:3]=Rotation.from_rotvec([0,.2,0]).as_matrix();b[:3,3]=[0,.3,0]
        reports=[fake_report(1,2,0,1,a),fake_report(1,2,1,2,b),fake_report(1,2,0,2,b@a)]
        self.assertEqual(len(audit_cycles(reports,[0,1,2])['qualified_pairs']),3)
        reports[-1]['verification']['t_target_source']=(a@b).tolist()
        self.assertFalse(audit_cycles(reports,[0,1,2])['cycles'][0]['consistent'])
        self.assertEqual(audit_cycles(reports,[0,1,2])['qualified_pairs'],[])

    def test_missing_cycle_cannot_certify_a_pair(self):
        self.assertEqual(audit_cycles([fake_report(1,2,0,1)],[0,1,2])['qualified_pairs'],[])

    def test_entry_refuses_overwrite(self):
        entry=Path(__file__).resolve().parents[2]/'python/audit_landmark_identity.py'
        with tempfile.TemporaryDirectory() as tmp:
            p=subprocess.run([sys.executable,str(entry),'--snapshot','missing','--dataset','missing','--output',tmp],capture_output=True,text=True)
            self.assertEqual(p.returncode,2);self.assertIn('refuse overwrite',p.stderr)
