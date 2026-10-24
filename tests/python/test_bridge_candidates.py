import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from adaptivefusion.bridge_candidates import rigid_fit, residuals, verify_rigid, sample_depth, propose_pair
from fixtures import scene


class BridgeCandidateTests(unittest.TestCase):
    def points(self):
        a=np.random.default_rng(4).uniform(-1,1,(80,3));a[:,2]+=4
        r=Rotation.from_rotvec([.03,-.12,.08]).as_matrix();b=a@r.T+[.1,-.05,.02]
        return a,b,r

    def test_transform_direction_metric_scale_and_noiseless_recovery(self):
        a,b,r=self.points();t=rigid_fit(a,b)
        np.testing.assert_allclose(t[:3,:3],r,atol=1e-14)
        np.testing.assert_allclose(t[:3,3],[.1,-.05,.02],atol=1e-14)
        self.assertLess(residuals(t,a,b).max(),1e-14)
        self.assertTrue(verify_rigid(a,b,np.arange(80))['accepted'])
        self.assertFalse(verify_rigid(a,b*2,np.arange(80))['accepted'])

    def test_outliers_robustness_and_repeatability(self):
        a,b,_=self.points();b+=np.random.default_rng(3).normal(0,.001,b.shape);b[::5]+=[2,1,-1]
        first=verify_rigid(a,b,np.arange(80));self.assertTrue(first['accepted'])
        self.assertEqual(first,verify_rigid(a,b,np.arange(80)))
        self.assertTrue(all(i%5!=0 for i in first['inlier_point_ids']))

    def test_ambiguous_half_contaminated_fit_split_is_safely_rejected(self):
        a,b,_=self.points();b[::4]+=[2,1,-1]
        result=verify_rigid(a,b,np.arange(80))
        self.assertFalse(result['accepted']);self.assertEqual(result['inlier_point_ids'],[])

    def test_cut_selection_and_association_ambiguity_are_explicit(self):
        entry=Path(__file__).resolve().parents[2]/'python/audit_bridge_candidates.py'
        spec=importlib.util.spec_from_file_location('candidate_entry',entry)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        s,_,_=scene(cameras=2,exact=True,quantized=True)
        self.assertEqual(module.select_cuts(s),[(1,len(s.points))])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'rgb.txt').write_text('1.0 rgb/a.png\n')
            (root/'depth.txt').write_text('0.99 depth/a.png\n1.01 depth/b.png\n')
            with self.assertRaisesRegex(ValueError,'tie'):module.associations(root)
            (root/'depth.txt').write_text('1.001 depth/b.png\n')
            self.assertEqual(module.associations(root)[1.0],(root/'rgb/a.png',root/'depth/b.png'))

    def test_disjoint_equal_cost_associations_commute(self):
        entry=Path(__file__).resolve().parents[2]/'python/audit_bridge_candidates.py'
        spec=importlib.util.spec_from_file_location('entry_ties',entry)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'rgb.txt').write_text('1.0 rgb/a.png\n2.0 rgb/b.png\n')
            (root/'depth.txt').write_text('1.0 depth/a.png\n2.0 depth/b.png\n')
            self.assertEqual(len(module.associations(root)),2)

    def test_validation_cannot_be_used_to_refit_bad_model(self):
        a,b,_=self.points();b[1::2]+=[1,0,0]
        j=verify_rigid(a,b,np.arange(80));self.assertFalse(j['accepted']);self.assertEqual(j['inlier_point_ids'],[])

    def test_planar_allowed_collinear_rejected_and_proper_rotation(self):
        a,b,_=self.points();a[:,2]=4;b=a+[.1,.2,0]
        self.assertGreater(np.linalg.det(rigid_fit(a,b)[:3,:3]),.999999)
        a[:,1]=0
        with self.assertRaises(ValueError):rigid_fit(a,a)
        self.assertFalse(verify_rigid(a,a,np.arange(80))['accepted'])

    def test_empty_nonfinite_duplicate_ids_and_insufficient_split(self):
        self.assertFalse(verify_rigid([],[],[])['accepted'])
        a,b,_=self.points()
        with self.assertRaises(ValueError):verify_rigid(a,b,np.zeros(80))
        a[0,0]=np.nan
        with self.assertRaises(ValueError):verify_rigid(a,b,np.arange(80))
        self.assertFalse(verify_rigid(b,b,np.arange(80)*2)['accepted'])

    def test_native_depth_rounding_border_discontinuity_and_units(self):
        d=np.full((8,8),10000,np.uint16)
        self.assertEqual(sample_depth(d,[2.5,2],5000),2.)
        self.assertIsNone(sample_depth(d,[0,2],5000));d[2,3]=0
        self.assertIsNone(sample_depth(d,[2.5,2],5000));d[2,3]=11000
        self.assertIsNone(sample_depth(d,[2.5,2],5000))
        with self.assertRaises(ValueError):sample_depth(d.astype(float),[2,2],5000)

    def test_image_provenance_error_and_no_state_mutation(self):
        s,_,_=scene(cameras=2,exact=True,quantized=True)
        gray=np.zeros((480,640),np.uint8);d=np.full(gray.shape,5000,np.uint16)
        before=[k.t_wc.copy() for k in s.keyframes]
        with self.assertRaisesRegex(ValueError,'provenance'):propose_pair(s,0,1,gray,gray,d,d)
        for k,p in zip(s.keyframes,before):np.testing.assert_array_equal(k.t_wc,p)

    def test_actual_lk_on_translated_texture_and_constant_depth(self):
        from adaptivefusion.state import Snapshot,Keyframe,Point,Observation
        import cv2
        rng=np.random.default_rng(2);g=rng.integers(0,256,(160,200),dtype=np.uint8)
        h=cv2.warpAffine(g,np.array([[1,0,2],[0,1,0]],float),(200,160))
        points={};obs=[]
        for i,(y,x) in enumerate(( (y,x) for y in range(25,140,15) for x in range(25,175,15))):
            points[i]=Point(i,0,np.array([x/100,y/100,2.]));obs.append(Observation(i,0,i,float(x),float(y),10000))
        s=Snapshot(np.array([100.,100.,100.,80.]),[Keyframe(0,0,1.,200,160,np.eye(4)),Keyframe(1,1,2.,200,160,np.eye(4))],points,obs).validate()
        d=np.full(g.shape,10000,np.uint16);j=propose_pair(s,0,1,g,h,d,d)
        self.assertTrue(j['accepted']);self.assertGreater(len(j['proposals']),50)
        np.testing.assert_allclose(np.array(j['t_target_source'])[:3,3],[.04,0,0],atol=1e-4)
        self.assertEqual(len(s.observations),len(obs))

    def test_entry_refuses_overwrite_before_reading_inputs(self):
        entry=Path(__file__).resolve().parents[2]/'python/audit_bridge_candidates.py'
        with tempfile.TemporaryDirectory() as tmp:
            p=subprocess.run([sys.executable,str(entry),'--snapshot','missing','--dataset','missing','--output',tmp],capture_output=True,text=True)
            self.assertEqual(p.returncode,2);self.assertIn('refuse overwrite',p.stderr)
