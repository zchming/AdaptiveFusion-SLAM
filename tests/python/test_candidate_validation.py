import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from scipy.spatial.transform import Rotation
from adaptivefusion.candidate_validation import depth_probe,relative_error,truth_endpoints,pair_value,_depth_mask
from adaptivefusion.geometry import inverse


class CandidateValidationTests(unittest.TestCase):
    def plane(self):return np.full((240,320),10000,np.uint16)
    def probe(self,transform,source=None,target=None,features=None):
        d=self.plane()
        return depth_probe(d if source is None else source,d if target is None else target,[] if features is None else features,[],[200.,200.,160.,120.],[5000.,.1,8.],transform)

    def test_exact_static_depth_and_bidirectional_translation(self):
        t=np.eye(4);j=self.probe(t)
        self.assertTrue(j['depth_consistency_pass']);self.assertEqual(j['median_abs_m'],0)
        t[2,3]=.1
        target=np.full((240,320),10500,np.uint16)
        self.assertTrue(self.probe(t,target=target)['depth_consistency_pass'])
        reverse=depth_probe(target,self.plane(),[],[],[200,200,160,120],[5000,.1,8],inverse(t))
        self.assertTrue(reverse['depth_consistency_pass'])

    def test_wrong_axial_translation_is_not_filtered_away(self):
        t=np.eye(4);t[2,3]=.1;j=self.probe(t)
        self.assertEqual(j['comparisons'],j['source_eligible']);self.assertFalse(j['depth_consistency_pass'])
        self.assertAlmostEqual(j['median_abs_m'],.1);self.assertEqual(j['predicted_behind_fraction'],1)

    def test_planar_tangential_motion_is_a_depth_blind_spot(self):
        t=np.eye(4);t[0,3]=.2;j=self.probe(t)
        self.assertTrue(j['depth_consistency_pass'])
        error=relative_error(t,np.eye(4),np.eye(4));self.assertFalse(error['gt_fidelity_pass'])

    def test_all_native_feature_neighborhoods_are_excluded(self):
        mask=_depth_mask(self.plane(),5000,.1,8,[[8.,8.]])
        self.assertFalse(mask[4:13,4:13].any());self.assertTrue(mask[20,20])
        first=self.probe(np.eye(4));second=self.probe(np.eye(4),features=[[8.,8.]])
        self.assertEqual(first['source_eligible']-second['source_eligible'],1)

    def test_holes_edges_and_outside_predictions_reduce_coverage(self):
        d=self.plane();d[:,100:]=0;j=self.probe(np.eye(4),target=d)
        self.assertLess(j['comparisons'],j['source_eligible']);self.assertFalse(j['depth_consistency_pass'])
        t=np.eye(4);t[0,3]=100;j=self.probe(t)
        self.assertEqual(j['comparisons'],0);self.assertEqual(j['status'],'insufficient_coverage')
        self.assertIsNone(j['median_abs_m']);self.assertFalse(j['depth_consistency_pass'])

    def test_invalid_depth_camera_pose_and_calibration_reject(self):
        with self.assertRaises(ValueError):self.probe(np.eye(4),source=self.plane().astype(float))
        with self.assertRaises(ValueError):depth_probe(self.plane(),self.plane(),[],[],[0,200,160,120],[5000,.1,8],np.eye(4))
        t=np.eye(4);t[0,0]=2
        with self.assertRaises(ValueError):self.probe(t)

    def test_relative_direction_rotation_and_global_gauge_invariance(self):
        a=np.eye(4);b=np.eye(4);b[:3,:3]=Rotation.from_rotvec([.1,.2,0]).as_matrix();b[:3,3]=[.1,.2,.3]
        exact=inverse(b)@a;j=relative_error(exact,a,b)
        self.assertLess(j['translation_m'],1e-12);self.assertLess(j['rotation_deg'],1e-12)
        g=b@b;self.assertEqual(j['gt_fidelity_pass'],relative_error(exact,g@a,g@b)['gt_fidelity_pass'])
        self.assertFalse(relative_error(inverse(exact),a,b)['gt_fidelity_pass'])

    def test_truth_nearest_and_bracketed_interpolation(self):
        times=np.arange(0.,.101,.01);poses=np.repeat(np.eye(4)[None,:,:],len(times),axis=0)
        poses[:,0,3]=times
        for i,t in enumerate(times):poses[i,:3,:3]=Rotation.from_rotvec([0,0,t]).as_matrix()
        j=truth_endpoints(times,poses,.025,.075)
        self.assertIsNotNone(j['nearest']);self.assertIsNotNone(j['interpolated'])
        self.assertAlmostEqual(j['interpolated'][0][0,3],.025)
        self.assertAlmostEqual(Rotation.from_matrix(j['interpolated'][1][:3,:3]).magnitude(),.075)

    def test_truth_never_extrapolates_or_fills_large_gaps(self):
        poses=np.repeat(np.eye(4)[None,:,:],2,axis=0)
        self.assertIsNone(truth_endpoints([0.,1.],poses,.3,.7)['interpolated'])
        j=truth_endpoints([0.,1.],poses,-.1,1.1)
        self.assertIsNone(j['nearest']);self.assertIsNone(j['interpolated'])
        with self.assertRaises(ValueError):truth_endpoints([0.,1.],poses,1.,0.)

    def test_pair_value_requires_absolute_and_relative_gain(self):
        candidate=dict(translation_m=.02,rotation_deg=.5,gt_fidelity_pass=True)
        baseline=dict(translation_m=.03,rotation_deg=.6,gt_fidelity_pass=True)
        self.assertTrue(pair_value(candidate,baseline)['pair_level_value_supported'])
        baseline['translation_m']=.021;self.assertFalse(pair_value(candidate,baseline)['pair_level_value_supported'])
        candidate['gt_fidelity_pass']=False;baseline['translation_m']=.1
        self.assertFalse(pair_value(candidate,baseline)['pair_level_value_supported'])

    def test_cli_pretruth_evidence_model_immutability_and_failed_truth(self):
        import cv2
        from test_frontend_features import sidecar
        from test_landmark_identity import duplicate_scene
        from adaptivefusion.state import file_sha256
        s,_,d=duplicate_scene();entry=Path(__file__).resolve().parents[2]/'python/evaluate_candidate_proposals.py'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);features=root/'features';features.mkdir();matching=root/'matching/descriptor';matching.mkdir(parents=True)
            for k in s.keyframes:
                cv2.imwrite(str(root/f'depth{k.id}.png'),d)
                obs=sorted([o for o in s.observations if o.keyframe_id==k.id],key=lambda o:o.feature_index)
                lines=sidecar(len(obs)).splitlines();lines[1]=f'META fixture {k.id} after_insertion_before_ba'
                lines[2]=f'KEYFRAME {k.id} {k.id} {k.timestamp} {k.timestamp} {k.width} {k.height}'
                lines[3]='CAMERA '+' '.join(map(str,s.camera));lines[7]=f'IMAGES "rgb{k.id}.png" "depth{k.id}.png"'
                for i,o in enumerate(obs):
                    r=lines[i+9].split();r[2:4]=[str(o.u),str(o.v)];r[10]=str(o.point_id);lines[i+9]=' '.join(r)
                (features/f'{k.id}.features').write_text('\n'.join(lines)+'\n')
            header='frame_id timestamp valid reference_id anchor_id '+' '.join(f't{i}' for i in range(16))
            lines=[header]+[f'{k.id} {k.timestamp} 1 {k.id-1} {k.id} '+' '.join(map(str,k.t_wc.ravel())) for k in s.keyframes]
            (root/'frames.txt').write_text('\n'.join(lines)+'\n')
            (root/'truth.txt').write_text('\n'.join(f'{k.timestamp} {-.04*k.id} 0 0 0 0 0 1' for k in s.keyframes)+'\n')
            model=np.eye(4);model[0,3]=.08
            proposal=matching/'0-2.json';proposal.write_text(json.dumps({'source_keyframe':0,'target_keyframe':2,'crossed_cuts':[2],'verification':{'accepted':True,'t_target_source':model.tolist()}}))
            original=file_sha256(proposal)
            cmd=[sys.executable,str(entry),'--matching',str(root/'matching'),'--features',str(features),'--frames',str(root/'frames.txt'),'--dataset',str(root),'--truth',str(root/'truth.txt'),'--output',str(root/'out')]
            run=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr)
            report=json.loads((root/'out/report.json').read_text());case=report['cases'][0]
            self.assertLess(case['truth_errors']['nearest']['candidate']['translation_m'],1e-12)
            self.assertAlmostEqual(case['truth_errors']['nearest']['raw_baseline']['translation_m'],.08)
            depth=json.loads((root/'out/depth-only.json').read_text());self.assertFalse(depth['truth_read'])
            self.assertEqual(file_sha256(root/'out/depth-only.json'),report['depth_only_sha256'])
            self.assertEqual(file_sha256(proposal),original)
            again=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(again.returncode,2)
            cmd[-1]=str(root/'failed');cmd[cmd.index('--truth')+1]=str(root/'missing_truth')
            failed=subprocess.run(cmd,capture_output=True,text=True);self.assertNotEqual(failed.returncode,0)
            self.assertTrue((root/'failed/depth-only.json').exists());self.assertFalse((root/'failed/report.json').exists())
            self.assertEqual(file_sha256(proposal),original)
