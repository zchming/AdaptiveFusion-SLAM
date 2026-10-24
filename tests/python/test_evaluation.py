import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from adaptivefusion.geometry import inverse
from adaptivefusion.trajectory import write_tum
from adaptivefusion.evaluation import evaluate, fixed_interval_pairs, associate, read_tum


class EvaluationTests(unittest.TestCase):
    def test_se3_alignment_fixed_interval_and_coverage(self):
        transform=np.eye(4)
        transform[:3,:3]=Rotation.from_rotvec([.1,-.2,.3]).as_matrix()
        transform[:3,3]=[1,-2,.5]
        truth=[]
        for i in range(31):
            t=i*.1;pose=np.eye(4)
            pose[:3,3]=[t,np.sin(t),.2*t*t]
            pose[:3,:3]=Rotation.from_rotvec([0,.1*t,0]).as_matrix()
            truth.append((t,pose))
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);write_tum(d/'gt',truth)
            write_tum(d/'estimate',[(t,inverse(transform)@pose) for t,pose in truth])
            report=evaluate(d/'gt',d/'estimate')
            self.assertLess(report['ate_rmse_m'],1e-10)
            self.assertEqual(report['fixed_interval_rpe']['pairs'],21)
            self.assertLess(report['fixed_interval_rpe']['translation_rmse_m'],1e-10)
            self.assertLess(report['fixed_interval_rpe']['rotation_rmse_rad'],1e-10)
            self.assertEqual(report['duration_coverage_ratio'],1)
            self.assertEqual(report['jump_indices'],[])
            write_tum(d/'partial',truth[5:25])
            partial=evaluate(d/'gt',d/'partial')
            self.assertAlmostEqual(partial['duration_coverage_ratio'],1.9/3)

    def test_scale_error_not_aligned_away(self):
        truth=[]
        for i in range(20):
            pose=np.eye(4);pose[:3,3]=[i*.05,np.sin(i*.2),np.cos(i*.3)]
            truth.append((i*.1,pose))
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);write_tum(d/'gt',truth)
            scaled=[]
            for t,p in truth:
                q=p.copy();q[:3,3]*=2;scaled.append((t,q))
            write_tum(d/'estimate',scaled)
            report=evaluate(d/'gt',d/'estimate')
            self.assertGreater(report['ate_rmse_m'],.1)
            self.assertGreater(report['fixed_interval_rpe']['translation_rmse_m'],.1)

    def test_interval_not_adjacent_and_missing_pairs(self):
        times=np.array([0.,.033,.067,1.,1.034,2.])
        self.assertEqual(fixed_interval_pairs(times),[(0,3),(1,4),(3,5)])
        self.assertEqual(fixed_interval_pairs(np.array([0.,.1,.2])),[])
        self.assertEqual(associate(np.array([0.,.04,.08]),np.array([0.,.001,.041,.081])),[(0,0),(1,2),(2,3)])

    def test_invalid_trajectory_and_jump_detection(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            (d/'invalid').write_text('0 0 0 0 0 0 0 0\n')
            with self.assertRaises(ValueError):read_tum(d/'invalid')
            poses=[]
            for i in range(20):
                p=np.eye(4);p[:3,3]=[i*.001,np.sin(i*.1)*.01,0];poses.append((i*.05,p))
            write_tum(d/'gt',poses)
            changed=[(t,p.copy()) for t,p in poses];changed[5][1][0,3]+=1
            write_tum(d/'estimate',changed)
            report=evaluate(d/'gt',d/'estimate')
            self.assertEqual(report['jump_indices'],[5,6])
            self.assertIsNone(report['fixed_interval_rpe']['translation_rmse_m'])

    def test_large_unix_timestamps_roundtrip_without_millisecond_loss(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trajectory'
            timestamp=1305031102.175304
            write_tum(path,[(timestamp,np.eye(4)),(timestamp+.033,np.eye(4))])
            times,_=read_tum(path)
            self.assertEqual(times[0],timestamp)
            self.assertEqual(times[1],timestamp+.033)


if __name__=='__main__':unittest.main()
