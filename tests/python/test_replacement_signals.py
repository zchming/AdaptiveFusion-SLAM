import unittest
import numpy as np
from adaptivefusion.replacement_signals import fit_partition, closure, common_check


class ReplacementSignalsTest(unittest.TestCase):
    def test_partition_fixed_and_complementary(self):
        pixels = np.array([[10,10],[170,10],[10,130],[170,130]])
        self.assertEqual(fit_partition(pixels).tolist(),[True,False,False,True])

    def test_reverse_cycle_direction(self):
        from scipy.spatial.transform import Rotation
        forward=np.eye(4); forward[:3,:3]=Rotation.from_euler('z',30,degrees=True).as_matrix()
        forward[:3,3]=[1,2,3]
        signal=closure(forward,np.linalg.inv(forward))
        self.assertLess(signal['translation_m'],1e-12)
        self.assertLess(signal['rotation_deg'],1e-12)

    def test_common_positive_depth(self):
        raw=np.eye(4); candidate=np.eye(4);candidate[2,3]=-2
        result=common_check(raw,candidate,[[0,0,1],[0,0,3]],[[0,0],[0,0]],[1,1,0,0])
        self.assertEqual(result['count'],2)
        self.assertEqual(result['positive_count'],1)
        self.assertEqual(result['raw']['median_px'],0)


if __name__=='__main__':unittest.main()
