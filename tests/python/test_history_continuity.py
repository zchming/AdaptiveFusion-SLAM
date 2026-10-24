import unittest
import numpy as np
from fixtures import scene
from adaptivefusion.trajectory import FrameRecord,correct_history_continuous
from adaptivefusion.geometry import inverse


class HistoryContinuityTests(unittest.TestCase):
    def test_anchor_gauge_endpoints_interpolation_and_tail_relative_motion(self):
        before,_,_=scene();after=before.clone()
        for k in after.keyframes[1:]:k.t_wc[0,3]+=.02*k.id
        records=[]
        for i,t in enumerate([.1,.6,1.1,1.6,2.1,2.6,3.1]):
            pose=np.eye(4);pose[0,3]=.1*i
            # Match raw keyframe source IDs 0,2,4, but raw poses differ from map.
            records.append(FrameRecord(i,t,True,i-1,min(i//2,2),pose))
        output=correct_history_continuous(records,before,after)
        gauge=records[0].t_wc@inverse(before.keyframes[0].t_wc)
        for k in after.keyframes:
            np.testing.assert_allclose(output[k.source_frame_id][1],gauge@k.t_wc,atol=1e-12)
        np.testing.assert_allclose(inverse(output[-2][1])@output[-1][1],
                                   inverse(records[-2].t_wc)@records[-1].t_wc,atol=1e-12)
        # A missing frame remains absent, not interpolated into a valid pose.
        records[3].valid=False
        self.assertEqual(len(correct_history_continuous(records,before,after)),6)

    def test_unchanged_window_does_not_import_global_map_gauge(self):
        before,_,_=scene();after=before.clone()
        records=[FrameRecord(k.source_frame_id,k.timestamp,True,-1,k.id,k.t_wc.copy()) for k in before.keyframes]
        for k in before.keyframes:k.t_wc[0,3]+=2
        for k in after.keyframes:k.t_wc[0,3]+=2
        result=correct_history_continuous(records,before,after)
        for f,(_,pose) in zip(records,result):np.testing.assert_allclose(pose,f.t_wc,atol=1e-12)


if __name__=='__main__':unittest.main()
