import unittest
import numpy as np
from fixtures import scene
from adaptivefusion.local_ba import Problem,solve
from adaptivefusion.schur import normal_blocks,schur_step
from scipy.linalg import block_diag


class BoundaryTests(unittest.TestCase):
    def test_external_observations_preserved_and_fixed_pose_derivatives(self):
        s,_,_=scene(cameras=6,quantized=False);s.config.window_size=2
        s.observations=[o for o in s.observations if not(o.point_id==100 and o.keyframe_id==4)]
        window=Problem(s);p=Problem(s,boundary='fixed-external')
        self.assertNotIn(100,window.pids);self.assertIn(100,p.pids)
        self.assertEqual(p.free,[5]);self.assertTrue(any(o.keyframe_id<4 for o in p.obs))
        x=p.x0
        numerical=np.column_stack([(p.residual(x+np.eye(len(x))[j]*1e-6)-p.residual(x-np.eye(len(x))[j]*1e-6))/(2e-6) for j in range(len(x))])
        np.testing.assert_allclose(p.jacobian(x).toarray(),numerical,atol=5e-5,rtol=2e-5)
        blocks=normal_blocks(p,x);a,b,c,gc,gp=blocks
        cross=np.concatenate(list(b),axis=1)
        full=np.block([[a,cross],[cross.T,block_diag(*c)]])
        full+=.01*np.diag(np.maximum(np.diag(full),1e-8))
        np.testing.assert_allclose(schur_step(blocks,.01),np.linalg.solve(full,-np.r_[gc,gp.ravel()]),atol=1e-10)
        r=solve(s,method='schur-lm',acceptance_policy='usable',boundary='fixed-external')
        self.assertTrue(r.accepted,r.reason)
        for i in range(5):np.testing.assert_array_equal(s.keyframes[i].t_wc,r.snapshot.keyframes[i].t_wc)
        self.assertLess(r.final['robust_cost'],r.initial['robust_cost'])


if __name__=='__main__':unittest.main()
