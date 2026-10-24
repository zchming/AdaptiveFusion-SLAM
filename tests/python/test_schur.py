import unittest
import numpy as np
from scipy.linalg import block_diag
from fixtures import scene
from adaptivefusion.local_ba import Problem,solve
from adaptivefusion.schur import normal_blocks,schur_step


class SchurTests(unittest.TestCase):
    def test_reduced_step_matches_full_normal_system(self):
        s,_,_=scene();p=Problem(s)
        blocks=normal_blocks(p,p.x0);a,b,c,gc,gp=blocks
        cross=np.concatenate([b[i] for i in range(len(b))],axis=1)
        full=np.block([[a,cross],[cross.T,block_diag(*c)]])
        damping=.01
        full+=damping*np.diag(np.maximum(np.diag(full),1e-8))
        expected=np.linalg.solve(full,-np.r_[gc,gp.ravel()])
        np.testing.assert_allclose(schur_step(blocks,damping),expected,atol=2e-11,rtol=2e-9)
        gradient=np.r_[gc,gp.ravel()]
        np.testing.assert_allclose(gradient,p.jacobian(p.x0).T@p.residual(p.x0),atol=2e-10)
        sparse=normal_blocks(p,p.x0,sparse_camera=True)
        np.testing.assert_allclose(schur_step(sparse,damping),expected,atol=2e-11,rtol=2e-9)

    def test_metric_recovery_and_no_input_mutation(self):
        s,truth,_=scene(quantized=False)
        result=solve(s,method='schur-lm')
        self.assertTrue(result.accepted,result.reason)
        np.testing.assert_array_equal(result.snapshot.keyframes[0].t_wc,s.keyframes[0].t_wc)
        self.assertLess(result.final['robust_cost'],result.initial['robust_cost']*1e-6)
        for k in range(3):np.testing.assert_allclose(result.snapshot.keyframes[k].t_wc,truth[k],atol=1e-6)

    def test_budget_failure_returns_original(self):
        s,_,_=scene()
        result=solve(s,method='schur-lm',max_nfev=1)
        self.assertFalse(result.accepted)
        for a,b in zip(s.keyframes,result.snapshot.keyframes):np.testing.assert_array_equal(a.t_wc,b.t_wc)

    def test_usable_budget_is_explicitly_distinct_from_convergence(self):
        s,_,_=scene();s.config.maximum_iterations=1
        strict=solve(s,method='schur-lm',acceptance_policy='converged')
        usable=solve(s,method='schur-lm',acceptance_policy='usable')
        self.assertFalse(strict.accepted)
        self.assertTrue(usable.accepted,usable.reason)
        self.assertEqual(usable.diagnostics['termination_status'],0)
        self.assertLess(usable.final['robust_cost'],usable.initial['robust_cost'])


if __name__=='__main__':unittest.main()
