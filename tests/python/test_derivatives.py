import unittest
import numpy as np
from fixtures import scene
from adaptivefusion.local_ba import Problem,solve
from adaptivefusion.geometry import block_huber,block_huber_jacobian,so3_right_jacobian


class DerivativeTests(unittest.TestCase):
    def test_block_loss_derivative_inside_outside_and_zero(self):
        for r in [np.array([0.,0.]),np.array([.2,-.4]),np.array([3.,4.]),np.array([-10.])]:
            numerical=np.column_stack([(block_huber(r+np.eye(len(r))[j]*1e-6,2.)-
                                        block_huber(r-np.eye(len(r))[j]*1e-6,2.))/(2e-6) for j in range(len(r))])
            np.testing.assert_allclose(block_huber_jacobian(r,2.),numerical,atol=2e-9,rtol=2e-8)

    def test_full_sparse_jacobian_matches_central_differences(self):
        s,_,_=scene()
        for exact in [False,True]:
            if exact:s,_,_=scene(exact=True)
            p=Problem(s);x=p.x0.copy()
            analytic=p.jacobian(x).toarray()
            numerical=np.column_stack([(p.residual(x+np.eye(len(x))[j]*1e-6)-
                                        p.residual(x-np.eye(len(x))[j]*1e-6))/(2e-6) for j in range(len(x))])
            np.testing.assert_allclose(analytic,numerical,atol=5e-5,rtol=2e-5)
            self.assertTrue(np.all((np.abs(analytic)>1e-12)<=p.sparsity().toarray().astype(bool)))
        np.testing.assert_array_equal(so3_right_jacobian(np.zeros(3)),np.eye(3))

    def test_analytic_and_fd_solve_same_synthetic_problem(self):
        s,_,_=scene()
        a=solve(s,jacobian='analytic',max_nfev=200)
        f=solve(s,max_nfev=200)
        self.assertTrue(a.accepted,a.reason);self.assertTrue(f.accepted,f.reason)
        self.assertLess(abs(a.final['robust_cost']-f.final['robust_cost']),1e-8)
        for k in range(3):np.testing.assert_allclose(a.snapshot.keyframes[k].t_wc,f.snapshot.keyframes[k].t_wc,atol=2e-6)


if __name__=='__main__':unittest.main()
