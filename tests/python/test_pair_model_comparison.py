import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from adaptivefusion.geometry import project
from adaptivefusion.pair_model_comparison import refine_pair


class PairModelComparisonTests(unittest.TestCase):
    def scene(self):
        rng = np.random.default_rng(32)
        a = rng.uniform([-.5, -.4, 1.], [.5, .4, 3.], (60, 3))
        truth = np.eye(4); truth[:3, :3] = Rotation.from_rotvec([.02, -.03, .01]).as_matrix()
        truth[:3, 3] = [.05, -.02, .03]
        b = a@truth[:3, :3].T+truth[:3, 3]
        k = [517.3, 516.5, 318.6, 255.3]
        return a, b, project(k, b), k, truth

    def test_noiseless_recovery_both_models_and_no_mutation(self):
        a, b, uv, k, truth = self.scene(); initial = np.eye(4)
        saved = [x.copy() for x in [a, b, uv, initial]]
        for model in ['3d3d', '3d2d']:
            r = refine_pair(a, uv, k, initial, model=model, target_points=b)
            self.assertTrue(r['applied'], r)
            np.testing.assert_allclose(r['t_target_source'], truth, atol=1e-7)
        for x, y in zip([a, b, uv, initial], saved): np.testing.assert_array_equal(x, y)

    def test_target_depth_perturbation_affects_only_3d3d_objective(self):
        a, b, uv, k, truth = self.scene(); shifted = b+[0, 0, .05]
        p0 = refine_pair(a, uv, k, np.eye(4), model='3d2d', target_points=b)
        p1 = refine_pair(a, uv, k, np.eye(4), model='3d2d', target_points=shifted)
        np.testing.assert_array_equal(p0['t_target_source'], p1['t_target_source'])
        other = refine_pair(a, uv, k, np.eye(4), model='3d3d', target_points=shifted)
        self.assertTrue(other['applied'], other)
        self.assertAlmostEqual(np.array(other['t_target_source'])[2, 3]-truth[2, 3], .05, places=6)
        self.assertGreater(np.linalg.norm(np.array(other['t_target_source'])-truth), .049)

    def test_noisy_projection_improves_independent_clean_pixels(self):
        a, b, uv, k, truth = self.scene()
        noisy = uv.copy()+np.random.default_rng(2).normal(0, .3, uv.shape)
        noisy[:3] += 30
        r = refine_pair(a, noisy, k, np.eye(4), model='3d2d')
        self.assertTrue(r['applied'], r)
        t = np.array(r['t_target_source'])
        error = np.median(np.linalg.norm(project(k, a@t[:3, :3].T+t[:3, 3])-uv, axis=1))
        original = np.median(np.linalg.norm(project(k, a)-uv, axis=1))
        self.assertLess(error, .5); self.assertLess(error, original)

    def test_insufficient_collinear_invalid_target_and_behind_roll_back(self):
        a, b, uv, k, truth = self.scene()
        for source, pixels, target in [(a[:3], uv[:3], b[:3]),
                                      (np.c_[np.zeros(60), np.zeros(60), np.linspace(1, 3, 60)], uv, b),
                                      (a, uv, b*np.nan)]:
            r = refine_pair(source, pixels, k, truth, model='3d3d', target_points=target)
            self.assertFalse(r['applied']); np.testing.assert_array_equal(r['t_target_source'], truth)
        bad = np.eye(4); bad[2, 3] = -10
        r = refine_pair(a, uv, k, bad, model='3d2d')
        self.assertFalse(r['applied']); np.testing.assert_array_equal(r['t_target_source'], bad)

    def test_solver_budget_failure_completely_rolls_back(self):
        a, b, uv, k, truth = self.scene()
        fake = SimpleNamespace(nfev=100, cost=0., status=0, message='budget', success=False)
        with patch('adaptivefusion.pair_model_comparison.least_squares', return_value=fake):
            r = refine_pair(a, uv, k, np.eye(4), model='3d2d')
        self.assertFalse(r['applied']); self.assertIn('converge', r['reason'])
        np.testing.assert_array_equal(r['t_target_source'], np.eye(4))
