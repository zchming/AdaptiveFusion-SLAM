import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from adaptivefusion.geometry import inverse, project
from adaptivefusion.proposal_diagnostics import (
    fixed_model_errors, support_geometry, epipolar_errors, centroid_decomposition)


class ProposalDiagnosticsTests(unittest.TestCase):
    def scene(self):
        rng = np.random.default_rng(24)
        a = rng.uniform([-.4, -.3, 1.5], [.4, .3, 2.5], (40, 3))
        t = np.eye(4)
        t[:3, :3] = Rotation.from_rotvec([.01, .03, -.02]).as_matrix()
        t[:3, 3] = [.1, -.04, .02]
        return a, a@t[:3, :3].T+t[:3, 3], t, [500, 500, 320, 240]

    def test_exact_transform_and_wrong_direction(self):
        a, b, t, k = self.scene()
        exact = fixed_model_errors(t, a, b, project(k, b), k)
        self.assertLess(exact['metric_m']['p95'], 1e-12)
        self.assertLess(exact['reprojection_px']['p95'], 1e-10)
        self.assertGreater(fixed_model_errors(inverse(t), a, b, project(k, b), k)['metric_m']['median'], .1)

    def test_epipolar_ignores_depth_scale_but_detects_off_line_pixels(self):
        a, b, t, k = self.scene()
        uv_a, uv_b = project(k, a), project(k, b)
        result = epipolar_errors(t, uv_a, uv_b, k)
        self.assertLess(result['measured']['p95'], 1e-10)
        bad = epipolar_errors(t, uv_a, uv_b+[0, 20], k)
        self.assertGreater(bad['measured']['median'], 5)
        wrong_depth = fixed_model_errors(t, a, b*1.1, uv_b, k)
        self.assertGreater(wrong_depth['metric_m']['median'], .1)
        self.assertLess(wrong_depth['reprojection_px']['median'], 1e-10)

    def test_pure_rotation_epipolar_unassessed_and_behind_camera_counted(self):
        a, b, t, k = self.scene()
        self.assertEqual(epipolar_errors(np.eye(4), project(k, a), project(k, a), k)['status'], 'zero_translation_unassessed')
        t = np.eye(4); t[2, 3] = -10
        errors = fixed_model_errors(t, a, b, project(k, b), k)
        self.assertEqual(errors['nonpositive_predictions'], len(a))
        self.assertIsNone(errors['reprojection_px']['median'])
        self.assertEqual(errors['metric_m']['count'], len(a))

    def test_rotation_translation_compensation_is_exact(self):
        a, _, _, _ = self.scene()
        t = np.eye(4); t[:3, :3] = Rotation.from_rotvec([0, .1, 0]).as_matrix()
        t[:3, 3] = -(t[:3, :3]-np.eye(3))@a.mean(0)
        d = centroid_decomposition(t, np.eye(4), a)
        self.assertLess(d['centroid_action_delta_m'], 1e-12)
        self.assertGreater(d['translation_delta_m'], .1)
        np.testing.assert_allclose(np.array(d['rotation_lever_vector_m'])+d['translation_delta_vector_m'], d['centroid_action_delta_vector_m'])

    def test_support_spectrum_cells_and_invalid_inputs(self):
        uv = np.array([[10, 10], [310, 10], [10, 230], [310, 230.]])
        xyz = np.c_[uv/100, np.ones(4)]
        s = support_geometry(xyz, uv, 320, 240)
        self.assertEqual(s['occupied_4x4_cells'], 4)
        self.assertAlmostEqual(s['centered_rms_axes_m'][-1], 0)
        with self.assertRaises(ValueError): support_geometry(xyz, uv-100, 320, 240)
        with self.assertRaises(ValueError): centroid_decomposition(np.eye(4), np.eye(4), [])
        with self.assertRaises(ValueError): fixed_model_errors(np.eye(4), xyz, xyz[:2], uv, [500, 500, 160, 120])
