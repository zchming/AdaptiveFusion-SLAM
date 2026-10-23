"""Native file contract and metric pose recovery, never real images or GT files."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from adaptivefusion.geometry import project
from adaptivefusion.native_pnp import write_pair, native_fit


class NativePnpInputTests(unittest.TestCase):
    def test_reject_invalid_and_refuse_input_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'input'
            a = np.array([[.1, .2, 2.]])
            for source, ids, seed in [(a, [1, 2], 17), (a, [-1], 17),
                                      (a*np.nan, [1], 17), (a, [1], -1)]:
                with self.assertRaises(ValueError):
                    write_pair(path, source, [[20, 30]], [525, 525, 320, 240], ids, seed)
                self.assertFalse(path.exists())
            write_pair(path, a, [[20, 30]], [525, 525, 320, 240], [7], 17)
            saved = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_pair(path, a, [[20, 30]], [525, 525, 320, 240], [7], 17)
            self.assertEqual(saved, path.read_bytes())

    def test_timeout_exact_fallback_and_preserved_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            raw = np.eye(4); raw[0, 3] = .123
            with patch('adaptivefusion.native_pnp.subprocess.run',
                       side_effect=subprocess.TimeoutExpired('native', 60)):
                result = native_fit('unused', np.ones((6, 3)), np.zeros((6, 2)),
                                    [525, 525, 320, 240], np.arange(6), raw, 17,
                                    Path(d)/'arm')
            self.assertFalse(result['applied']); self.assertTrue(result['timed_out'])
            np.testing.assert_array_equal(result['t_target_source'], raw)
            self.assertTrue((Path(d)/'arm/input.pnp').is_file())
            self.assertTrue((Path(d)/'arm/fit.json').is_file())


@unittest.skipUnless(os.environ.get('AFS_NATIVE_PNP'), 'AFS_NATIVE_PNP not supplied')
class NativePnpExecutionTests(unittest.TestCase):
    def fixture(self, camera):
        a = np.random.default_rng(13).uniform([-.7, -.5, 1.], [.7, .5, 4.], (80, 3))
        truth = np.eye(4)
        truth[:3, :3] = Rotation.from_rotvec([.03, -.02, .01]).as_matrix()
        truth[:3, 3] = [.07, -.04, .025]
        uv = project(camera, a@truth[:3, :3].T+truth[:3, 3])
        return a, uv, truth

    def fit(self, directory, camera, a, uv, raw=None, seed=17):
        return native_fit(os.environ['AFS_NATIVE_PNP'], a, uv, camera,
                          np.arange(len(a))*2+100, np.eye(4) if raw is None else raw,
                          seed, directory)

    def test_both_cameras_recover_direction_and_metre_translation(self):
        for k in [[517.3, 516.5, 318.6, 255.3], [525, 525, 319.5, 239.5]]:
            with tempfile.TemporaryDirectory() as d:
                a, uv, truth = self.fixture(k); saved = [x.copy() for x in (a, uv, truth)]
                r = self.fit(Path(d)/'arm', k, a, uv)
                self.assertTrue(r['applied'], r)
                self.assertTrue(r['opencv_version'].startswith('4.6.'))
                np.testing.assert_allclose(r['t_target_source'], truth, atol=2e-6)
                self.assertLess(abs(np.array(r['t_target_source'])[0, 3]-.07), 1e-6)
                for x, y in zip((a, uv, truth), saved): np.testing.assert_array_equal(x, y)

    def test_outliers_identity_and_repeatability(self):
        with tempfile.TemporaryDirectory() as d:
            k = [525, 525, 319.5, 239.5]; a, uv, truth = self.fixture(k)
            uv[:12] += [100, -80]
            one = self.fit(Path(d)/'one', k, a, uv, seed=41)
            two = self.fit(Path(d)/'two', k, a, uv, seed=41)
            self.assertTrue(one['applied'], one)
            np.testing.assert_allclose(one['t_target_source'], truth, atol=2e-6)
            self.assertEqual(one['t_target_source'], two['t_target_source'])
            self.assertEqual(one['inlier_source_ids'], two['inlier_source_ids'])
            self.assertTrue(set(one['inlier_source_ids']) <= set(np.arange(12, 80)*2+100))

    def test_native_rejection_exact_nonidentity_fallback_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            k = [525, 525, 319.5, 239.5]; a, uv, raw = self.fixture(k)
            r = self.fit(Path(d)/'arm', k, a[:5], uv[:5], raw)
            self.assertEqual(r['process_returncode'], 0)
            self.assertFalse(r['applied'])
            np.testing.assert_array_equal(r['t_target_source'], raw)
            report = Path(d)/'arm/native/report.json'; saved = report.read_bytes()
            proc = subprocess.run([os.environ['AFS_NATIVE_PNP'], '--input',
                                   str(Path(d)/'arm/input.pnp'), '--output', str(report.parent)],
                                  capture_output=True)
            self.assertEqual(proc.returncode, 2); self.assertEqual(saved, report.read_bytes())

    def test_malformed_native_input_fails_without_running_estimator(self):
        with tempfile.TemporaryDirectory() as d:
            k = [525, 525, 319.5, 239.5]; a, uv, truth = self.fixture(k)
            path = Path(d)/'input'; write_pair(path, a[:6], uv[:6], k, np.arange(6), 17)
            original = path.read_text()
            changes = [original.replace('C 1 ', 'C 0 '), original.replace('SEED 17', 'SEED -1'),
                       original.replace('END', ''), original+'EXTRA\n',
                       original.replace('CAMERA 525', 'CAMERA nan', 1)]
            for index, text in enumerate(changes):
                bad = Path(d)/f'bad{index}'; bad.write_text(text); output = Path(d)/f'out{index}'
                p = subprocess.run([os.environ['AFS_NATIVE_PNP'], '--input', str(bad),
                                    '--output', str(output)], capture_output=True)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertFalse(output.exists())
