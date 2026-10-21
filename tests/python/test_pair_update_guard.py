import unittest
import numpy as np
from adaptivefusion.geometry import project
from adaptivefusion.pair_update_guard import guard_update


class PairUpdateGuardTests(unittest.TestCase):
    def scene(self):
        rng = np.random.default_rng(14)
        source = rng.uniform([-.4, -.3, 1.], [.4, .3, 2.], (20, 3))
        camera = np.array([500., 500., 320., 240.])
        truth = np.eye(4); truth[0, 3] = .03
        pixels = project(camera, source+truth[:3, 3])
        return source, pixels, camera, truth

    def test_good_update_passes_without_mutating_inputs(self):
        source, pixels, camera, candidate = self.scene(); initial = np.eye(4)
        saved = [x.copy() for x in [source, pixels, camera, candidate, initial]]
        result = guard_update(initial, candidate, source, pixels, camera, solver_applied=True)
        self.assertTrue(result['accepted']); self.assertLess(result['candidate_check']['median_px'], 1e-10)
        np.testing.assert_array_equal(result['selected_t_target_source'], candidate)
        for x, y in zip([source, pixels, camera, candidate, initial], saved): np.testing.assert_array_equal(x, y)

    def test_large_error_rejected_even_if_it_improves_very_bad_initial(self):
        source, pixels, camera, candidate = self.scene()
        initial = np.eye(4); initial[0, 3] = .5
        candidate[0, 3] = .2
        result = guard_update(initial, candidate, source, pixels, camera, solver_applied=True)
        self.assertFalse(result['accepted']); self.assertIn('bound', result['reason'])
        np.testing.assert_array_equal(result['selected_t_target_source'], initial)

    def test_small_error_rejected_if_it_worsens_initial(self):
        source, pixels, camera, initial = self.scene(); candidate = initial.copy(); candidate[0, 3] += .002
        result = guard_update(initial, candidate, source, pixels, camera, solver_applied=True)
        self.assertLess(result['candidate_check']['median_px'], 4)
        self.assertFalse(result['accepted']); self.assertIn('worsens', result['reason'])
        np.testing.assert_array_equal(result['selected_t_target_source'], initial)

    def test_solver_failure_and_too_few_pixels_roll_back(self):
        source, pixels, camera, candidate = self.scene(); initial = np.eye(4)
        for count, applied in [(20, False), (5, True), (0, True)]:
            result = guard_update(initial, candidate, source[:count], pixels[:count], camera, solver_applied=applied)
            self.assertFalse(result['accepted'])
            np.testing.assert_array_equal(result['selected_t_target_source'], initial)

    def test_invalid_pose_observations_and_behind_camera_roll_back(self):
        source, pixels, camera, candidate = self.scene(); initial = np.eye(4)
        bad = candidate.copy(); bad[0, 0] = 2
        result = guard_update(initial, bad, source, pixels, camera, solver_applied=True)
        self.assertFalse(result['accepted'])
        candidate[2, 3] = -10
        result = guard_update(initial, candidate, source, pixels, camera, solver_applied=True)
        self.assertFalse(result['accepted']); self.assertIn('nonpositive', result['reason'])
        result = guard_update(initial, np.eye(4), source*np.nan, pixels, camera, solver_applied=True)
        self.assertFalse(result['accepted'])

    def test_low_median_can_hide_half_bad_correspondences(self):
        source, pixels, camera, candidate = self.scene(); initial = np.eye(4)
        pixels[:8] += 100
        result = guard_update(initial, candidate, source, pixels, camera, solver_applied=True)
        self.assertTrue(result['accepted'])
        # Deliberate limitation: the frozen median rule is not an identity or tail-error certificate.
        self.assertLess(result['candidate_check']['median_px'], 1e-10)

    def test_cli_decisions_precede_evaluation_and_missing_evaluation_preserves_them(self):
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from test_frontend_features import sidecar
        from adaptivefusion.state import file_sha256
        entry = Path(__file__).resolve().parents[2]/'python/check_pair_updates.py'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); features = root/'features'; features.mkdir()
            matching = root/'matching/descriptor'; matching.mkdir(parents=True)
            hashes = {}
            for kid in [0, 1]:
                lines = sidecar(20).splitlines()
                lines[1] = f'META fixture {kid} after_insertion_before_ba'
                lines[2] = f'KEYFRAME {kid} {kid} {1+kid} {1+kid} 640 480'
                fp = features/f'{kid}.features'; fp.write_text('\n'.join(lines)+'\n')
                hashes['feature/'+fp.name] = file_sha256(fp)
            proposal = matching/'0-1.json'
            proposal.write_text(json.dumps(dict(source_keyframe=0, target_keyframe=1,
                records=[dict(source_feature=i, target_feature=i) for i in range(20)])))
            hashes['proposal/0-1.json'] = file_sha256(proposal)
            pose = np.eye(4).tolist()
            arm = dict(model='3d2d', initialization='raw_baseline',
                       fit=dict(applied=True, t_target_source=pose), truth_errors={'sentinel': 1})
            case = dict(pair='0-1', raw_baseline=pose, fit_source_ids=list(range(0, 20, 2)),
                        validation_source_ids=list(range(1, 20, 2)), arms=[arm], raw_baseline_truth={'sentinel': 2})
            comparison = root/'fit-only.json'
            comparison.write_text(json.dumps(dict(truth_read=False, input_sha256=hashes, cases=[case])))
            evaluation = root/'evaluation.json'
            evaluation.write_text(json.dumps(dict(fit_only_sha256=file_sha256(comparison), cases=[case])))
            cmd = [sys.executable, str(entry), '--comparison', str(comparison), '--matching', str(root/'matching'),
                   '--features', str(features), '--evaluation', str(evaluation), '--output', str(root/'first')]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            first = root/'first/guard-only.json'
            self.assertFalse(json.loads(first.read_text())['truth_read'])
            self.assertTrue(json.loads(first.read_text())['cases'][0]['decision']['accepted'])
            original = file_sha256(first)
            arm['truth_errors'] = {'sentinel': -999}
            evaluation.write_text(json.dumps(dict(fit_only_sha256=file_sha256(comparison), cases=[case])))
            cmd[-1] = str(root/'second')
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(file_sha256(root/'second/guard-only.json'), original)
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 2)
            cmd[-1] = str(root/'missing'); cmd[cmd.index('--evaluation')+1] = str(root/'absent.json')
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertTrue((root/'missing/guard-only.json').exists())
            self.assertFalse((root/'missing/report.json').exists())
