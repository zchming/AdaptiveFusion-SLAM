"""Frozen input identities, no GT read before fitting, and sealed evaluation."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import FrameRecord

ENTRY = Path(__file__).resolve().parents[2]/'python/run_camera_pnp_comparison.py'
spec = importlib.util.spec_from_file_location('camera_comparison', ENTRY)
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class CameraPnpComparisonTests(unittest.TestCase):
    def fixture(self, root):
        (root/'matching/descriptor').mkdir(parents=True)
        (root/'features').mkdir(); (root/'data').mkdir()
        Image.fromarray(np.full((12, 12), 5000, np.uint16)).save(root/'data/depth.png')
        (root/'frames').write_text('mock frame fixture')
        frames = [FrameRecord(i, float(i+1), True, -1, i, np.eye(4)) for i in range(2)]
        features = []
        for kid in range(2):
            (root/'features'/f'{kid}.features').write_text('mock feature fixture')
            features.append(SimpleNamespace(
                metadata=dict(keyframe_id=kid, source_frame_id=kid, rgb_timestamp=float(kid+1),
                              camera=[517.3, 516.5, 318.6, 255.3], depth=[5000., .1, 8.],
                              images=['rgb.png', 'depth.png'], height=12, width=12),
                keypoints=np.array([[2+i, 2+i] for i in range(6)], float)))
        proposal = dict(source_keyframe=0, target_keyframe=1, depth_valid_matches=6,
                        records=[dict(source_feature=i, target_feature=i) for i in range(6)])
        path = root/'matching/descriptor/0-1.json'; path.write_text(json.dumps(proposal))
        p = dict(audit_summary=[dict(scene='synthetic', feature_cameras=[features[0].metadata['camera']],
                 pairs=[dict(pair='0-1', sha256=file_sha256(path), records=6, even_fit_rows=3, odd_check_rows=3)])])
        sources = {'synthetic':dict(matching=str(root/'matching'), features=str(root/'features'),
                                    frames=str(root/'frames'), dataset=str(root/'data'),
                                    truth=str(root/'must-not-open-truth'))}
        return p, sources, frames, features

    def prepare(self, protocol, sources, frames, features):
        with patch.object(runner, 'load_frames', return_value=frames), patch.object(
                runner, 'read_features', side_effect=lambda p: features[int(p.stem)]):
            return runner.prepare(protocol, sources)

    def test_preparation_preserves_split_pixels_depths_and_never_needs_truth(self):
        with tempfile.TemporaryDirectory() as d:
            p, sources, frames, features = self.fixture(Path(d))
            cases, hashes = self.prepare(p, sources, frames, features)
            self.assertEqual(cases[0]['ids'], list(range(6)))
            self.assertEqual(cases[0]['source_depth_m'], [1.]*6)
            self.assertEqual(cases[0]['source_pixels'], features[0].keypoints.tolist())
            self.assertFalse(any('truth' in path for path in hashes))
            runner.check_hashes(hashes)
            Path(sources['synthetic']['frames']).write_text('changed')
            with self.assertRaisesRegex(ValueError, 'input changed'): runner.check_hashes(hashes)

    def test_inventory_hash_split_and_calibration_tampering_stop_before_fit(self):
        with tempfile.TemporaryDirectory() as d:
            p, sources, frames, features = self.fixture(Path(d))
            p['audit_summary'][0]['pairs'][0]['even_fit_rows'] = 4
            with self.assertRaisesRegex(ValueError, 'identity/split'): self.prepare(p, sources, frames, features)
            p['audit_summary'][0]['pairs'][0]['even_fit_rows'] = 3
            features[1].metadata['camera'] = [525, 525, 319.5, 239.5]
            with self.assertRaisesRegex(ValueError, 'calibration'): self.prepare(p, sources, frames, features)
            features[1].metadata['camera'] = features[0].metadata['camera']
            (Path(d)/'matching/descriptor/0-1.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'proposal hash'): self.prepare(p, sources, frames, features)

    def test_evaluation_refuses_overwrite_and_changed_fit_before_reading_truth(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); study = root/'study'; study.mkdir()
            (study/'fit-only.json').write_text('{}')
            (study/'fit-seal.json').write_text(json.dumps({'fit-only.json':'0'*64}))
            args = SimpleNamespace(output=root/'out', study=study, sources=root/'missing')
            with self.assertRaisesRegex(ValueError, 'seal mismatch'): runner.evaluate(args)
            self.assertFalse(args.output.exists())
            args.output.mkdir()
            with self.assertRaisesRegex(ValueError, 'output exists'): runner.evaluate(args)

    def test_unavailable_truth_remains_an_explicit_row(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); study = root/'study'; study.mkdir()
            fit = dict(cases=[dict(scene='synthetic', pair='0-1', source_timestamp=1., target_timestamp=2.,
                       raw_baseline=np.eye(4).tolist(), arms=[dict(camera_profile='historical',
                       estimator=runner.LOCAL, seed=None, fit=dict(applied=False, reason='insufficient',
                       t_target_source=np.eye(4).tolist()), process_returncode=0, timed_out=False,
                       process_wall_s=.1, peak_rss_kib=1, check_pixels=dict(count=0, positive_count=0,
                       median_px=None))])], input_sha256={})
            runner.write_json(study/'fit-only.json', fit)
            runner.write_json(study/'fit-seal.json', {'fit-only.json':file_sha256(study/'fit-only.json')})
            sources = root/'sources.json'; runner.write_json(sources, {'synthetic':{'truth':str(root/'missing')}})
            runner.evaluate(SimpleNamespace(output=root/'out', study=study, sources=sources))
            result = json.loads((root/'out/report.json').read_text())
            self.assertIn('synthetic', result['unavailable_truth'])
            self.assertEqual(len(result['rows']), 2)
            self.assertTrue(all(r['translation_m'] is None for r in result['rows']))

    def test_empty_and_behind_camera_check_rows_are_accounted(self):
        t = np.eye(4)
        value = runner.pixel_check(t, np.empty((0, 3)), np.empty((0, 2)), [525, 525, 320, 240])
        self.assertEqual(value['count'], 0); self.assertIsNone(value['median_px'])
        value = runner.pixel_check(t, np.array([[0., 0., -1.]]), np.zeros((1, 2)), [525, 525, 320, 240])
        self.assertEqual(value['count'], 1); self.assertEqual(value['positive_count'], 0)
        self.assertIsNone(value['median_px'])
