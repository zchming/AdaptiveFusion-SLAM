"""Causal controls: preserve source quality while removing target-depth gating."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import run_orb_observation_controls as runner


class OrbObservationControlsTests(unittest.TestCase):
    def test_invalid_target_depth_is_retained_only_in_source_arm(self):
        feature = SimpleNamespace(keypoints=np.array([[3., 3.], [8., 8.]]))
        source = np.full((12, 12), 5000, np.uint16)
        target = source.copy(); target[7, 7] = 0
        rows = runner.eligible_rows([(0, 0, 20), (1, 1, 21)], feature, feature, source, target)
        self.assertEqual([r['source_feature'] for r in rows], [0, 1])
        self.assertEqual([r['source_feature'] for r in rows if r['target_depth_valid']], [0])
        self.assertEqual([r['source_depth_m'] for r in rows], [1., 1.])

    def test_source_hole_and_discontinuity_still_reject(self):
        feature = SimpleNamespace(keypoints=np.array([[3., 3.], [8., 8.]]))
        target = np.full((12, 12), 5000, np.uint16)
        source = target.copy(); source[2, 2] = 0; source[7, 7] = 5300
        self.assertEqual(runner.eligible_rows([(0, 0, 10), (1, 1, 10)], feature, feature, source, target), [])

    def test_all_valid_observations_preserve_odd_and_even_feature_ids(self):
        feature = SimpleNamespace(keypoints=np.array([[3., 3.], [8., 8.]]))
        depth = np.full((12, 12), 5000, np.uint16)
        rows = runner.eligible_rows([(0, 1, 23), (1, 0, 32)], feature, feature, depth, depth)
        self.assertEqual([(r['source_feature'], r['target_feature'], r['hamming']) for r in rows],
                         [(0, 1, 23), (1, 0, 32)])

    def test_duplicate_source_observations_rejected(self):
        feature = SimpleNamespace(keypoints=np.array([[3., 3.], [8., 8.]]))
        depth = np.full((12, 12), 5000, np.uint16)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            runner.eligible_rows([(0, 0, 10), (0, 1, 11)], feature, feature, depth, depth)

    def test_existing_output_stops_before_inputs_or_fitting(self):
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'build_cases') as build:
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                runner.run(SimpleNamespace(output=Path(d)))
            build.assert_not_called()


if __name__ == '__main__':
    unittest.main()
