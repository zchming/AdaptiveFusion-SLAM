import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from fixtures import scene
from adaptivefusion.components import keyframe_components
from adaptivefusion.local_ba import Problem, solve
from adaptivefusion.offline import component_history, full_history
from adaptivefusion.state import Point, Observation, write_snapshot, load_snapshot
from adaptivefusion.evaluation import read_tum
from adaptivefusion.trajectory import FrameRecord


def disconnected(*, interleaved=False):
    s, truth, _ = scene(cameras=6, quantized=False)
    s.config.window_size = 6
    selected = {0, 2, 4} if interleaved else {0, 1, 2}
    other = sorted(set(range(6)) - selected)
    for pid, point in list(s.points.items()):
        s.points[pid + 1000] = Point(pid + 1000, other[0], point.xyz.copy())
    s.observations = [Observation(
        o.point_id if o.keyframe_id in selected else o.point_id + 1000,
        o.keyframe_id, o.feature_index, o.u, o.v, o.raw_depth)
        for o in s.observations]
    return s.validate(), truth, selected


def records(s):
    result = []
    for k in s.keyframes:
        result.append(FrameRecord(2*k.id, k.timestamp, True, -1, k.id, k.t_wc.copy()))
        pose = k.t_wc.copy(); pose[0, 3] += .025
        result.append(FrameRecord(2*k.id+1, k.timestamp+.4, True, 2*k.id, k.id, pose))
    return result


class ComponentTests(unittest.TestCase):
    def assert_state_equal(self, a, b):
        self.assertEqual(a.config, b.config)
        self.assertEqual(a.observations, b.observations)
        self.assertEqual(set(a.points), set(b.points))
        for x, y in zip(a.keyframes, b.keyframes):
            np.testing.assert_array_equal(x.t_wc, y.t_wc)
        for pid in a.points:
            np.testing.assert_array_equal(a.points[pid].xyz, b.points[pid].xyz)

    def test_partition_and_noncontiguous_indices(self):
        s, _, selected = disconnected(interleaved=True)
        groups = keyframe_components(s, range(6), s.points)
        self.assertEqual(groups, [[0, 2, 4], [1, 3, 5]])
        p = Problem(s, component_policy='anchor')
        self.assertEqual(set(p.kids), selected)
        self.assertEqual(p.free, [2, 4])
        p.preflight()
        self.assertTrue(all(o.keyframe_id in selected for o in p.obs))

    def test_only_selected_geometry_recovers_metric_pose(self):
        s, truth, selected = disconnected()
        original = s.clone()
        strict = solve(s, method='schur-lm')
        self.assertFalse(strict.accepted)
        self.assertEqual(strict.reason, 'disconnected window')
        fit = solve(s, method='schur-lm', component_policy='anchor')
        self.assertTrue(fit.accepted, fit.reason)
        self.assertTrue(fit.diagnostics['partial_scope'])
        self.assertLess(fit.final['robust_cost'], fit.initial['robust_cost']*1e-6)
        for k in s.keyframes:
            if k.id in selected:
                np.testing.assert_allclose(fit.snapshot.keyframes[k.id].t_wc, truth[k.id], atol=1e-6)
            else:
                np.testing.assert_array_equal(k.t_wc, fit.snapshot.keyframes[k.id].t_wc)
        for pid, point in s.points.items():
            if point.host not in selected:
                np.testing.assert_array_equal(point.xyz, fit.snapshot.points[pid].xyz)
        np.testing.assert_array_equal(s.keyframes[0].t_wc, fit.snapshot.keyframes[0].t_wc)
        self.assertEqual(s.observations, fit.snapshot.observations)
        self.assert_state_equal(s, original)

    def test_failed_subgraph_solve_rolls_back_every_component(self):
        s, _, _ = disconnected()
        fit = solve(s, method='schur-lm', component_policy='anchor', max_nfev=1)
        self.assertFalse(fit.accepted)
        self.assert_state_equal(s, fit.snapshot)

    def test_connected_policy_is_numerically_identical(self):
        s, _, _ = scene(quantized=False)
        a = solve(s, method='schur-lm')
        b = solve(s, method='schur-lm', component_policy='anchor')
        self.assertTrue(a.accepted and b.accepted)
        self.assert_state_equal(a.snapshot, b.snapshot)
        self.assertEqual(a.initial, b.initial)
        self.assertEqual(a.final, b.final)
        self.assertEqual(a.nfev, b.nfev)

    def test_isolated_anchor_and_local_window_are_not_relaxed(self):
        s, _, _ = disconnected()
        s.observations = [o for o in s.observations if o.keyframe_id != 0]
        # Move the host of the first component's points to its first observation.
        for point in s.points.values():
            if point.host == 0: point.host = 1
        fit = solve(s.validate(), method='schur-lm', component_policy='anchor')
        self.assertFalse(fit.accepted)
        self.assert_state_equal(s, fit.snapshot)
        s.config.window_size = 2
        with self.assertRaisesRegex(ValueError, 'full batch'):
            Problem(s, component_policy='anchor')

    def test_history_preserves_excluded_ordinary_frames_and_lost_coverage(self):
        s, _, selected = disconnected(interleaved=True)
        frames = records(s)
        frames[3].valid = False
        poses = {k.id: k.t_wc.copy() for k in s.keyframes}
        for k in selected - {0}: poses[k][0, 3] += .03
        corrected = component_history(frames, s, poses, selected)
        self.assertEqual(len(corrected), len(frames)-1)
        for frame, (timestamp, pose) in zip((f for f in frames if f.valid), corrected):
            self.assertEqual(frame.timestamp, timestamp)
            if frame.anchor_id not in selected:
                np.testing.assert_array_equal(pose, frame.t_wc)
        for kid in selected:
            output = dict(corrected)[s.keyframes[kid].timestamp]
            np.testing.assert_allclose(output, poses[kid], atol=1e-12)
        poses[1][0, 3] += .01
        with self.assertRaisesRegex(ValueError, 'excluded'):
            component_history(frames, s, poses, selected)

    def test_all_selected_history_matches_original_interpolation(self):
        s, _, _ = scene()
        frames = records(s)
        poses = {k.id: k.t_wc.copy() for k in s.keyframes}
        poses[1][0, 3] += .05
        a = full_history(frames, s, poses)
        b = component_history(frames, s, poses, range(3))
        for (ta, pa), (tb, pb) in zip(a, b):
            self.assertEqual(ta, tb)
            np.testing.assert_array_equal(pa, pb)

    def test_partial_cli_and_output_ownership(self):
        s, _, selected = disconnected()
        frames = records(s)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); cache = base/'cache'; cache.mkdir()
            write_snapshot(cache/'42.before.snapshot', s)
            (cache/'complete.json').write_text(json.dumps({
                'processed_frames': len(frames), 'risk_enabled': False,
                'online_writeback': False}))
            with (cache/'frames.txt').open('x') as f:
                f.write('frame_id timestamp valid reference_id anchor_id '+
                        ' '.join(f't{i}' for i in range(16))+'\n')
                for frame in frames:
                    f.write(f'{frame.id} {frame.timestamp:.17g} 1 {frame.reference_id} {frame.anchor_id} '+
                            ' '.join(f'{v:.17g}' for v in frame.t_wc.ravel())+'\n')
            root = Path(__file__).resolve().parents[2]
            command = [sys.executable, str(root/'python/run_batch_reference.py'),
                       '--cache', str(cache), '--output', str(base/'result'),
                       '--component-policy', 'anchor']
            process = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
            report = json.loads((base/'result/report.json').read_text())
            self.assertEqual(report['status'], 'accepted_partial')
            after = load_snapshot(base/'result/output.snapshot')
            for k in s.keyframes:
                if k.id not in selected:
                    np.testing.assert_array_equal(k.t_wc, after.keyframes[k.id].t_wc)
            times, poses = read_tum(base/'result/corrected.tum')
            for frame, timestamp, pose in zip(frames, times, poses):
                self.assertEqual(frame.timestamp, timestamp)
                if frame.anchor_id not in selected:
                    np.testing.assert_allclose(pose, frame.t_wc, atol=1e-12, rtol=0)
            previous = (base/'result/corrected.tum').read_bytes()
            repeated = subprocess.run(command, capture_output=True, timeout=30)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(previous, (base/'result/corrected.tum').read_bytes())


if __name__ == '__main__': unittest.main()
