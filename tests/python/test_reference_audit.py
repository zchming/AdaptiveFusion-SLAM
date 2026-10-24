import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from fixtures import scene
from adaptivefusion.local_ba import solve, Problem
from adaptivefusion.offline import full_history
from adaptivefusion.reference_audit import geometry_metrics, audit_geometry, audit_history
from adaptivefusion.state import write_snapshot, write_sparse_ply, Observation
from adaptivefusion.trajectory import FrameRecord, write_tum


class ReferenceAuditTests(unittest.TestCase):
    def bundle(self, accepted=True):
        before, _, _ = scene(); before.config.window_size = len(before.keyframes)
        after = solve(before, method='schur-lm').snapshot if accepted else before.clone()
        p = Problem(before)
        protocol = dict(holdout_modulus=0, active_keyframe_ids=p.kids, component_policy='all',
                        truth_used=False, online=False, active_points=len(p.pids), active_observations=len(p.obs))
        report = dict(accepted=accepted, status='accepted' if accepted else 'rejected',
                      diagnostics={'active_keyframe_ids':p.kids},
                      initial=p.metrics(p.x0) if accepted else {},
                      final=Problem(after).metrics(Problem(after).x0) if accepted else {})
        frames = []
        for i in range(5):
            k = before.keyframes[i//2]
            frames.append(FrameRecord(i, .1+i*.5, i!=3, max(0,i-1) if i!=4 else 2, i//2, k.t_wc.copy()))
        frames[0].reference_id = -1
        history = full_history(frames, before, {k.id:k.t_wc for k in after.keyframes})
        return before, after, protocol, report, frames, history

    def test_direct_matrix_objective_matches_optimizer(self):
        before, after, protocol, report, _, _ = self.bundle()
        result = audit_geometry(before, after, protocol, report)
        self.assertTrue(result['accepted']); self.assertFalse(result['accuracy_certified'])
        for snapshot in [before, after]:
            m = geometry_metrics(snapshot, protocol['active_keyframe_ids'])
            p = Problem(snapshot)
            for key, value in p.metrics(p.x0).items(): self.assertAlmostEqual(m[key], value, places=7)

    def test_rejection_requires_every_point_and_pose_unchanged(self):
        before, after, protocol, report, _, _ = self.bundle(False)
        self.assertTrue(audit_geometry(before, after, protocol, report)['exact_rollback'])
        after.points[100].xyz[0] += .001
        with self.assertRaisesRegex(ValueError, 'point changed'): audit_geometry(before, after, protocol, report)
        after = before.clone(); after.keyframes[1].t_wc[0,3] += .001
        with self.assertRaisesRegex(ValueError, 'pose changed'): audit_geometry(before, after, protocol, report)

    def test_anchor_samples_objective_and_status_tampering_reject(self):
        before, after, protocol, report, _, _ = self.bundle()
        bad = after.clone(); bad.keyframes[0].t_wc[0,3] += .001
        with self.assertRaisesRegex(ValueError, 'anchor'): audit_geometry(before, bad, protocol, report)
        bad_report = copy.deepcopy(report); bad_report['final']['robust_cost'] += 1
        with self.assertRaisesRegex(ValueError, 'objective mismatch'): audit_geometry(before, after, protocol, bad_report)
        bad_report = copy.deepcopy(report); bad_report['status'] = 'running'
        with self.assertRaisesRegex(ValueError, 'completion'): audit_geometry(before, after, protocol, bad_report)
        bad = after.clone(); o = bad.observations[0]
        bad.observations[0] = Observation(o.point_id,o.keyframe_id,o.feature_index,o.u,o.v,o.raw_depth+1)
        with self.assertRaises(ValueError): audit_geometry(before, bad, protocol, report)

    def test_independent_history_matches_and_lost_frames_are_not_filled(self):
        before, after, _, _, frames, history = self.bundle()
        times, poses = zip(*history)
        r = audit_history(before, after, frames, times, poses, accepted=True, active={0,1,2})
        self.assertEqual(r['valid_poses'],4); self.assertEqual(r['lost_frames_preserved'],1)
        self.assertLess(r['maximum_pose_matrix_difference'],1e-12)
        bad = np.array(poses); bad[1,0,3] += .01
        with self.assertRaisesRegex(ValueError, 'inconsistency'): audit_history(before, after, frames, times, bad, accepted=True, active={0,1,2})
        with self.assertRaisesRegex(ValueError, 'coverage'): audit_history(before, after, frames, list(times)+[1.6], list(poses)+[np.eye(4)], accepted=True, active={0,1,2})

    def test_rejected_history_and_wrong_raw_anchor_reject(self):
        before, after, _, _, frames, history = self.bundle(False)
        times, poses = zip(*history)
        audit_history(before, after, frames, times, poses, accepted=False, active={0,1,2})
        bad = np.array(poses); bad[-1,1,3] += .001
        with self.assertRaises(ValueError): audit_history(before, after, frames, times, bad, accepted=False, active={0,1,2})
        frames[2].t_wc[0,3] += .001
        with self.assertRaisesRegex(ValueError, 'raw frame'): audit_history(before, after, frames, times, poses, accepted=False, active={0,1,2})

    def test_cli_checks_map_and_refuses_overwrite(self):
        before, after, protocol, report, frames, history = self.bundle()
        entry = Path(__file__).resolve().parents[2]/'python/audit_offline_reference.py'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cache=root/'cache'; run=root/'run'; cache.mkdir(); run.mkdir()
            header='frame_id timestamp valid reference_id anchor_id '+' '.join(f't{i}' for i in range(16))
            lines=[header]+[f'{f.id} {f.timestamp} {int(f.valid)} {f.reference_id} {f.anchor_id} '+' '.join(map(str,f.t_wc.ravel())) for f in frames]
            (cache/'frames.txt').write_text('\n'.join(lines)+'\n')
            (cache/'complete.json').write_text(json.dumps(dict(risk_enabled=False, online_writeback=False, processed_frames=len(frames))))
            write_snapshot(run/'input.snapshot', before); write_snapshot(run/'output.snapshot',after)
            (run/'protocol.json').write_text(json.dumps(protocol)); (run/'report.json').write_text(json.dumps(report))
            write_tum(run/'corrected.tum',history); write_sparse_ply(run/'shared_map.ply',after)
            cmd=[sys.executable,str(entry),'--cache',str(cache),'--run',str(run),'--output',str(root/'audit')]
            result=subprocess.run(cmd,capture_output=True,text=True); self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            r=json.loads((root/'audit/report.json').read_text()); self.assertFalse(r['ground_truth_used']); self.assertEqual(r['status'],'consistency_pass')
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,2)
            ply=run/'shared_map.ply'; ply.write_text(ply.read_text().replace('units metres','units millimetres'))
            cmd[-1]=str(root/'bad-audit'); result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,1,result.stdout+result.stderr)
            self.assertIn('map schema',json.loads((root/'bad-audit/report.json').read_text())['error'])
