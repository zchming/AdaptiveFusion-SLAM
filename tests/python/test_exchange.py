"""Optional integration checks with locally built C++ executables.

No real dataset. Specify AFS_CERES_REPLAY / AFS_ODOMETRY to enable checks.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from fixtures import scene
from adaptivefusion.state import write_snapshot, load_snapshot
from adaptivefusion.local_ba import Problem, solve

ROOT=Path(__file__).resolve().parents[2]


class ExchangeTests(unittest.TestCase):
    def test_batch_cli_uses_cache_and_preserves_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            fixture=subprocess.run([sys.executable,str(ROOT/'tests/python/make_fixture.py'),
                                    '--output',str(base/'cache')],capture_output=True,text=True)
            self.assertEqual(fixture.returncode,0,fixture.stderr)
            for cameras,frame in [(2,2),(3,4)]:
                snapshot,_,_=scene(cameras=cameras);snapshot.state_version=frame
                write_snapshot(base/f'cache/{frame}.before.snapshot',snapshot)
            final,_,_=scene();final.state_version=6
            write_snapshot(base/'cache/final.snapshot',final)
            (base/'cache/complete.json').write_text(json.dumps({'processed_frames':6,'risk_enabled':False,'online_writeback':False}))
            command=[sys.executable,str(ROOT/'python/run_batch_reference.py'),'--cache',str(base/'cache'),
                     '--output',str(base/'batch')]
            run=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            report=json.loads((base/'batch/report.json').read_text())
            self.assertTrue(report['accepted'])
            self.assertLess(report['final']['robust_cost'],report['initial']['robust_cost'])
            self.assertEqual(len((base/'batch/corrected.tum').read_text().splitlines()),6)
            self.assertIn('element vertex 8',(base/'batch/shared_map.ply').read_text())
            original=(base/'batch/corrected.tum').read_bytes()
            retry=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(retry.returncode,0)
            self.assertEqual(original,(base/'batch/corrected.tum').read_bytes())

    def test_rejected_cli_preserves_original_history(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            fixture=subprocess.run([sys.executable,str(ROOT/'tests/python/make_fixture.py'),
                                    '--output',str(base/'fixture')],capture_output=True,text=True)
            self.assertEqual(fixture.returncode,0,fixture.stderr)
            run=subprocess.run([sys.executable,str(ROOT/'python/run_geometry_replay.py'),
                '--snapshot',str(base/'fixture/input.snapshot'),
                '--frames',str(base/'fixture/frames.txt'),'--output',str(base/'rejected'),
                '--max-nfev','1'],capture_output=True,text=True,timeout=30)
            self.assertEqual(run.returncode,2,run.stdout+run.stderr)
            report=json.loads((base/'rejected/report.json').read_text())
            self.assertFalse(report['python']['accepted'])
            self.assertEqual((base/'rejected/original.tum').read_bytes(),
                             (base/'rejected/python_corrected.tum').read_bytes())
            before=load_snapshot(base/'fixture/input.snapshot')
            after=load_snapshot(base/'rejected/python.snapshot')
            for a,b in zip(before.keyframes,after.keyframes):np.testing.assert_array_equal(a.t_wc,b.t_wc)

    @unittest.skipUnless(os.environ.get('AFS_CERES_REPLAY'), 'AFS_CERES_REPLAY not supplied')
    def test_shared_snapshot_ceres_and_scipy(self):
        from run_geometry_replay import candidate_metrics
        s,_,_=scene()
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);write_snapshot(base/'input.snapshot',s)
            process=subprocess.run([os.environ['AFS_CERES_REPLAY'],str(base/'input.snapshot'),str(base/'ceres')],
                                   capture_output=True,text=True,timeout=30)
            self.assertEqual(process.returncode,0,process.stderr)
            ceres=load_snapshot(base/'ceres'/'optimized.snapshot')
            scipy=solve(s,max_nfev=800)
            self.assertTrue(scipy.accepted,scipy.reason)
            metrics=candidate_metrics(Problem(s),ceres)
            self.assertLess(abs(metrics['robust_cost']-scipy.final['robust_cost']),1e-6)
            for k in range(3):np.testing.assert_allclose(ceres.keyframes[k].t_wc,scipy.snapshot.keyframes[k].t_wc,atol=2e-5)
            self.assertEqual(set(ceres.observations),set(s.observations))

    @unittest.skipUnless(os.environ.get('AFS_ODOMETRY'), 'AFS_ODOMETRY not supplied')
    def test_export_does_not_change_tracking_and_replays(self):
        import cv2
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);data=base/'data';data.mkdir()
            image=np.random.default_rng(11).integers(0,256,(480,640,3),dtype=np.uint8)
            image=cv2.GaussianBlur(image,(3,3),.6)
            rgb,depth=[],[]
            for i in range(18):
                ts=f'{1+i*.04:.9f}'
                warped=cv2.warpAffine(image,np.float32([[1,0,i*8],[0,1,0]]),(640,480))
                cv2.imwrite(str(data/f'rgb{i}.png'),warped)
                cv2.imwrite(str(data/f'depth{i}.png'),np.full((480,640),10000,np.uint16))
                rgb.append(f'{ts} rgb{i}.png\n');depth.append(f'{ts} depth{i}.png\n')
            (data/'rgb.txt').write_text(''.join(rgb));(data/'depth.txt').write_text(''.join(depth))
            executable=os.environ['AFS_ODOMETRY']
            plain=subprocess.run([executable,str(data),str(base/'plain.tum')],capture_output=True,text=True,timeout=60)
            self.assertEqual(plain.returncode,0,plain.stderr)
            export=subprocess.run([executable,str(data),str(base/'export.tum'),'18','-','none',str(base/'cache')],
                                  capture_output=True,text=True,timeout=60)
            self.assertEqual(export.returncode,0,export.stderr)
            self.assertEqual((base/'plain.tum').read_bytes(),(base/'export.tum').read_bytes())
            self.assertEqual((base/'plain.tum.health.csv').read_bytes(),(base/'export.tum.health.csv').read_bytes())
            self.assertTrue((base/'cache'/'complete.json').is_file())
            snapshots=sorted((base/'cache').glob('*.before.snapshot'))
            self.assertTrue(snapshots,'synthetic movement must exercise actual BA export')
            from adaptivefusion.trajectory import load_frames
            frames=load_frames(base/'cache'/'frames.txt');self.assertEqual(len(frames),18)
            for path in snapshots:
                s=load_snapshot(path)
                post=load_snapshot(path.with_name(path.name.replace('.before.','.ceres.')))
                from run_geometry_replay import candidate_metrics
                self.assertLessEqual(candidate_metrics(Problem(s),post)['robust_cost'],Problem(s).metrics(Problem(s).x0)['robust_cost']+1e-6)
            result=subprocess.run([sys.executable,str(ROOT/'python'/'run_geometry_replay.py'),
                '--snapshot',str(snapshots[-1]),'--frames',str(base/'cache'/'frames.txt'),
                '--ceres-snapshot',str(snapshots[-1]).replace('.before.','.ceres.'),
                '--output',str(base/'analysis'),'--max-nfev','800'],capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(len((base/'analysis'/'original.tum').read_text().splitlines()),
                             len((base/'analysis'/'python_corrected.tum').read_text().splitlines()))
            previous=(base/'export.tum').read_bytes()
            repeat=subprocess.run([executable,str(data),str(base/'export.tum'),'18','-','none',str(base/'cache')],capture_output=True)
            self.assertNotEqual(repeat.returncode,0)
            self.assertEqual(previous,(base/'export.tum').read_bytes())
