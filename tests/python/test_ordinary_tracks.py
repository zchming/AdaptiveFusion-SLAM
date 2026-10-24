from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from adaptivefusion.landmark_identity import _flow, audit_pair
from adaptivefusion.ordinary_tracks import track_chain, chain_endpoints
from test_landmark_identity import duplicate_scene


class OrdinaryTrackTests(unittest.TestCase):
    def test_one_step_matches_existing_direct_flow_exactly(self):
        s,images,d=duplicate_scene();obs=sorted([o for o in s.observations if o.keyframe_id==0],key=lambda o:o.point_id)
        uv=np.array([[o.u,o.v] for o in obs],np.float32)
        expected,valid=_flow(images[0],images[1],uv);actual,mask,history=track_chain(images[:2],uv)
        np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(mask,valid)
        self.assertEqual(history,[int(valid.sum())])

    def test_multiple_steps_metric_direction_and_seed_immutability(self):
        s,images,d=duplicate_scene();uv=np.array([[40.,40.],[80.,80.]],np.float32);before=uv.copy()
        moved,mask,history=track_chain(images,uv)
        self.assertTrue(mask.any());np.testing.assert_allclose(moved[mask],(uv+[6,0])[mask],atol=.02)
        np.testing.assert_array_equal(uv,before);self.assertTrue(all(a>=b for a,b in zip([len(uv)]+history,history)))

    def test_lost_track_never_reactivates_or_reaches_later_flow(self):
        im=np.zeros((20,20),np.uint8)
        with patch('adaptivefusion.ordinary_tracks._flow',side_effect=[(np.array([[3.,3.],[4.,4.]]),np.array([False,True])),(np.array([[5.,5.]]),np.array([True]))]) as mock:
            _,mask,history=track_chain([im,im,im],[[1,1],[2,2]])
            np.testing.assert_array_equal(mask,[False,True]);self.assertEqual(history,[1,1])
            self.assertEqual(len(mock.call_args_list[1].args[2]),1)

    def test_all_lost_skips_further_calls(self):
        im=np.zeros((20,20),np.uint8)
        with patch('adaptivefusion.ordinary_tracks._flow',return_value=(np.array([[3.,3.]]),np.array([False]))) as mock:
            _,mask,history=track_chain([im]*4,[[1,1]])
            self.assertFalse(mask.any());self.assertEqual(history,[0,0,0]);self.assertEqual(mock.call_count,1)

    def test_empty_tracks_and_invalid_images_or_seeds(self):
        im=np.zeros((20,20),np.uint8)
        p,m,h=track_chain([im]*3,[]);self.assertEqual(p.shape,(0,2));self.assertEqual(h,[0,0])
        for images,uv in [([im],[]),([im,im.astype(float)],[]),([im,im],[[np.nan,0]])]:
            with self.assertRaises(ValueError):track_chain(images,uv)

    def test_missing_ordinary_frame_is_refused(self):
        s,images,d=duplicate_scene()
        with self.assertRaisesRegex(ValueError,'skipped'):chain_endpoints(s,0,2,[images[0],images[2]])
        with self.assertRaises(ValueError):chain_endpoints(s,2,0,images[:3])

    def test_endpoint_injection_preserves_default_geometry_exactly(self):
        s,images,d=duplicate_scene();source=[]
        for kid in [0,2]:source.append(np.array([[o.u,o.v] for o in sorted([o for o in s.observations if o.keyframe_id==kid],key=lambda o:o.point_id)]))
        f,ok_a=_flow(images[0],images[2],source[0]);b,ok_b=_flow(images[2],images[0],source[1])
        default=audit_pair(s,0,2,images[0],images[2],d,d)
        injected=audit_pair(s,0,2,images[0],images[2],d,d,endpoint_flows=(f,ok_a,b,ok_b))
        self.assertEqual(default,injected)

    def test_chain_reaches_existing_identity_checks_without_state_change(self):
        s,images,d=duplicate_scene();before=[k.t_wc.copy() for k in s.keyframes]
        flows,transport=chain_endpoints(s,0,2,images[:3])
        p=audit_pair(s,0,2,images[0],images[2],d,d,endpoint_flows=flows)
        self.assertTrue(p['verification']['accepted']);self.assertGreater(p['different_id_matches'],50)
        self.assertEqual(transport['ordinary_intermediate_images'],1)
        for a,b in zip(s.keyframes,before):np.testing.assert_array_equal(a.t_wc,b)

    def test_complete_cli_with_synthetic_dataset_and_frame_stream(self):
        import cv2
        from adaptivefusion.state import write_snapshot, file_sha256
        s,images,d=duplicate_scene()
        entry=Path(__file__).resolve().parents[2]/'python/audit_ordinary_tracks.py'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rgb=[];depth=[]
            for i,image in enumerate(images):
                cv2.imwrite(str(root/f'rgb{i}.png'),image);cv2.imwrite(str(root/f'depth{i}.png'),d)
                rgb.append(f'{i+1}.0 rgb{i}.png');depth.append(f'{i+1}.0 depth{i}.png')
            (root/'rgb.txt').write_text('\n'.join(rgb)+'\n');(root/'depth.txt').write_text('\n'.join(depth)+'\n')
            write_snapshot(root/'input.snapshot',s)
            header='frame_id timestamp valid reference_id anchor_id '+' '.join(f't{i}' for i in range(16))
            lines=[header]+[f'{k.id} {k.timestamp} 1 {k.id-1} {k.id} '+' '.join(map(str,k.t_wc.ravel())) for k in s.keyframes]
            (root/'frames.txt').write_text('\n'.join(lines)+'\n')
            before=file_sha256(root/'input.snapshot')
            p=subprocess.run([sys.executable,str(entry),'--snapshot',str(root/'input.snapshot'),'--frames',str(root/'frames.txt'),'--dataset',str(root),'--output',str(root/'out')],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            j=json.loads((root/'out/report.json').read_text())
            self.assertEqual(j['unique_pairs'],6)
            self.assertTrue(all(n>0 for n in j['accepted_rigid_pairs'].values()))
            self.assertTrue(j['snapshot_unchanged']);self.assertFalse(j['map_updated'])
            self.assertEqual(file_sha256(root/'input.snapshot'),before)
            self.assertGreater(sum(w['identity']['candidate_edges'] for w in j['windows']['chain']),50)

    def test_entry_refuses_existing_output(self):
        entry=Path(__file__).resolve().parents[2]/'python/audit_ordinary_tracks.py'
        with tempfile.TemporaryDirectory() as tmp:
            p=subprocess.run([sys.executable,str(entry),'--snapshot','missing','--frames','missing','--dataset','missing','--output',tmp],capture_output=True,text=True)
            self.assertEqual(p.returncode,2);self.assertIn('refuse overwrite',p.stderr)
