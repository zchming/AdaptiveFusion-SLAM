import tempfile
import json
import subprocess
import sys
from pathlib import Path
import unittest
import numpy as np
from adaptivefusion.frontend_features import read_features, descriptor_matches


def sidecar(n=4):
    header='''AFS_FEATURES 1
META fixture 7 after_insertion_before_ba
KEYFRAME 2 7 1 1.001 640 480
CAMERA 500 500 320 240
DEPTH 5000 0.1 8
ORB 1000 1.2000000476837158 8 20
MATCH 0.75 64 1
IMAGES "rgb/a.png" "depth/a.png"
FEATURES %d 32
'''%n
    return header+''.join(f'F {i} {100+i} 200 31 10 0.5 2 -1 10000 {i if i%2 else -1} '+bytes((i*37+j*3+7)%256 for j in range(32)).hex()+'\n' for i in range(n))+'END\n'


class FrontendFeatureTests(unittest.TestCase):
    def parse(self,text):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.features';p.write_text(text);return read_features(p)

    def test_lossless_fields_descriptor_rows_and_ownership(self):
        f=self.parse(sidecar());self.assertEqual(f.descriptors.shape,(4,32))
        np.testing.assert_array_equal(f.descriptors[2],[(2*37+j*3+7)%256 for j in range(32)])
        self.assertEqual(f.metadata['source_frame_id'],7);self.assertEqual(f.metadata['images'][0],'rgb/a.png')
        np.testing.assert_array_equal(f.point_ids,[-1,1,-1,3]);np.testing.assert_array_equal(f.raw_depth,[10000]*4)
        np.testing.assert_array_equal(f.keypoints[0],[100,200,31,10,.5,2,-1])

    def test_empty_features_are_explicit_not_fake_descriptors(self):
        f=self.parse(sidecar(0));self.assertEqual(f.descriptors.shape,(0,32))
        self.assertEqual(descriptor_matches(f.descriptors,f.descriptors),[])

    def test_truncated_extra_data_and_row_ids_reject(self):
        for text in [sidecar().replace('END\n',''),sidecar()+'extra\n',sidecar().replace('F 1 ','F 0 ',1),sidecar().replace('FEATURES 4 32','FEATURES 5 32')]:
            with self.subTest(text=text[-30:]),self.assertRaises(ValueError):self.parse(text)

    def test_invalid_native_rows_and_descriptor_payload_reject(self):
        source=sidecar()
        for text in [source.replace(' 10000 ',' 65536 ',1),source.replace(' 31 10 ',' -1 10 ',1),source.replace(' 200 31',' 500 31',1),source.replace('F 0 100 ','F 0 100.00000001 ',1),source.replace('070a','zz0a',1),source.replace(' 2 -1 ',' 9 -1 ',1),source.replace('10000 3','10000 1')]:
            with self.subTest(text=text[-30:]),self.assertRaises(ValueError):self.parse(text)

    def test_metadata_paths_phase_and_camera_reject(self):
        for text in [sidecar().replace('after_insertion_before_ba','after_ba'),sidecar().replace('500 500','0 500'),sidecar().replace('rgb/a.png','../a.png'),sidecar().replace('META fixture 7','META fixture 8'),sidecar().replace('1.001 640','1.1 640')]:
            with self.subTest(text=text[:90]),self.assertRaises(ValueError):self.parse(text)

    def test_numpy_matches_opencv_policy_on_actual_bytes(self):
        import cv2
        rng=np.random.default_rng(8);a=rng.integers(0,256,(100,32),dtype=np.uint8)
        b=a[rng.permutation(len(a))].copy();b[::3,0]^=3
        expected=[];matcher=cv2.BFMatcher(cv2.NORM_HAMMING,False)
        reverse={m.queryIdx:m.trainIdx for m in matcher.match(b,a)}
        for neighbors in matcher.knnMatch(a,b,k=2):
            if len(neighbors)<2:continue
            m,n=neighbors
            if m.distance<=64 and m.distance<.75*n.distance and reverse[m.trainIdx]==m.queryIdx:
                expected.append((m.queryIdx,m.trainIdx,int(m.distance)))
        self.assertEqual(descriptor_matches(a,b),expected)

    def test_ratio_ties_and_absolute_distance_are_rejected(self):
        q=np.zeros((1,32),np.uint8);t=np.zeros((2,32),np.uint8)
        self.assertEqual(descriptor_matches(q,t),[])
        t[1]=255;t[0,:9]=255
        self.assertEqual(descriptor_matches(q,t),[])
        t[0]=q[0];self.assertEqual(descriptor_matches(q,t),[(0,0,0)])

    def test_mutual_match_prevents_two_sources_owning_one_target(self):
        q=np.zeros((2,32),np.uint8);q[1,0]=1;t=np.zeros((2,32),np.uint8);t[1]=255
        self.assertEqual(descriptor_matches(q,t),[(0,0,0)])
        self.assertEqual(len(descriptor_matches(q,t,mutual=False)),2)

    def test_invalid_descriptor_shape_dtype_policy_and_single_target(self):
        a=np.zeros((2,32),np.uint8)
        for value in [a.astype(float),np.zeros((2,31),np.uint8)]:
            with self.assertRaises(ValueError):descriptor_matches(value,a)
        with self.assertRaises(ValueError):descriptor_matches(a,a,ratio=1)
        self.assertEqual(descriptor_matches(a,a[:1]),[])

    def test_cli_full_feature_geometry_controls_and_overwrite(self):
        import cv2
        from adaptivefusion.state import write_snapshot,file_sha256
        from test_landmark_identity import duplicate_scene
        s,images,d=duplicate_scene();entry=Path(__file__).resolve().parents[2]/'python/audit_frontend_features.py'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);features=root/'features';features.mkdir()
            for k,image in zip(s.keyframes,images):
                cv2.imwrite(str(root/f'rgb{k.id}.png'),image);cv2.imwrite(str(root/f'depth{k.id}.png'),d)
                obs=sorted([o for o in s.observations if o.keyframe_id==k.id],key=lambda o:o.feature_index)
                lines=sidecar(len(obs)).splitlines();lines[1]=f'META fixture {k.id} after_insertion_before_ba'
                lines[2]=f'KEYFRAME {k.id} {k.id} {k.timestamp} {k.timestamp} {k.width} {k.height}'
                lines[3]='CAMERA '+' '.join(map(str,s.camera));lines[7]=f'IMAGES "rgb{k.id}.png" "depth{k.id}.png"'
                for i,o in enumerate(obs):
                    r=lines[i+9].split();r[2:4]=[str(o.u),str(o.v)];r[10]=str(o.point_id);lines[i+9]=' '.join(r)
                (features/f'{k.id}.features').write_text('\n'.join(lines)+'\n')
            (features/'complete.json').write_text(json.dumps({'schema_version':1,'keyframes':4,'capture_phase':'after_insertion_before_ba'}))
            write_snapshot(root/'geometry.snapshot',s);before=file_sha256(root/'geometry.snapshot')
            cmd=[sys.executable,str(entry),'--geometry',str(root/'geometry.snapshot'),'--features',str(features),'--dataset',str(root),'--output',str(root/'out')]
            p=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
            report=json.loads((root/'out/report.json').read_text());self.assertEqual(report['accepted_pairs']['descriptor'],6)
            self.assertGreater(len(report['windows']['descriptor'][0]['qualified_crosscut_pairs']),0)
            self.assertEqual(file_sha256(root/'geometry.snapshot'),before)
            again=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(again.returncode,2)
            self.assertIn('refuse overwrite',again.stderr)

    def test_bounded_blocks_preserve_global_reverse_index(self):
        rng=np.random.default_rng(21);a=rng.integers(0,256,(150,32),dtype=np.uint8);b=a[::-1].copy()
        self.assertEqual(descriptor_matches(a,b),[(i,149-i,0) for i in range(150)])
