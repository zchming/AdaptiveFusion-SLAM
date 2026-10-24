#!/usr/bin/env python3
"""Descriptor vs LK geometric proposals on identical original C++ feature rows."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import time
import cv2
import numpy as np
from audit_bridge_candidates import select_cuts
from adaptivefusion.frontend_features import read_features, descriptor_matches
from adaptivefusion.landmark_identity import _flow, reciprocal_matches, audit_cycles
from adaptivefusion.bridge_candidates import sample_depth, verify_rigid
from adaptivefusion.geometry import unproject
from adaptivefusion.state import load_snapshot,file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['geometry','features','dataset','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists; refuse overwrite')
    start=time.perf_counter();s=load_snapshot(args.geometry)
    marker=json.loads((args.features/'complete.json').read_text())
    if marker!={'schema_version':1,'keyframes':len(s.keyframes),'capture_phase':'after_insertion_before_ba'}:
        raise ValueError('incomplete/invalid feature export')
    expected={f'{k.id}.features' for k in s.keyframes}
    if {p.name for p in args.features.glob('*.features')}!=expected or list(args.features.glob('*.tmp')):
        raise ValueError('feature coverage or partial export mismatch')
    cuts=select_cuts(s);windows={c:list(range(max(0,c-2),min(len(s.keyframes),c+2))) for c,_ in cuts}
    pairs=sorted({p for kids in windows.values() for p in combinations(kids,2)})
    needed=sorted({k for p in pairs for k in p});features={};images={};evidence={}
    for kid in needed:
        path=args.features/f'{kid}.features';f=read_features(path);m=f.metadata;k=s.keyframes[kid]
        if (m['keyframe_id'],m['source_frame_id'],m['rgb_timestamp'],m['width'],m['height'])!=(kid,k.source_frame_id,k.timestamp,k.width,k.height):
            raise ValueError('feature/geometry frame alignment failed')
        if not np.array_equal(m['camera'],s.camera) or m['depth'][0]!=s.config.depth_scale:
            raise ValueError('feature/geometry calibration failed')
        image_pair=[]
        for i,relative in enumerate(m['images']):
            image_path=args.dataset/relative
            if not image_path.resolve().is_relative_to(args.dataset.resolve()):raise ValueError('image escapes dataset')
            image=cv2.imread(str(image_path),cv2.IMREAD_GRAYSCALE if i==0 else cv2.IMREAD_UNCHANGED)
            if image is None or image.shape!=(k.height,k.width) or image.dtype!=(np.uint8 if i==0 else np.uint16):
                raise ValueError('native image mismatch')
            evidence[relative]=file_sha256(image_path);image_pair.append(image)
        for i,(u,v) in enumerate(f.keypoints[:,:2]):
            x,y=np.floor([u+.5,v+.5]).astype(int)
            raw=int(image_pair[1][y,x]) if 0<=x<k.width and 0<=y<k.height else 0
            if raw!=int(f.raw_depth[i]):raise ValueError('native feature depth provenance mismatch')
        for o in s.observations:
            if o.keyframe_id!=kid:continue
            if o.feature_index>=len(f.keypoints) or int(f.point_ids[o.feature_index])!=o.point_id or list(f.keypoints[o.feature_index,:2])!=[o.u,o.v] or int(f.raw_depth[o.feature_index])!=o.raw_depth:
                raise ValueError('feature/geometry observation identity mismatch')
        features[kid]=f;images[kid]=image_pair
    args.output.mkdir(parents=True);reports={arm:{} for arm in ['descriptor','lk']}
    for a,b in pairs:
        fa,fb=features[a],features[b]
        if any(fa.metadata[k]!=fb.metadata[k] for k in ['source_revision','camera','depth','orb','matching']):
            raise ValueError('incompatible frontend settings')
        ratio,distance,mutual=fa.metadata['matching']
        desc=descriptor_matches(fa.descriptors,fb.descriptors,ratio,distance,mutual)
        f,ok_a=_flow(images[a][0],images[b][0],fa.keypoints[:,:2]);g,ok_b=_flow(images[b][0],images[a][0],fb.keypoints[:,:2])
        lk=[(i,j,None) for i,j in reciprocal_matches(fa.keypoints[:,:2],fb.keypoints[:,:2],f,g,ok_a,ok_b)]
        for arm,matches in [('descriptor',desc),('lk',lk)]:
            xyz_a=[];xyz_b=[];ids=[];records=[]
            for i,j,hamming in matches:
                uv_a,uv_b=fa.keypoints[i,:2],fb.keypoints[j,:2]
                z_a=sample_depth(images[a][1],uv_a,s.config.depth_scale);z_b=sample_depth(images[b][1],uv_b,s.config.depth_scale)
                if z_a is None or z_b is None:continue
                ids.append(i);xyz_a.append(unproject(s.camera,uv_a,z_a));xyz_b.append(unproject(s.camera,uv_b,z_b))
                records.append({'source_feature':i,'target_feature':j,'source_point_id':int(fa.point_ids[i]),'target_point_id':int(fb.point_ids[j]),'hamming':hamming})
            fit=verify_rigid(xyz_a,xyz_b,ids)
            row={'source_keyframe':a,'target_keyframe':b,'source_features':len(fa.keypoints),'target_features':len(fb.keypoints),
                 'matches':len(matches),'depth_valid_matches':len(ids),'crossed_cuts':[c for c,_ in cuts if a<c<=b],
                 'verification':fit,'records':records}
            reports[arm][a,b]=row
            directory=args.output/arm;directory.mkdir(exist_ok=True)
            (directory/f'{a}-{b}.json').write_text(json.dumps(row,indent=2)+'\n')
    summaries={}
    for arm,rows in reports.items():
        summaries[arm]=[]
        for cut,_ in cuts:
            cycle=audit_cycles([rows[p] for p in combinations(windows[cut],2)],windows[cut])
            summaries[arm].append({'cut':cut,'cycles':cycle,'qualified_crosscut_pairs':[p for p in cycle['qualified_pairs'] if p[0]<cut<=p[1]]})
    report={'unique_pairs':len(pairs),'accepted_pairs':{arm:sum(p['verification']['accepted'] for p in rows.values()) for arm,rows in reports.items()},
            'windows':summaries,'images_sha256':evidence,'geometry_sha256':file_sha256(args.geometry),
            'feature_sha256':{str(k):file_sha256(args.features/f'{k}.features') for k in needed},
            'scope':'new offline matches; not original frontend decisions, identity merges, BA or trajectory accuracy',
            'source_sha256':{str(p.relative_to(Path(__file__).parent)):file_sha256(p) for p in [Path(__file__),*sorted((Path(__file__).parent/'adaptivefusion').glob('*.py'))]},
            'wall_s':time.perf_counter()-start}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ['unique_pairs','accepted_pairs','wall_s','scope']}))
    return 0


if __name__=='__main__':raise SystemExit(main())
