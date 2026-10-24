#!/usr/bin/env python3
"""Freeze image-disjoint within-sequence pairs without GT or inference outputs."""
import argparse
import json
from pathlib import Path
import shlex
import numpy as np
from adaptivefusion.geometry import inverse
from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames
from run_camera_pnp_comparison import write_json, check_hashes


def prepare(args):
    if args.output.exists(): raise ValueError('output exists; refuse overwrite')
    sources = json.loads(args.sources.read_text())
    if any(set(p) != {'frames','features','dataset'} for p in sources.values()):
        raise ValueError('inference sources must contain only frames/features/dataset')
    prior_path = args.development_study/'fit-only.json'
    old = json.loads(prior_path.read_text())
    old_seal = json.loads((args.development_study/'fit-seal.json').read_text())
    if file_sha256(prior_path) != old_seal['fit-only.json']: raise ValueError('development inventory changed')
    hashes = {str(p.resolve()):file_sha256(p) for p in [prior_path,args.sources,args.native_binary,Path(__file__)]}
    cases = []; inventories = []; used_images = set()
    for scene,paths in sources.items():
        frames = load_frames(paths['frames'])
        hashes[str(Path(paths['frames']).resolve())] = file_sha256(Path(paths['frames']))
        metadata = {}
        for path in Path(paths['features']).glob('*.features'):
            with path.open() as f:
                if next(f).strip() != 'AFS_FEATURES 1': raise ValueError('feature schema changed')
                next(f); row=shlex.split(next(f))
            kid,fid=int(row[1]),int(row[2]);stamp=float(row[3])
            if row[0]!='KEYFRAME' or path.stem!=str(kid) or frames[fid].timestamp!=stamp:
                raise ValueError('keyframe/trajectory timestamp mismatch')
            metadata[kid]=dict(timestamp=stamp,frame_id=fid,path=path)
        oldids={int(i) for c in old['cases'] if c['scene']==scene for i in c['pair'].split('-')}
        oldtimes=[metadata[i]['timestamp'] for i in oldids]
        candidates=[];occupied=set()
        for a in sorted(metadata):
            b=a+1
            if b not in metadata or a in occupied or b in occupied:continue
            if not frames[metadata[a]['frame_id']].valid or not frames[metadata[b]['frame_id']].valid:continue
            if any(abs(metadata[k]['timestamp']-t)<args.exclusion_s for k in [a,b] for t in oldtimes):continue
            candidates.append([a,b]);occupied.update([a,b])
        if len(candidates)<args.pairs_per_scene:raise ValueError('insufficient predeclared pairs: '+scene)
        indices=np.linspace(0,len(candidates)-1,args.pairs_per_scene,dtype=int).tolist()
        chosen=[candidates[i] for i in indices]
        inventories.append(dict(scene=scene,available_disjoint_pairs=len(candidates),candidate_pairs=candidates,
            development_keyframes=sorted(oldids),selected_indices=indices,selected_pairs=chosen))
        for a,b in chosen:
            fa,fb=[frames[metadata[k]['frame_id']] for k in [a,b]]
            for kid in [a,b]:
                identity=(scene,kid)
                if kid in oldids or identity in used_images:raise ValueError('image overlap')
                used_images.add(identity);p=metadata[kid]['path'];hashes[str(p.resolve())]=file_sha256(p)
            cases.append(dict(scene=scene,pair=f'{a}-{b}',source_timestamp=fa.timestamp,target_timestamp=fb.timestamp,
                raw_baseline=(inverse(fb.t_wc)@fa.t_wc).tolist()))
    check_hashes(hashes)
    args.output.mkdir()
    write_json(args.output/'selection-protocol.json',dict(status='frozen_before_inference_or_GT',
        scope='within-sequence image-disjoint holdout, not unseen-sequence generalization',
        pairs_per_scene=args.pairs_per_scene,cases=len(cases),images=len(used_images),
        development_exclusion_s=args.exclusion_s,
        selection='consecutive keyframe IDs; valid raw endpoints; greedily vertex-disjoint; temporal buffer from every development image; evenly spaced candidate indices',
        inventories=inventories,input_sha256=hashes,truth_read=False))
    # Compatible case-manifest container; no fitted results are claimed here.
    write_json(args.output/'fit-only.json',dict(stage='pair_manifest_only_no_fitting',cases=cases,input_sha256=hashes,truth_read=False))
    write_json(args.output/'fit-seal.json',{'fit-only.json':file_sha256(args.output/'fit-only.json'),
        'selection-protocol.json':file_sha256(args.output/'selection-protocol.json')})
    print(json.dumps(dict(cases=len(cases),images=len(used_images),truth_read=False)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['sources','development-study','native-binary','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--pairs-per-scene',type=int,default=12)
    p.add_argument('--exclusion-s',type=float,default=1.)
    args=p.parse_args()
    if args.pairs_per_scene<1 or args.exclusion_s<0:raise ValueError('invalid declared selection budget')
    prepare(args)


if __name__=='__main__':main()
