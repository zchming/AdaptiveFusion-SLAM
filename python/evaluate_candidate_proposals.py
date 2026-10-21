#!/usr/bin/env python3
"""Independent evaluation only: fixed V9 models, unused depth pixels, then truth.

Never writes proposals, map state, trajectories or optimization admission masks.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from adaptivefusion.candidate_validation import CONFIG,depth_probe,relative_error,truth_endpoints,pair_value
from adaptivefusion.frontend_features import read_features
from adaptivefusion.trajectory import load_frames
from adaptivefusion.geometry import check_pose,inverse
from adaptivefusion.state import file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['matching','features','frames','dataset','truth','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists; refuse overwrite')
    frames=load_frames(args.frames);cases=[];input_hashes={'frames':file_sha256(args.frames)}
    paths=sorted((args.matching/'descriptor').glob('*.json'))
    if not paths:raise ValueError('no frozen descriptor reports')
    args.output.mkdir(parents=True)
    for path in paths:
        proposal=json.loads(path.read_text())
        if not proposal['verification']['accepted']:continue
        a,b=proposal['source_keyframe'],proposal['target_keyframe'];data=[]
        input_hashes['proposal/'+path.name]=file_sha256(path)
        for kid in [a,b]:
            feature_path=args.features/f'{kid}.features';f=read_features(feature_path);m=f.metadata
            if m['keyframe_id']!=kid or m['source_frame_id']>=len(frames) or frames[m['source_frame_id']].timestamp!=m['rgb_timestamp']:
                raise ValueError('feature/frame identity mismatch')
            input_hashes['feature/'+feature_path.name]=file_sha256(feature_path)
            depth_path=args.dataset/m['images'][1]
            if not depth_path.resolve().is_relative_to(args.dataset.resolve()):raise ValueError('depth path escapes dataset')
            depth=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
            if depth is None or depth.dtype!=np.uint16 or depth.shape!=(m['height'],m['width']):raise ValueError('native depth mismatch')
            uv=np.floor(f.keypoints[:,:2]+.5).astype(int);good=(uv[:,0]<depth.shape[1])&(uv[:,1]<depth.shape[0]);samples=np.zeros(len(uv),np.uint16);samples[good]=depth[uv[good,1],uv[good,0]]
            if not np.array_equal(samples,f.raw_depth):raise ValueError('native feature depth provenance mismatch')
            input_hashes['depth/'+m['images'][1]]=file_sha256(depth_path);data.append((f,depth))
        (fa,da),(fb,db)=data
        if any(fa.metadata[k]!=fb.metadata[k] for k in ['source_revision','camera','depth','orb','matching']):raise ValueError('incompatible feature calibration/source')
        candidate=check_pose(np.array(proposal['verification']['t_target_source']))
        baseline=inverse(frames[fb.metadata['source_frame_id']].t_wc)@frames[fa.metadata['source_frame_id']].t_wc
        depth={}
        for name,transform in [('candidate',candidate),('raw_baseline',baseline)]:
            depth[name]={'forward':depth_probe(da,db,fa.keypoints[:,:2],fb.keypoints[:,:2],fa.metadata['camera'],fa.metadata['depth'],transform),
                         'reverse':depth_probe(db,da,fb.keypoints[:,:2],fa.keypoints[:,:2],fa.metadata['camera'],fa.metadata['depth'],inverse(transform))}
        cases.append({'source_keyframe':a,'target_keyframe':b,'crossed_cuts':proposal['crossed_cuts'],
                      'source_timestamp':fa.metadata['rgb_timestamp'],'target_timestamp':fb.metadata['rgb_timestamp'],
                      'source_depth_timestamp':fa.metadata['depth_timestamp'],'target_depth_timestamp':fb.metadata['depth_timestamp'],
                      'candidate_t_target_source':candidate.tolist(),'raw_baseline_t_target_source':baseline.tolist(),
                      'depth_checks':depth})
    # This immutable evidence file is finalized before opening ground truth.
    pretruth={'config':CONFIG,'input_sha256':input_hashes,'cases':cases,'truth_read':False,
              'scope':'fixed-model depth probes only; excluded all native ORB depth neighborhoods; no fitting or visibility-residual filtering'}
    depth_output=args.output/'depth-only.json';depth_output.write_text(json.dumps(pretruth,indent=2)+'\n')
    depth_sha=file_sha256(depth_output)
    from adaptivefusion.evaluation import read_tum
    times,poses=read_tum(args.truth)
    evaluated=[]
    for case in cases:
        gt=truth_endpoints(times,poses,case['source_timestamp'],case['target_timestamp']);modes={}
        for mode in ['nearest','interpolated']:
            if gt[mode] is None:continue
            candidate=relative_error(np.array(case['candidate_t_target_source']),*gt[mode])
            baseline=relative_error(np.array(case['raw_baseline_t_target_source']),*gt[mode])
            modes[mode]={'candidate':candidate,'raw_baseline':baseline,'value':pair_value(candidate,baseline)}
        complete=len(modes)==2;sensitive=None
        if complete:
            x,y=[modes[k]['candidate'] for k in ['nearest','interpolated']]
            sensitive=(x['gt_fidelity_pass']!=y['gt_fidelity_pass'] or abs(x['translation_m']-y['translation_m'])>CONFIG['meaningful_translation_m'] or abs(x['rotation_deg']-y['rotation_deg'])>CONFIG['meaningful_rotation_deg'])
        depth_pass=all(v['depth_consistency_pass'] for v in case['depth_checks']['candidate'].values())
        passes=bool(complete and not sensitive and depth_pass and all(v['candidate']['gt_fidelity_pass'] for v in modes.values()))
        evaluated.append({**case,'truth_errors':modes,'truth_association_differences_s':gt['association_differences_s'],
                          'truth_bracket_spans_s':gt['bracket_spans_s'],'evaluation_status':'complete' if complete else 'not_fully_evaluable',
                          'timestamp_sensitive':sensitive,'independent_fidelity_checks_pass':passes,
                          'pair_level_value_supported':bool(passes and all(v['value']['pair_level_value_supported'] for v in modes.values()))})
    assert file_sha256(depth_output)==depth_sha
    report={'config':CONFIG,'cases':evaluated,'depth_only_sha256':depth_sha,'truth_sha256':file_sha256(args.truth),
            'scope':'evaluation-only classifications; never measurement selection or BA admission; original proposals unchanged',
            'new_pose_fit':False,'trajectory_accuracy_evaluated':False,'online':False}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(evaluated),'fidelity_checks_pass':sum(v['independent_fidelity_checks_pass'] for v in evaluated),
                      'pair_level_value_supported':sum(v['pair_level_value_supported'] for v in evaluated)}))
    return 0


if __name__=='__main__':raise SystemExit(main())
