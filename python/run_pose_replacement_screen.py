#!/usr/bin/env python3
"""Offline full/split forward/reverse native PnP screen, without GT or gates."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from adaptivefusion.bridge_candidates import sample_depth
from adaptivefusion.frontend_features import read_features
from adaptivefusion.geometry import inverse, unproject
from adaptivefusion.matching_verification import checked_mask
from adaptivefusion.replacement_signals import fit_partition, closure, common_check
from adaptivefusion.state import file_sha256
from run_camera_pnp_comparison import write_json, arm_process, check_hashes, NATIVE
from run_orb_observation_controls import support

CAMERA = [517.3, 516.5, 318.6, 255.3]


def sealed(study):
    path = study/'fit-only.json'
    if file_sha256(path) != json.loads((study/'fit-seal.json').read_text())['fit-only.json']:
        raise ValueError('sealed study changed: '+str(study))
    return json.loads(path.read_text())


def fit(args):
    if args.output.exists():
        raise ValueError('output exists; refuse overwrite')
    import cv2
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    plan = json.loads(args.protocol.read_text())
    if plan['split'] != 'source_4x4_checkerboard_even_fit_odd_check_before_F_and_depth':
        raise ValueError('unknown split protocol')
    baseline, reference = sealed(args.baseline_study), sealed(args.reference_study)
    if len(baseline['cases']) != plan['cases'] or len(reference['cases']) != plan['cases']:
        raise ValueError('case count mismatch')
    refs = {(c['scene'], c['pair']):c for c in reference['cases']}
    sources = json.loads(args.sources.read_text())
    if any(set(p) != {'frames', 'features', 'dataset'} for p in sources.values()):
        raise ValueError('only inference paths are accepted')
    binary = args.native_binary.resolve()
    if baseline['input_sha256'].get(str(binary)) != file_sha256(binary):
        raise ValueError('native binary differs from frozen baseline')
    paths = [args.protocol, args.sources, binary, args.baseline_study/'fit-only.json',
             args.reference_study/'fit-only.json', Path(__file__),
             Path(__file__).parent/'adaptivefusion/replacement_signals.py']
    hashes = {str(p.resolve()):file_sha256(p) for p in paths}
    args.output.mkdir(); (args.output/'inputs').mkdir(); (args.output/'arms').mkdir()
    write_json(args.output/'execution-protocol.json', dict(plan=plan, truth_read=False,
        opencv=cv2.__version__, parameter_tuning=False, worker_python=str(args.worker_python)))
    cases = []; begin = time.perf_counter()
    for old in baseline['cases']:
        ref = refs[old['scene'], old['pair']]
        if ref['raw_baseline'] != old['raw_baseline']:
            raise ValueError('raw fallback differs')
        previous = next(a for a in old['arms'] if a['camera_profile']=='rootsift_lightglue')
        full = next(a for a in ref['arms'] if a['camera_profile']=='F_MAGSAC')
        files = [args.matching/previous['match_file'],
                 *[args.matching/f['feature_file'] for f in old['features']]]
        for p in files:
            if baseline['input_sha256'].get(str(p.resolve())) != file_sha256(p):
                raise ValueError('matching cache changed')
            hashes[str(p.resolve())] = file_sha256(p)
        with np.load(files[0]) as z:
            matches = z['matches'].copy()
        uv = []
        for side, p in enumerate(files[1:]):
            with np.load(p) as z:
                uv.append(z['keypoints'][matches[:, side]].copy())
        partition = fit_partition(uv[0])
        depths = []
        for kid in old['pair'].split('-'):
            p = Path(sources[old['scene']]['features'])/(kid+'.features')
            m = read_features(p).metadata
            dp = Path(sources[old['scene']]['dataset'])/m['images'][1]
            with Image.open(dp) as image:
                depths.append(np.array(image))
            for item in [p, dp]: hashes[str(item.resolve())] = file_sha256(item)
        zs = [np.asarray([sample_depth(depth, p, 5000.) or np.nan for p in pixels])
              for depth, pixels in zip(depths, uv)]
        mp = args.verification/'pairs'/f"{old['scene']}_{old['pair']}_models.npz"
        if reference['input_sha256'].get(str(mp.resolve())) != file_sha256(mp):
            raise ValueError('full F mask changed')
        hashes[str(mp.resolve())] = file_sha256(mp)
        with np.load(mp) as z:
            if not np.array_equal(z['matches'], matches): raise ValueError('match identity mismatch')
            fullmask = z['F_mask'].copy() if 'F_mask' in z else np.zeros(len(matches), bool)
        # Reproduce the full forward identities before using the cached result.
        expected = matches[fullmask & np.isfinite(zs[0]), 0].tolist()
        if expected != [r['source_feature'] for r in full['rows']]:
            raise ValueError('cached full-forward row identity mismatch')
        splitmask = np.zeros(len(matches), bool); f_error = None
        try:
            if partition.sum()<8: raise ValueError('fewer than8 training2D matches')
            cv2.setRNGSeed(17)
            matrix, mask = cv2.findFundamentalMat(uv[0][partition], uv[1][partition], cv2.USAC_MAGSAC,
                ransacReprojThreshold=3., confidence=.999, maxIters=10000)
            matrix, mask = checked_mask(matrix, mask, int(partition.sum()))
            splitmask[partition] = mask
        except Exception as error:
            f_error = str(error)
        case = {k:old[k] for k in ['scene', 'pair', 'source_timestamp', 'target_timestamp', 'raw_baseline']}
        raw = np.asarray(case['raw_baseline'])
        case.update(split=dict(all_2d=len(matches), fit_2d=int(partition.sum()), check_2d=int((~partition).sum()),
                               split_F_inliers=int(splitmask.sum()), F_failure=f_error), arms=[])
        checkmask = ~partition & np.isfinite(zs[0])
        checkxyz = np.asarray([unproject(CAMERA, p, z) for p,z in zip(uv[0][checkmask],zs[0][checkmask])]).reshape(-1,3)
        case['split']['check_support'] = support(uv[0][checkmask],zs[0][checkmask],[640,480])
        case['split']['fit_source_ids'] = matches[partition,0].tolist()
        case['split']['check_source_ids'] = matches[~partition,0].tolist()
        for name, reverse, mask in [('full_F',False,fullmask), ('full_reverse',True,fullmask),
                                    ('split_F',False,splitmask), ('split_reverse',True,splitmask)]:
            side = int(reverse); chosen = mask & np.isfinite(zs[side])
            xyz = np.asarray([unproject(CAMERA,p,z) for p,z in zip(uv[side][chosen],zs[side][chosen])]).reshape(-1,3)
            initial = inverse(raw) if reverse else raw
            arm = dict(camera_profile=name, estimator=NATIVE, seed=17,
                input_support=support(uv[side][chosen],zs[side][chosen],[640,480]),
                depth_eligible_count=int(np.isfinite(zs[side]).sum()),
                F_2d_count=int(mask.sum()), is_reverse=reverse)
            if name=='full_F':
                for key in ['fit','process_returncode','timed_out','process_wall_s','peak_rss_kib']:
                    arm[key]=full[key]
                arm['reused_fit'] = True
            elif (name.startswith('split') and f_error) or len(xyz)<6 or not mask.any():
                arm.update(process_returncode=None,timed_out=False,process_wall_s=0.,peak_rss_kib=None,
                    fit=dict(applied=False,reason=f_error or 'insufficient native support',t_target_source=initial.tolist()))
            else:
                prefix = f"{case['scene']}_{case['pair']}_{name}"
                inp = args.output/'inputs'/(prefix+'.json'); output = args.output/'arms'/prefix
                write_json(inp,dict(camera=CAMERA,ids=matches[chosen,side].tolist(),source=xyz.tolist(),
                    pixels=uv[1-side][chosen].tolist(),initial=initial.tolist()))
                arm.update(arm_process([str(args.worker_python),'-B',str(Path(__file__).with_name('run_camera_pnp_comparison.py')),
                    'fit-arm','--input',str(inp),'--output',str(output),'--native-binary',str(binary),
                    '--estimator',NATIVE,'--seed','17'],output,
                    args.output/'arms'/(prefix+'.log'),args.output/'arms'/(prefix+'.rss.txt')))
                arm['fit'].setdefault('t_target_source',initial.tolist())
            if reverse:
                arm['reverse_fit'] = arm['fit']
                arm['fit'] = dict(arm['reverse_fit'])
                arm['fit']['t_target_source'] = inverse(np.asarray(arm['reverse_fit']['t_target_source'])).tolist() if arm['fit']['applied'] else raw.tolist()
            if not arm['fit']['applied'] and not np.array_equal(arm['fit']['t_target_source'],raw):
                raise ValueError('exact raw fallback changed')
            arm['check_pixels'] = common_check(raw,np.asarray(arm['fit']['t_target_source']),checkxyz,uv[1][checkmask],CAMERA)
            arm['check_pixels']['used_by_fit'] = name.startswith('full')
            case['arms'].append(arm)
        case['cycles']={}
        for prefix in ['full','split']:
            forward = next(a for a in case['arms'] if a['camera_profile']==prefix+'_F')
            reverse = next(a for a in case['arms'] if a['camera_profile']==prefix+'_reverse')
            case['cycles'][prefix] = closure(forward['fit']['t_target_source'],reverse['reverse_fit']['t_target_source']) if forward['fit']['applied'] and reverse['fit']['applied'] else None
        cases.append(case)
        write_json(args.output/'inputs'/f"{case['scene']}_{case['pair']}_completed.json",case)
        print(case['scene'],case['pair'],{a['camera_profile']:a['fit']['applied'] for a in case['arms']},flush=True)
    check_hashes(hashes)
    write_json(args.output/'fit-only.json',dict(cases=cases,input_sha256=hashes,truth_read=False,
        fit_phase_s=time.perf_counter()-begin,gate_applied=False))
    write_json(args.output/'fit-seal.json',{str(p.relative_to(args.output)):file_sha256(p)
        for p in args.output.rglob('*') if p.is_file()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['protocol','sources','baseline-study','reference-study','verification','matching','native-binary','worker-python','output']:
        p.add_argument('--'+key,type=Path,required=True)
    fit(p.parse_args())


if __name__=='__main__':
    main()
