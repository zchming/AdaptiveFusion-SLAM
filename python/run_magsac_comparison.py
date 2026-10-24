#!/usr/bin/env python3
"""V19 cached LightGlue -> F-MAGSAC vs equal-count controls; fixed native PnP."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
from adaptivefusion.geometry import unproject
from adaptivefusion.matching_verification import checked_mask, filtered_rows, equal_count_rows
from adaptivefusion.state import file_sha256
from run_camera_pnp_comparison import write_json, arm_process, check_hashes, NATIVE
from run_orb_observation_controls import support

ARMS = ['F_MAGSAC', 'equal_count_subset_17', 'equal_count_subset_23', 'equal_count_subset_41']
CAMERA = [517.3, 516.5, 318.6, 255.3]


def verify(args):
    if args.output.exists(): raise ValueError('output exists; refuse overwrite')
    import cv2
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    plan = json.loads(args.plan.read_text())
    fconf, hconf = plan['fundamental'], plan['homography']
    if (fconf['threshold_px'], fconf['confidence'], fconf['max_iterations']) != (3., .999, 10000):
        raise ValueError('frozen F settings differ')
    if (hconf['threshold_px'], hconf['confidence'], hconf['max_iterations']) != (3., .999, 10000):
        raise ValueError('frozen H settings differ')
    parent_path = args.baseline_study/'fit-only.json'
    parent = json.loads(parent_path.read_text())
    parent_seal = json.loads((args.baseline_study/'fit-seal.json').read_text())
    if file_sha256(parent_path) != parent_seal['fit-only.json']:
        raise ValueError('V19 baseline changed')
    if len(parent['cases']) != args.expected_cases: raise ValueError('pair count differs from explicit expected count')
    hashes = {str(p.resolve()):file_sha256(p) for p in [args.plan, parent_path, Path(__file__),
        Path(__file__).parent/'adaptivefusion/matching_verification.py']}
    args.output.mkdir(); (args.output/'pairs').mkdir()
    write_json(args.output/'execution-protocol.json', dict(plan=plan, opencv=cv2.__version__,
        actual_scope='F verification and diagnostic H only; no GT, depth warp or new matching inference',
        requested_global_seed=17, threads=1, truth_read=False, input_sha256=hashes))
    cases = []; begin = time.perf_counter()
    for old in parent['cases']:
        case = {k:old[k] for k in ['scene', 'pair', 'source_timestamp', 'target_timestamp', 'raw_baseline']}
        previous = next(a for a in old['arms'] if a['camera_profile']=='rootsift_lightglue')
        case['original_rows'] = previous['rows']
        case['reference_fit'] = previous['fit']
        path = args.matching/previous['match_file']
        features = [args.matching/r['feature_file'] for r in old['features']]
        for p in [path, *features]:
            if parent['input_sha256'].get(str(p.resolve())) != file_sha256(p):
                raise ValueError('cached feature/match changed: '+str(p))
            hashes[str(p.resolve())] = file_sha256(p)
        with np.load(path) as archive: matches = archive['matches'].copy()
        points = []
        for side, p in enumerate(features):
            with np.load(p) as archive: points.append(archive['keypoints'][matches[:, side]].copy())
        if not all(np.isfinite(p).all() for p in points): raise ValueError('nonfinite cached pixels')
        lookup = {tuple(map(int,pair)):i for i,pair in enumerate(matches)}
        for row in case['original_rows']:
            i = lookup[row['source_feature'],row['target_feature']]
            if not np.array_equal(points[0][i], row['source_pixel']) or not np.array_equal(points[1][i], row['target_pixel']):
                raise ValueError('original pixel identity changed')
        case['models'] = {}; masks = {}; matrices = {}
        for name in ['F', 'H']:
            start = time.perf_counter()
            model = dict(input_matches=len(matches))
            try:
                if len(matches) < (8 if name=='F' else 4): raise ValueError('insufficient2D verification matches')
                cv2.setRNGSeed(17)
                if name=='F':
                    matrix, mask = cv2.findFundamentalMat(points[0], points[1], cv2.USAC_MAGSAC,
                        ransacReprojThreshold=3., confidence=.999, maxIters=10000)
                else:
                    matrix, mask = cv2.findHomography(points[0], points[1], cv2.USAC_MAGSAC,
                        ransacReprojThreshold=3., confidence=.999, maxIters=10000)
                matrix, mask = checked_mask(matrix, mask, len(matches))
                matrices[name] = matrix; masks[name] = mask
                pixel_support = support(points[0][mask], [1.]*int(mask.sum()), [640,480])
                pixel_support.pop('depth_range_m')  # A 2D model has no depth measurement.
                model.update(success=True, matrix=matrix.tolist(), inliers=int(mask.sum()),
                    source_support=pixel_support)
            except Exception as error:
                model.update(success=False, failure=str(error), traceback=traceback.format_exc())
            model['wall_s'] = time.perf_counter()-start
            case['models'][name] = model
        np.savez_compressed(args.output/'pairs'/f"{case['scene']}_{case['pair']}_models.npz",
            matches=matches, **{name+'_mask':mask for name,mask in masks.items()})
        case['arms'] = []
        if case['models']['F']['success']:
            selected = filtered_rows(matches, masks['F'], case['original_rows'])
            for name in ARMS:
                subset = selected if name=='F_MAGSAC' else equal_count_rows(case['original_rows'],len(selected),int(name.rsplit('_',1)[1]))
                case['arms'].append(dict(camera_profile=name, estimator=NATIVE, seed=17, rows=subset))
        else:
            for name in ARMS:
                case['arms'].append(dict(camera_profile=name, estimator=NATIVE, seed=17, rows=[],
                    failure='F verification failed: '+case['models']['F']['failure'] if name=='F_MAGSAC' else 'equal-count control unavailable: F verification failed'))
        cases.append(case)
        write_json(args.output/'pairs'/f"{case['scene']}_{case['pair']}.json", case)
        print(case['scene'],case['pair'],'2D/F/H/depth/Fdepth',len(matches),case['models']['F'].get('inliers'),
              case['models']['H'].get('inliers'),len(case['original_rows']),len(case['arms'][0]['rows']),flush=True)
    check_hashes(hashes)
    write_json(args.output/'verification.json',dict(cases=cases,input_sha256=hashes,truth_read=False,
        verification_phase_s=time.perf_counter()-begin,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def fit(args):
    if args.output.exists(): raise ValueError('output exists; refuse overwrite')
    data = json.loads((args.verification/'verification.json').read_text())
    hashes = data['input_sha256']; check_hashes(hashes)
    for p in args.verification.rglob('*'):
        if p.is_file(): hashes[str(p.resolve())] = file_sha256(p)
    parent = json.loads((args.baseline_study/'fit-only.json').read_text())
    binary = args.native_binary.resolve()
    if parent['input_sha256'].get(str(binary)) != file_sha256(binary): raise ValueError('V19 binary changed')
    hashes[str(binary)] = file_sha256(binary)
    args.output.mkdir(); (args.output/'inputs').mkdir(); (args.output/'arms').mkdir()
    begin = time.perf_counter(); cases = data['cases']
    for case in cases:
        for arm in case['arms']:
            name = arm['camera_profile']; rows = arm['rows']
            uv = [r['source_pixel'] for r in rows]; z = [r['source_depth_m'] for r in rows]
            prefix = f"{case['scene']}_{case['pair']}_{name}"
            if 'failure' in arm:
                arm.update(process_returncode=None,timed_out=False,process_wall_s=0.,peak_rss_kib=None,
                    fit=dict(applied=False,reason=arm['failure'],t_target_source=case['raw_baseline']))
            else:
                inp = args.output/'inputs'/f'{prefix}.json'; output = args.output/'arms'/prefix
                payload = dict(camera=CAMERA, ids=[r['source_feature'] for r in rows],
                    source=[unproject(CAMERA,p,depth).tolist() for p,depth in zip(uv,z)],
                    pixels=[r['target_pixel'] for r in rows], initial=case['raw_baseline'])
                write_json(inp,payload)
                arm.update(arm_process([sys.executable,'-B',str(Path(__file__).with_name('run_camera_pnp_comparison.py')),
                    'fit-arm','--input',str(inp),'--output',str(output),'--native-binary',str(binary),
                    '--estimator',NATIVE,'--seed','17'],output,args.output/'arms'/f'{prefix}.log',args.output/'arms'/f'{prefix}.rss.txt'))
            arm['fit'].setdefault('t_target_source',case['raw_baseline'])
            if not arm['fit']['applied'] and not np.array_equal(arm['fit']['t_target_source'],case['raw_baseline']):
                raise ValueError('exact raw fallback changed')
            arm['input_support'] = support(uv,z,[640,480])
            ids = set(arm['fit'].get('inlier_source_ids',[])); chosen = [r for r in rows if r['source_feature'] in ids]
            arm['inlier_support'] = support([r['source_pixel'] for r in chosen],[r['source_depth_m'] for r in chosen],[640,480])
            arm['check_pixels'] = dict(count=0,positive_count=0,median_px=None,p95_px=None)
        write_json(args.output/'inputs'/f"{case['scene']}_{case['pair']}_completed.json",case)
        print(case['scene'],case['pair'],'4 pose arms retained',flush=True)
    check_hashes(hashes)
    write_json(args.output/'fit-only.json',dict(cases=cases,input_sha256=hashes,truth_read=False,
        fit_phase_s=time.perf_counter()-begin))
    write_json(args.output/'fit-seal.json',{str(p.relative_to(args.output)):file_sha256(p)
        for p in args.output.rglob('*') if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    v = sub.add_parser('verify')
    for key in ['plan','baseline-study','matching','output']: v.add_argument('--'+key,type=Path,required=True)
    v.add_argument('--expected-cases',type=int,default=39)
    f = sub.add_parser('fit')
    for key in ['verification','baseline-study','native-binary','output']: f.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args(); {'verify':verify,'fit':fit}[args.command](args)


if __name__ == '__main__': main()
