#!/usr/bin/env python3
"""Development-only 3D-3D/3D-2D comparison on frozen shared correspondences."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from adaptivefusion.bridge_candidates import sample_depth
from adaptivefusion.frontend_features import read_features
from adaptivefusion.geometry import unproject, inverse
from adaptivefusion.pair_model_comparison import CONFIG, refine_pair
from adaptivefusion.proposal_diagnostics import fixed_model_errors
from adaptivefusion.trajectory import load_frames
from adaptivefusion.state import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['matching', 'features', 'frames', 'dataset', 'truth', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('output exists; refuse overwrite')
    frames = load_frames(args.frames)
    paths = sorted((args.matching/'descriptor').glob('*.json'))
    if not paths: raise ValueError('no proposal records')
    cases = []
    hashes = {'frames': file_sha256(args.frames)}
    for path in paths:
        proposal = json.loads(path.read_text())
        hashes['proposal/'+path.name] = file_sha256(path)
        features, depths = [], []
        for kid in [proposal['source_keyframe'], proposal['target_keyframe']]:
            fp = args.features/f'{kid}.features'; f = read_features(fp); m = f.metadata
            if m['keyframe_id'] != kid or frames[m['source_frame_id']].timestamp != m['rgb_timestamp']:
                raise ValueError('frame/feature identity mismatch')
            dp = args.dataset/m['images'][1]
            if not dp.resolve().is_relative_to(args.dataset.resolve()): raise ValueError('unsafe depth path')
            d = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
            if d is None or d.dtype != np.uint16 or d.shape != (m['height'], m['width']): raise ValueError('invalid native depth')
            features.append(f); depths.append(d)
            hashes['feature/'+fp.name], hashes['depth/'+m['images'][1]] = file_sha256(fp), file_sha256(dp)
        fa, fb = features
        if fa.metadata['camera'] != fb.metadata['camera'] or fa.metadata['depth'] != fb.metadata['depth']: raise ValueError('calibration mismatch')
        source, target, pixels, ids = [], [], [], []
        for row in proposal['records']:
            i, j = row['source_feature'], row['target_feature']
            u, v = fa.keypoints[i, :2], fb.keypoints[j, :2]
            za = sample_depth(depths[0], u, fa.metadata['depth'][0])
            zb = sample_depth(depths[1], v, fb.metadata['depth'][0])
            if za is None or zb is None: raise ValueError('original records not reproducible')
            source.append(unproject(fa.metadata['camera'], u, za)); target.append(unproject(fb.metadata['camera'], v, zb))
            pixels.append(v); ids.append(i)
        source, target, pixels = [np.array(v).reshape(-1, n) for v, n in [(source, 3), (target, 3), (pixels, 2)]]
        ids = np.asarray(ids, int)
        if len(ids) != proposal['depth_valid_matches'] or len(set(ids)) != len(ids): raise ValueError('record identity mismatch')
        train = ids % 2 == 0
        raw = inverse(frames[fb.metadata['source_frame_id']].t_wc)@frames[fa.metadata['source_frame_id']].t_wc
        initializations = {'raw_baseline': raw}
        if 't_target_source' in proposal['verification']:
            initializations['frozen_3d3d_proposal'] = np.array(proposal['verification']['t_target_source'])
        arms = []
        for init_name, init in initializations.items():
            for model in ['3d3d', '3d2d']:
                fit = refine_pair(source[train], pixels[train], fa.metadata['camera'], init,
                                  model=model, target_points=target[train] if model == '3d3d' else None)
                transform = np.array(fit['t_target_source'])
                arms.append(dict(initialization=init_name, model=model, fit=fit,
                    validation=fixed_model_errors(transform, source[~train], target[~train], pixels[~train], fa.metadata['camera'])))
        cases.append(dict(pair=path.stem, source_timestamp=fa.metadata['rgb_timestamp'], target_timestamp=fb.metadata['rgb_timestamp'],
                          fit_source_ids=ids[train].tolist(), validation_source_ids=ids[~train].tolist(),
                          original_accepted=proposal['verification']['accepted'], raw_baseline=raw.tolist(), arms=arms))
    args.output.mkdir(parents=True)
    pretruth = args.output/'fit-only.json'
    pretruth.write_text(json.dumps(dict(config=CONFIG, cases=cases, input_sha256=hashes, truth_read=False), indent=2)+'\n')
    frozen_sha = file_sha256(pretruth)
    from adaptivefusion.evaluation import read_tum
    from adaptivefusion.candidate_validation import truth_endpoints, relative_error
    times, poses = read_tum(args.truth)
    for case in cases:
        endpoints = truth_endpoints(times, poses, case['source_timestamp'], case['target_timestamp'])
        case['raw_baseline_truth'] = {mode: relative_error(np.array(case['raw_baseline']), *endpoints[mode]) for mode in ['nearest', 'interpolated'] if endpoints[mode] is not None}
        for arm in case['arms']:
            arm['truth_errors'] = {mode: relative_error(np.array(arm['fit']['t_target_source']), *endpoints[mode]) for mode in ['nearest', 'interpolated'] if endpoints[mode] is not None}
    assert file_sha256(pretruth) == frozen_sha
    (args.output/'report.json').write_text(json.dumps(dict(cases=cases, config=CONFIG, fit_only_sha256=frozen_sha,
        truth_sha256=file_sha256(args.truth), scope='development pair objectives only; target-depth eligibility is shared and thus still a selection dependency; objective noise assumptions differ; no trajectory/BA/online feedback'), indent=2)+'\n')
    print(json.dumps({'pairs': len(cases), 'arms': sum(len(c['arms']) for c in cases)}))


if __name__ == '__main__': main()
