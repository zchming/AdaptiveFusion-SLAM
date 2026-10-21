#!/usr/bin/env python3
"""Describe fixed proposal errors; no new matches, fitting or admission decisions."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from adaptivefusion.bridge_candidates import sample_depth
from adaptivefusion.candidate_validation import truth_endpoints, relative_error
from adaptivefusion.evaluation import read_tum
from adaptivefusion.frontend_features import read_features
from adaptivefusion.geometry import unproject, inverse
from adaptivefusion.proposal_diagnostics import (
    fixed_model_errors, support_geometry, epipolar_errors, centroid_decomposition)
from adaptivefusion.state import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['evaluation', 'matching', 'features', 'dataset', 'truth', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; refuse overwrite')
    evaluation = json.loads(args.evaluation.read_text())
    if file_sha256(args.truth) != evaluation['truth_sha256']:
        raise ValueError('truth provenance mismatch')
    times, poses = read_tum(args.truth)
    hashes = {'evaluation': file_sha256(args.evaluation), 'truth': file_sha256(args.truth)}
    cases = []
    for case in evaluation['cases']:
        a, b = case['source_keyframe'], case['target_keyframe']
        proposal_path = args.matching/'descriptor'/f'{a}-{b}.json'
        proposal = json.loads(proposal_path.read_text())
        hashes[f'proposal/{a}-{b}'] = file_sha256(proposal_path)
        candidate = np.array(proposal['verification']['t_target_source'])
        if not np.array_equal(candidate, case['candidate_t_target_source']):
            raise ValueError('fixed model changed')
        features, depths = [], []
        for kid in [a, b]:
            p = args.features/f'{kid}.features'
            f = read_features(p)
            if f.metadata['keyframe_id'] != kid:
                raise ValueError('feature identity mismatch')
            dp = args.dataset/f.metadata['images'][1]
            if not dp.resolve().is_relative_to(args.dataset.resolve()):
                raise ValueError('image path escapes dataset')
            d = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
            if d is None or d.dtype != np.uint16 or d.shape != (f.metadata['height'], f.metadata['width']):
                raise ValueError('invalid depth')
            hashes[f'feature/{kid}'], hashes[f'depth/{kid}'] = file_sha256(p), file_sha256(dp)
            features.append(f); depths.append(d)
        fa, fb = features
        camera = fa.metadata['camera']
        if camera != fb.metadata['camera'] or fa.metadata['depth'] != fb.metadata['depth']:
            raise ValueError('calibration mismatch')
        xyz, target, uv_a, uv_b, ids = [], [], [], [], []
        for row in proposal['records']:
            i, j = row['source_feature'], row['target_feature']
            pa, pb = fa.keypoints[i, :2], fb.keypoints[j, :2]
            za = sample_depth(depths[0], pa, fa.metadata['depth'][0])
            zb = sample_depth(depths[1], pb, fb.metadata['depth'][0])
            if za is None or zb is None:
                raise ValueError('original depth-valid correspondence not reproducible')
            xyz.append(unproject(camera, pa, za)); target.append(unproject(camera, pb, zb))
            uv_a.append(pa); uv_b.append(pb); ids.append(i)
        xyz, target, uv_a, uv_b, ids = map(np.asarray, [xyz, target, uv_a, uv_b, ids])
        if len(ids) != proposal['depth_valid_matches'] or len(set(ids)) != len(ids):
            raise ValueError('record count/identity mismatch')
        old_inliers = np.isin(ids, proposal['verification']['inlier_point_ids'])
        reproduced = np.linalg.norm(xyz@candidate[:3, :3].T+candidate[:3, 3]-target, axis=1) <= .03
        if not np.array_equal(old_inliers, reproduced):
            raise ValueError('original inliers changed')
        masks = {'all': np.ones(len(ids), bool), 'fit': ids % 2 == 0,
                 'validation': ids % 2 == 1, 'original_inliers': old_inliers}
        rgb = truth_endpoints(times, poses, *[f.metadata['rgb_timestamp'] for f in features])
        depth = truth_endpoints(times, poses, *[f.metadata['depth_timestamp'] for f in features])
        models = {'candidate': candidate, 'raw_baseline': np.array(case['raw_baseline_t_target_source'])}
        for name, truth in [('rgb', rgb), ('depth', depth)]:
            if truth['interpolated'] is not None:
                models[name+'_truth'] = inverse(truth['interpolated'][1])@truth['interpolated'][0]
        result = {k: case[k] for k in ['source_keyframe', 'target_keyframe', 'crossed_cuts']}
        result.update(rgb_minus_depth_s=[f.metadata['rgb_timestamp']-f.metadata['depth_timestamp'] for f in features],
                      source_feature_ids=ids.tolist(), masks={k: ids[m].tolist() for k, m in masks.items()}, groups={})
        for group, mask in masks.items():
            result['groups'][group] = {
                'source_support': support_geometry(xyz[mask], uv_a[mask], fa.metadata['width'], fa.metadata['height']),
                'target_support': support_geometry(target[mask], uv_b[mask], fb.metadata['width'], fb.metadata['height']),
                'models': {name: {**fixed_model_errors(t, xyz[mask], target[mask], uv_b[mask], camera),
                                 'epipolar': epipolar_errors(t, uv_a[mask], uv_b[mask], camera)}
                           for name, t in models.items()}}
        if 'rgb_truth' in models:
            result['inlier_centroid_decomposition'] = centroid_decomposition(candidate, models['rgb_truth'], xyz[old_inliers])
            result['inverse_candidate_rgb_truth_error'] = relative_error(inverse(candidate), *rgb['interpolated'])
            result['candidate_rgb_truth_error'] = relative_error(candidate, *rgb['interpolated'])
        if 'depth_truth' in models:
            result['candidate_depth_truth_error'] = relative_error(candidate, *depth['interpolated'])
        cases.append(result)
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(dict(cases=cases, input_sha256=hashes,
        scope='descriptive fixed-model diagnostics, existing zero-distortion pinhole camera; no cause identification, tuning, fit, map mutation or truth-based admission'), indent=2)+'\n')
    print(json.dumps({'cases': len(cases), 'new_pose_fit': False}))


if __name__ == '__main__':
    main()
