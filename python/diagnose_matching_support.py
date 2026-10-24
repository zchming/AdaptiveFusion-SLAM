#!/usr/bin/env python3
"""Describe sealed pair fits without fitting, gating, inference or GT access."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from adaptivefusion.bridge_candidates import sample_depth
from adaptivefusion.frontend_features import read_features
from adaptivefusion.geometry import unproject
from adaptivefusion.state import file_sha256

CAMERA = [517.3, 516.5, 318.6, 255.3]


def grid_diagnostic(rows, depths, candidate, raw):
    pixels = np.asarray([[x, y] for y in range(8, 480, 16) for x in range(8, 640, 16)], float)
    # Exclude every fitted input location, not just selected inliers.
    distance, _ = cKDTree(np.asarray([r['source_pixel'] for r in rows])).query(pixels, p=np.inf)
    pixels = pixels[distance > 4.]
    target_tree = cKDTree(np.asarray([r['target_pixel'] for r in rows]))
    z = np.asarray([sample_depth(depths[0], p, 5000.) or np.nan for p in pixels])
    pixels, z = pixels[np.isfinite(z)], z[np.isfinite(z)]
    xyz = np.asarray([unproject(CAMERA, p, depth) for p, depth in zip(pixels, z)])
    if not len(xyz):
        return dict(grid_source_count=0, grid_common_count=0)
    errors = {}
    for name, transform in [('candidate', candidate), ('raw', raw)]:
        moved = xyz @ transform[:3, :3].T + transform[:3, 3]
        e = np.full(len(xyz), np.nan)
        for i, point in enumerate(moved):
            if point[2] <= 1e-8:
                continue
            uv = point[:2]/point[2]*CAMERA[:2]+CAMERA[2:]
            if target_tree.query(uv, p=np.inf)[0] <= 4.:
                continue
            target_z = sample_depth(depths[1], uv, 5000.)
            if target_z is not None:
                e[i] = point[2]-target_z
        errors[name] = e
    common = np.isfinite(errors['candidate']) & np.isfinite(errors['raw'])
    out = dict(grid_source_count=len(xyz), grid_common_count=int(common.sum()))
    if common.any():
        for name, error in errors.items():
            error = error[common]
            out[name+'_grid_z_abs_median_cm'] = float(np.median(np.abs(error))*100)
            out[name+'_grid_z_within3cm_fraction'] = float(np.mean(np.abs(error)<=.03))
    return out


def describe(case, arm, depths, metadata):
    rows = arm['rows']
    ids = set(arm['fit'].get('inlier_source_ids', []))
    selected = [r for r in rows if r['source_feature'] in ids]
    out = dict(scene=case['scene'], pair=case['pair'], arm=arm['camera_profile'],
               applied=arm['fit']['applied'], input_count=len(rows), inlier_count=len(selected),
               source_rgb_depth_ms=1000*(metadata[0]['depth_timestamp']-metadata[0]['rgb_timestamp']),
               target_rgb_depth_ms=1000*(metadata[1]['depth_timestamp']-metadata[1]['rgb_timestamp']))
    if len(selected) < 3:
        return out
    pixels = np.asarray([r['source_pixel'] for r in selected])
    xyz = np.asarray([unproject(CAMERA, r['source_pixel'], r['source_depth_m']) for r in selected])
    sv = np.linalg.svd(xyz-xyz.mean(axis=0), compute_uv=False)
    cells = np.clip((pixels/[640, 480]*4).astype(int), 0, 3)
    _, counts = np.unique(cells, axis=0, return_counts=True)
    out.update(occupied_4x4=len(counts), largest_cell_fraction=float(counts.max()/len(selected)),
               cloud_s3_s1=float(sv[-1]/sv[0]) if sv[0]>0 else None)
    try:
        out['hull_fraction'] = float(ConvexHull(pixels).volume/(640*480))
    except Exception:
        out['hull_fraction'] = None
    uv = np.asarray([r['target_pixel'] for r in selected])
    target_z = np.asarray([sample_depth(depths[1], p, 5000.) or np.nan for p in uv])
    valid = np.isfinite(target_z)
    out['target_depth_count'] = int(valid.sum())
    out['source_depth_median_m'] = float(np.median(xyz[:, 2]))
    candidate = np.asarray(arm['fit']['t_target_source'])
    raw = np.asarray(case['raw_baseline'])
    out.update(grid_diagnostic(rows, depths, candidate, raw))
    delta = candidate @ np.linalg.inv(raw)
    out['raw_candidate_translation_cm'] = float(np.linalg.norm(delta[:3, 3])*100)
    out['raw_candidate_rotation_deg'] = float(Rotation.from_matrix(delta[:3, :3]).magnitude()*180/np.pi)
    # Both transforms are scored on exactly the candidate's cached PnP inliers.
    # These are fitted observations, not independent validation observations.
    for name, transform in [('candidate', candidate), ('raw', raw)]:
        moved = xyz @ transform[:3, :3].T + transform[:3, 3]
        positive = moved[:, 2] > 1e-8
        out[name+'_positive_count'] = int(positive.sum())
        if positive.any():
            projected = moved[positive, :2]/moved[positive, 2, None]*CAMERA[:2]+CAMERA[2:]
            error = np.linalg.norm(projected-uv[positive], axis=1)
            out[name+'_reprojection_median_px'] = float(np.median(error))
            out[name+'_reprojection_p95_px'] = float(np.quantile(error, .95))
        mask = valid & positive
        if mask.any():
            zerror = moved[mask, 2]-target_z[mask]
            target_xyz = np.asarray([unproject(CAMERA, p, z) for p, z in zip(uv[mask], target_z[mask])])
            out[name+'_z_abs_median_cm'] = float(np.median(np.abs(zerror))*100)
            out[name+'_z_signed_median_cm'] = float(np.median(zerror)*100)
            out[name+'_z_within3cm_fraction'] = float(np.mean(np.abs(zerror)<=.03))
            out[name+'_xyz_median_cm'] = float(np.median(np.linalg.norm(moved[mask]-target_xyz, axis=1))*100)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['study', 'sources', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output exists; refuse overwrite')
    study_path = args.study/'fit-only.json'
    digest = file_sha256(study_path)
    seal = json.loads((args.study/'fit-seal.json').read_text())
    if digest != seal['fit-only.json']:
        raise ValueError('sealed fit changed')
    study = json.loads(study_path.read_text())
    sources = json.loads(args.sources.read_text())
    if any(set(paths)-{'frames', 'features', 'dataset'} for paths in sources.values()):
        raise ValueError('sources must contain inference paths only; no GT paths')
    records = []
    hashes = {str(study_path.resolve()): digest, str(args.sources.resolve()): file_sha256(args.sources),
              str(Path(__file__).resolve()): file_sha256(Path(__file__))}
    for case in study['cases']:
        paths = sources[case['scene']]
        metadata, depths = [], []
        for key in case['pair'].split('-'):
            feature_path = Path(paths['features'])/(key+'.features')
            meta = read_features(feature_path).metadata
            depth_path = Path(paths['dataset'])/meta['images'][1]
            metadata.append(meta)
            with Image.open(depth_path) as image:
                depths.append(np.array(image))
            for path in [feature_path, depth_path]:
                hashes[str(path.resolve())] = file_sha256(path)
        for arm in case['arms']:
            records.append(describe(case, arm, depths, metadata))
    if file_sha256(study_path) != digest:
        raise ValueError('input study changed during diagnosis')
    args.output.mkdir(parents=True)
    (args.output/'diagnosis.json').write_text(json.dumps(dict(
        records=records, input_sha256=hashes, truth_read=False, new_fits=0, new_inference=0,
        scope='descriptive diagnostics; fixed16px grid excludes4px square neighborhoods of both endpoint input matches; common valid grid points for raw/candidate; no gate or pose changes'), indent=2)+'\n')
    fields = list(dict.fromkeys(k for row in records for k in row))
    with (args.output/'diagnosis.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    print(json.dumps(dict(pairs=len(study['cases']), arms=len(records), new_fits=0, truth_read=False)))


if __name__ == '__main__':
    main()
