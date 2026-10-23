#!/usr/bin/env python3
"""Two ORB observation controls; reuse the sealed V17 native solver and evaluator."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from adaptivefusion.bridge_candidates import sample_depth
from adaptivefusion.frontend_features import read_features, descriptor_matches
from adaptivefusion.geometry import unproject, check_pose
from adaptivefusion.state import file_sha256
from run_camera_pnp_comparison import prepare, check_hashes, write_json, arm_process, NATIVE


ARMS = ('both_depth_all', 'source_depth_all')


def eligible_rows(matches, source, target, source_depth, target_depth):
    """Same source quality rule in both arms; target depth gates only the first."""
    rows = []
    for i, j, distance in matches:
        z = sample_depth(source_depth, source.keypoints[i, :2], 5000.)
        if z is None:
            continue
        target_ok = sample_depth(target_depth, target.keypoints[j, :2], 5000.) is not None
        rows.append(dict(source_feature=i, target_feature=j, hamming=distance,
                         source_depth_m=z, target_depth_valid=target_ok))
    if len({r['source_feature'] for r in rows}) != len(rows):
        raise ValueError('duplicate source observations')
    return rows


def build_cases(protocol, sources):
    cases, hashes = prepare(protocol, sources)
    for case in cases:
        paths = sources[case['scene']]
        proposal = json.loads((Path(paths['matching'])/'descriptor'/f"{case['pair']}.json").read_text())
        a, b = [read_features(Path(paths['features'])/f'{k}.features')
                for k in [proposal['source_keyframe'], proposal['target_keyframe']]]
        if a.metadata['matching'] != [.75, 64, True] or b.metadata['matching'] != [.75, 64, True]:
            raise ValueError('ORB policy changed')
        depths = []
        for feature in (a, b):
            with Image.open(Path(paths['dataset'])/feature.metadata['images'][1]) as image:
                depths.append(np.array(image))
        matches = descriptor_matches(a.descriptors, b.descriptors)
        if len(matches) != proposal['matches']:
            raise ValueError('descriptor match count differs from frozen cache')
        rows = eligible_rows(matches, a, b, *depths)
        both = [r for r in rows if r['target_depth_valid']]
        identities = lambda records: [(r['source_feature'], r['target_feature'], r['hamming']) for r in records]
        if identities(both) != identities(proposal['records']):
            raise ValueError('recomputed double-depth row identities/order changed')
        if [r['source_feature'] for r in both] != case['ids']:
            raise ValueError('V17 observation identity changed')
        case.update(descriptor_matches=len(matches), observation_rows=rows,
                    source_pixels=[a.keypoints[r['source_feature'], :2].tolist() for r in rows],
                    target_pixels=[b.keypoints[r['target_feature'], :2].tolist() for r in rows],
                    image_size=[a.metadata['width'], a.metadata['height']])
    check_hashes(hashes)
    return cases, hashes


def support(pixels, depths, size):
    if not len(pixels):
        return dict(count=0, occupied_4x4=0, bbox_px=None, depth_range_m=None)
    p = np.asarray(pixels)
    cells = np.clip((p/size*4).astype(int), 0, 3)
    return dict(count=len(p), occupied_4x4=len(set(map(tuple, cells.tolist()))),
                bbox_px=np.ptp(p, axis=0).tolist(), depth_range_m=[min(depths), max(depths)])


def run(args):
    if args.output.exists():
        raise ValueError('output exists; refuse overwrite')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise ValueError(key+' must be 1')
    protocol = json.loads(args.protocol.read_text())
    sources = json.loads(args.sources.read_text())
    repo = Path(__file__).resolve().parents[1]
    for rel, digest in protocol['source_sha256'].items():
        if file_sha256(repo/rel) != digest:
            raise ValueError('frozen implementation changed: '+rel)
    parent = json.loads((args.parent_study/'fit-only.json').read_text())
    parent_seal = json.loads((args.parent_study/'fit-seal.json').read_text())
    for rel, digest in parent_seal.items():
        if not (args.parent_study/rel).resolve().is_relative_to(args.parent_study.resolve()):
            raise ValueError('unsafe parent seal path')
        if file_sha256(args.parent_study/rel) != digest:
            raise ValueError('parent study seal changed')
    binary = args.native_binary.resolve()
    if parent['input_sha256'].get(str(binary)) != file_sha256(binary):
        raise ValueError('binary differs from V17')
    cases, hashes = build_cases(protocol, sources)
    if len(cases) != 39:
        raise ValueError('expected frozen 39 pairs')
    for case in cases:
        old = next(c for c in parent['cases'] if (c['scene'], c['pair']) == (case['scene'], case['pair']))
        if old['raw_baseline'] != case['raw_baseline']:
            raise ValueError('raw fallback differs from V17')
    for path in [args.protocol, args.sources, binary, args.parent_study/'fit-seal.json']:
        hashes[str(path.resolve())] = file_sha256(path)
    for base in ('python', 'src', 'include', 'app'):
        for path in (repo/base).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.cpp', '.h'):
                hashes[str(path)] = file_sha256(path)
    args.output.mkdir()
    camera = [517.3, 516.5, 318.6, 255.3]
    write_json(args.output/'execution-protocol.json', dict(
        arms=ARMS, cases=39, fits=78, camera=camera, requested_seed=17,
        estimator=NATIVE, sampler='OpenCV 4.6 legacy fixed local RNG; no independent seed claim',
        fit_rows='all eligible; no held-out residual metric', truth_read=False,
        source_depth='original sample_depth: 3x3 all nonzero, range <= 0.05 m, scale 5000',
        limits=dict(address_space_gib=4, seconds_per_arm=60), input_sha256=hashes))
    (args.output/'inputs').mkdir(); (args.output/'arms').mkdir()
    start = time.perf_counter()
    for case in cases:
        case['arms'] = []
        for name in ARMS:
            indices = [i for i, r in enumerate(case['observation_rows'])
                       if name == 'source_depth_all' or r['target_depth_valid']]
            rows = [case['observation_rows'][i] for i in indices]
            ids = [r['source_feature'] for r in rows]
            uv = [case['source_pixels'][i] for i in indices]
            z = [r['source_depth_m'] for r in rows]
            pixels = [case['target_pixels'][i] for i in indices]
            source = [unproject(camera, p, depth).tolist() for p, depth in zip(uv, z)]
            prefix = f"{case['scene']}_{case['pair']}_{name}"
            input_path = args.output/'inputs'/f'{prefix}.json'
            write_json(input_path, dict(source=source, pixels=pixels, ids=ids,
                                        camera=camera, initial=case['raw_baseline']))
            output = args.output/'arms'/prefix
            arm = arm_process([sys.executable, '-B', str(repo/'python/run_camera_pnp_comparison.py'),
                               'fit-arm', '--input', str(input_path), '--output', str(output),
                               '--native-binary', str(binary), '--estimator', NATIVE, '--seed', '17'],
                              output, args.output/'arms'/f'{prefix}.log',
                              args.output/'arms'/f'{prefix}.rss.txt')
            arm.update(camera_profile=name, estimator=NATIVE, seed=17,
                       input_support=support(uv, z, case['image_size']))
            arm['fit'].setdefault('t_target_source', case['raw_baseline'])
            returned = check_pose(arm['fit']['t_target_source'])
            if not arm['fit']['applied'] and not np.array_equal(returned, case['raw_baseline']):
                raise ValueError('fallback changed')
            inliers = set(arm['fit'].get('inlier_source_ids', []))
            selected = [i for i, value in enumerate(ids) if value in inliers]
            arm['inlier_support'] = support([uv[i] for i in selected], [z[i] for i in selected], case['image_size'])
            arm['check_pixels'] = dict(count=0, positive_count=0, median_px=None, p95_px=None,
                                       scope='not measured: all eligible observations used for fit')
            case['arms'].append(arm)
        write_json(args.output/'inputs'/f"{case['scene']}_{case['pair']}_completed.json", case)
        print(case['scene'], case['pair'], '2 arms retained', flush=True)
    check_hashes(hashes)
    write_json(args.output/'fit-only.json', dict(cases=cases, input_sha256=hashes, truth_read=False,
                                               fit_wall_s=time.perf_counter()-start))
    write_json(args.output/'fit-seal.json', {str(p.relative_to(args.output)):file_sha256(p)
               for p in args.output.rglob('*') if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'sources', 'native-binary', 'parent-study', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
