#!/usr/bin/env python3
"""Shared RootSIFT NN/LightGlue matching, then fixed native PnP (separate commands)."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import numpy as np
from PIL import Image
from adaptivefusion.bridge_candidates import sample_depth
from adaptivefusion.frontend_features import read_features
from adaptivefusion.geometry import unproject
from adaptivefusion.state import file_sha256
from run_camera_pnp_comparison import write_json, arm_process, NATIVE, check_hashes
from run_orb_observation_controls import support

ARMS = ('rootsift_nn', 'rootsift_lightglue')
WEIGHT_SHA = '5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0'
COMMIT = 'eb42fee2d71449efb0aa5c10549752b5d75384d8'
CAMERA = [517.3, 516.5, 318.6, 255.3]


def mutual_ratio_matches(first, second, ratio=.8):
    """Euclidean RootSIFT, strict Lowe ratio and reverse nearest-neighbor check."""
    import cv2
    if len(first) == 0 or len(second) < 2:
        return np.empty((0, 2), np.int64)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(first, second, k=2)
    backward = matcher.match(second, first)
    reverse = {r.queryIdx:r.trainIdx for r in backward}
    return np.asarray([(a.queryIdx, a.trainIdx) for a, b in forward
                       if a.distance < ratio*b.distance and reverse[a.trainIdx] == a.queryIdx],
                      dtype=np.int64).reshape(-1, 2)


def validate_features(f):
    n = len(f['keypoints'])
    if (f['keypoints'].shape != (n, 2) or f['descriptors'].shape != (n, 128)
            or f['scales'].shape != (n,) or f['oris'].shape != (n,)
            or f['image_size'].shape != (2,) or n > 2048
            or not all(np.isfinite(v).all() for v in f.values())
            or np.any(f['scales'] <= 0) or np.any(f['oris'] < 0) or np.any(f['oris'] > 2*np.pi)
            or np.any(f['keypoints'] < 0) or np.any(f['keypoints'] >= f['image_size'])
            or not np.allclose(np.linalg.norm(f['descriptors'], axis=1), 1., atol=1e-5)):
        raise ValueError('invalid RootSIFT features/metadata')


def source_rows(pairs, first, second, depth):
    rows = []
    for i, j in pairs:
        z = sample_depth(depth, first['keypoints'][i], 5000.)
        if z is not None:
            rows.append(dict(source_feature=int(i), target_feature=int(j), source_depth_m=z,
                             source_pixel=first['keypoints'][i].tolist(),
                             target_pixel=second['keypoints'][j].tolist()))
    return rows


def match(args):
    # Optional Torch/CUDA dependencies are confined to this command.
    if args.output.exists():
        raise ValueError('output exists; refuse overwrite')
    import cv2
    import torch
    sys.path.insert(0, str(args.lightglue_source))
    from lightglue import LightGlue
    from lightglue.sift import sift_to_rootsift, filter_dog_point
    commit = subprocess.check_output(['git', '-C', str(args.lightglue_source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != COMMIT or file_sha256(args.weights) != WEIGHT_SHA:
        raise ValueError('LightGlue source/weight differs from frozen candidate')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU resource-profile change')
    parent_path = args.baseline_study/'fit-only.json'
    parent_seal = json.loads((args.baseline_study/'fit-seal.json').read_text())
    if file_sha256(parent_path) != parent_seal['fit-only.json']:
        raise ValueError('baseline fit data changed')
    baseline = json.loads(parent_path.read_text())
    if len(baseline['cases']) != args.expected_cases:
        raise ValueError('pair count differs from explicit expected count')
    sources = json.loads(args.sources.read_text())
    hashes = {str(p.resolve()):file_sha256(p) for p in [parent_path, args.sources, args.weights, Path(__file__)]}
    for filename in ['sift.py', 'lightglue.py', 'utils.py']:
        p = args.lightglue_source/'lightglue'/filename
        hashes[str(p.resolve())] = file_sha256(p)
    torch.set_num_threads(8); torch.manual_seed(17)
    cv2.setNumThreads(8); cv2.setRNGSeed(17); cv2.ocl.setUseOpenCL(False)
    torch.cuda.set_per_process_memory_fraction(11*1024**3/torch.cuda.get_device_properties(0).total_memory)
    args.output.mkdir(); (args.output/'features').mkdir(); (args.output/'pairs').mkdir()
    write_json(args.output/'execution-protocol.json', dict(
        pairs=args.expected_cases, arms=args.matchers, input_resolution=[640, 480], camera=CAMERA,
        sift=dict(max_features=2048, contrast_threshold=.0066667, edge_threshold=10,
                  octave_layers=4, duplicate_filter_radius=0, grayscale='OpenCV uint8 BGR2GRAY',
                  resize=None, coordinates='OpenCV .pt unchanged', rootsift='official conversion'),
        nn=dict(metric='Euclidean', strict_ratio=.8, mutual=True),
        lightglue=dict(commit=commit, weight_sha256=WEIGHT_SHA, n_layers=9,
                       depth_confidence=.95, width_confidence=.99, filter_threshold=.1,
                       flash=True, mp=False),
        depth='source-only; unchanged sample_depth 3x3, 0.05m range, scale5000',
        feature_budget_change='2048 SIFT vs historical ORB 1000; NN/LG share all features',
        gpu=torch.cuda.get_device_name(0), allocator_budget_bytes=11*1024**3,
        versions=dict(torch=torch.__version__, opencv=cv2.__version__, numpy=np.__version__),
        truth_read=False, input_sha256=hashes))
    start = time.perf_counter()
    matcher = LightGlue(features=None, input_dim=128, add_scale_ori=True).eval()
    state = torch.load(args.weights, map_location='cpu', weights_only=True)
    for i in range(9):
        state = {k.replace(f'self_attn.{i}', f'transformers.{i}.self_attn').replace(
            f'cross_attn.{i}', f'transformers.{i}.cross_attn'):v for k, v in state.items()}
    loaded = matcher.load_state_dict(state, strict=False)
    if set(loaded.missing_keys) != {'confidence_thresholds'} or loaded.unexpected_keys:
        raise ValueError('unexpected pretrained state keys')
    matcher = matcher.cuda(); torch.cuda.synchronize()
    model_load_s = time.perf_counter()-start
    sift = cv2.SIFT_create(nfeatures=2048, contrastThreshold=.0066667, edgeThreshold=10, nOctaveLayers=4)
    cached = {}; cases = []

    def feature(scene, kid):
        key = f'{scene}_{kid}'
        if key in cached:
            return cached[key]
        paths = sources[scene]; fp = Path(paths['features'])/f'{kid}.features'
        metadata = read_features(fp).metadata
        rgb = Path(paths['dataset'])/metadata['images'][0]
        dp = Path(paths['dataset'])/metadata['images'][1]
        for p in [fp, rgb, dp]:
            hashes[str(p.resolve())] = file_sha256(p)
        begin = time.perf_counter()
        # Match the declared integer RGB-to-gray path explicitly.
        bgr = cv2.imread(str(rgb), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (480, 640, 3):
            raise ValueError('expected native 640x480 RGB')
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        detections, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            f = dict(keypoints=np.empty((0, 2), np.float32), descriptors=np.empty((0, 128), np.float32),
                     scales=np.empty(0, np.float32), oris=np.empty(0, np.float32),
                     image_size=np.array([640, 480], np.float32))
        else:
            xy = np.asarray([k.pt for k in detections], np.float32)
            scores = np.asarray([k.response for k in detections], np.float32)
            scales = np.asarray([k.size for k in detections], np.float32)
            angles = np.deg2rad(np.asarray([k.angle for k in detections], np.float32))
            keep = filter_dog_point(xy, scales, angles, gray.shape, 0, scores=scores)
            if len(keep) > 2048:
                keep = keep[torch.topk(torch.from_numpy(scores[keep]), 2048).indices.numpy()]
            f = dict(keypoints=xy[keep], descriptors=sift_to_rootsift(torch.from_numpy(descriptors[keep])).numpy(),
                     scales=scales[keep], oris=angles[keep], image_size=np.array([640, 480], np.float32))
        validate_features(f)
        record = dict(feature_file=f'features/{key}.npz', extraction_s=time.perf_counter()-begin,
                      feature_count=len(f['keypoints']), rgb_sha256=hashes[str(rgb.resolve())],
                      depth_path=str(dp.resolve()))
        np.savez_compressed(args.output/record['feature_file'], **f)
        cached[key] = f, record
        return f, record

    run_begin = time.perf_counter()
    for old in baseline['cases']:
        case = {k:old[k] for k in ['scene', 'pair', 'source_timestamp', 'target_timestamp', 'raw_baseline']}
        case['arms'] = []
        try:
            a, b = map(int, case['pair'].split('-'))
            f0, r0 = feature(case['scene'], a); f1, r1 = feature(case['scene'], b)
            case['features'] = [r0, r1]
            with Image.open(r0['depth_path']) as image:
                depth = np.array(image)
            if depth.dtype != np.uint16 or depth.shape != (480, 640):
                raise ValueError('invalid source depth')
            for name in args.matchers:
                torch.cuda.reset_peak_memory_stats()
                begin = time.perf_counter()
                arm = dict(camera_profile=name, estimator=NATIVE, seed=17)
                try:
                    if name == 'rootsift_nn':
                        pairs = mutual_ratio_matches(f0['descriptors'], f1['descriptors'])
                        scores = np.full(len(pairs), np.nan)
                    elif not len(f0['keypoints']) or not len(f1['keypoints']):
                        pairs = np.empty((0, 2), np.int64); scores = np.empty(0)
                        arm['empty_features'] = True
                    else:
                        gpu = [{k:torch.from_numpy(v)[None].cuda() for k,v in f.items()} for f in [f0, f1]]
                        with torch.inference_mode():
                            result = matcher(dict(image0=gpu[0], image1=gpu[1]))
                        pairs = result['matches'][0].cpu().numpy()
                        scores = result['scores'][0].cpu().numpy()
                        arm['stop_layer'] = int(result['stop'])
                        del gpu, result
                    torch.cuda.synchronize()
                    arm.update(matching_s=time.perf_counter()-begin,
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
                    for side, f in enumerate([f0, f1]):
                        if (len(set(pairs[:, side].tolist())) != len(pairs)
                                or np.any(pairs[:, side] < 0) or np.any(pairs[:, side] >= len(f['keypoints']))):
                            raise ValueError('matcher index/uniqueness contract failed')
                    order = np.argsort(pairs[:, 0], kind='stable'); pairs = pairs[order]; scores = scores[order]
                    pair_file = f"pairs/{case['scene']}_{case['pair']}_{name}.npz"
                    np.savez_compressed(args.output/pair_file, matches=pairs, scores=scores)
                    arm.update(match_file=pair_file, matches=len(pairs), rows=source_rows(pairs, f0, f1, depth))
                except Exception as error:
                    arm.update(failure=str(error), traceback=traceback.format_exc(), rows=[], matches=None,
                               matching_s=time.perf_counter()-begin)
                    if isinstance(error, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                case['arms'].append(arm)
        except Exception as error:
            case['arms'] = [dict(camera_profile=name, estimator=NATIVE, seed=17, rows=[], matches=None,
                                 failure=str(error), traceback=traceback.format_exc()) for name in args.matchers]
        cases.append(case)
        write_json(args.output/'pairs'/f"{case['scene']}_{case['pair']}.json", case)
        print(case['scene'], case['pair'], [(a['camera_profile'], a['matches'], len(a['rows'])) for a in case['arms']], flush=True)
    check_hashes(hashes)
    write_json(args.output/'matching.json', dict(cases=cases, input_sha256=hashes, truth_read=False,
        model_load_s=model_load_s, matching_phase_wall_s=time.perf_counter()-run_begin,
        unique_features=len(cached), extraction_total_s=sum(r['extraction_s'] for _, r in cached.values()),
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def fit(args):
    if args.output.exists():
        raise ValueError('output exists; refuse overwrite')
    matching = json.loads((args.matching/'matching.json').read_text())
    hashes = matching['input_sha256']; check_hashes(hashes)
    for p in args.matching.rglob('*'):
        if p.is_file(): hashes[str(p.resolve())] = file_sha256(p)
    binary = args.native_binary.resolve()
    baseline = json.loads((args.baseline_study/'fit-only.json').read_text())
    if baseline['input_sha256'].get(str(binary)) != file_sha256(binary):
        raise ValueError('native binary changed since ORB control')
    hashes[str(binary)] = file_sha256(binary)
    args.output.mkdir(); (args.output/'inputs').mkdir(); (args.output/'arms').mkdir()
    cases = matching['cases']; start = time.perf_counter()
    for case in cases:
        for arm in case['arms']:
            name = arm['camera_profile']; rows = arm['rows']
            uv = [r['source_pixel'] for r in rows]; z = [r['source_depth_m'] for r in rows]
            source = [unproject(CAMERA, p, depth).tolist() for p, depth in zip(uv, z)]
            prefix = f"{case['scene']}_{case['pair']}_{name}"
            if 'failure' in arm:
                arm.update(process_returncode=None, timed_out=False, process_wall_s=0., peak_rss_kib=None,
                           fit=dict(applied=False, reason='matching failed: '+arm['failure'],
                                    t_target_source=case['raw_baseline']))
            else:
                inp = args.output/'inputs'/f'{prefix}.json'; output = args.output/'arms'/prefix
                write_json(inp, dict(source=source, pixels=[r['target_pixel'] for r in rows],
                    ids=[r['source_feature'] for r in rows], camera=CAMERA, initial=case['raw_baseline']))
                arm.update(arm_process([sys.executable, '-B', str(Path(__file__).with_name('run_camera_pnp_comparison.py')),
                    'fit-arm', '--input', str(inp), '--output', str(output), '--native-binary', str(binary),
                    '--estimator', NATIVE, '--seed', '17'], output, args.output/'arms'/f'{prefix}.log',
                    args.output/'arms'/f'{prefix}.rss.txt'))
                arm['fit'].setdefault('t_target_source', case['raw_baseline'])
            if not arm['fit']['applied'] and not np.array_equal(arm['fit']['t_target_source'], case['raw_baseline']):
                raise ValueError('exact fallback changed')
            arm['input_support'] = support(uv, z, [640, 480])
            inliers = set(arm['fit'].get('inlier_source_ids', []))
            selected = [r for r in rows if r['source_feature'] in inliers]
            arm['inlier_support'] = support([r['source_pixel'] for r in selected],
                [r['source_depth_m'] for r in selected], [640, 480])
            arm['check_pixels'] = dict(count=0, positive_count=0, median_px=None, p95_px=None)
        print(case['scene'], case['pair'], len(case['arms']), 'pose arms retained', flush=True)
    check_hashes(hashes)
    write_json(args.output/'fit-only.json', dict(cases=cases, input_sha256=hashes, truth_read=False,
                                               fit_wall_s=time.perf_counter()-start))
    write_json(args.output/'fit-seal.json', {str(p.relative_to(args.output)):file_sha256(p)
               for p in args.output.rglob('*') if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    m = sub.add_parser('match')
    for key in ('baseline-study', 'sources', 'lightglue-source', 'weights', 'output'):
        m.add_argument('--'+key, type=Path, required=True)
    m.add_argument('--expected-cases', type=int, default=39)
    m.add_argument('--matchers', choices=ARMS, nargs='+', default=list(ARMS))
    f = sub.add_parser('fit')
    for key in ('matching', 'baseline-study', 'native-binary', 'output'):
        f.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    {'match':match, 'fit':fit}[args.command](args)


if __name__ == '__main__':
    main()
