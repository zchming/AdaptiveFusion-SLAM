#!/usr/bin/env python3
"""Frozen camera x estimator comparison. Fit and GT evaluation are separate commands."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import numpy as np
from PIL import Image
from adaptivefusion.bridge_candidates import sample_depth
from adaptivefusion.frontend_features import read_features
from adaptivefusion.geometry import inverse, unproject, check_pose
from adaptivefusion.native_pnp import native_fit
from adaptivefusion.pair_model_comparison import CONFIG, refine_pair
from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames


LOCAL = 'current_3d2d_huber_refinement'
NATIVE = 'existing_native_pnp_ransac'


def write_json(path, obj):
    with Path(path).open('x') as f:
        f.write(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def prepare(protocol, sources):
    """Reproduce frozen eligibility/identities before any fit; never read truth."""
    cases, hashes = [], {}
    for scene in protocol['audit_summary']:
        name = scene['scene']; paths = sources[name]
        frame_path = Path(paths['frames']); frames = load_frames(frame_path)
        hashes[str(frame_path)] = file_sha256(frame_path)
        actual = {p.stem for p in (Path(paths['matching'])/'descriptor').glob('*.json')}
        if actual != {p['pair'] for p in scene['pairs']}:
            raise ValueError('frozen pair inventory differs: '+name)
        for expected in scene['pairs']:
            pair = expected['pair']; path = Path(paths['matching'])/'descriptor'/f'{pair}.json'
            digest = file_sha256(path)
            if digest != expected['sha256']: raise ValueError('proposal hash mismatch: '+name+'/'+pair)
            hashes[str(path)] = digest; proposal = json.loads(path.read_text())
            a, b = proposal['source_keyframe'], proposal['target_keyframe']
            if pair != f'{a}-{b}': raise ValueError('proposal pair identity mismatch')
            features, depth = [], []
            for kid in [a, b]:
                fp = Path(paths['features'])/f'{kid}.features'
                feature = read_features(fp); m = feature.metadata
                if (m['keyframe_id'] != kid or m['camera'] != scene['feature_cameras'][0]
                        or m['depth'] != [5000., .1, 8.]
                        or frames[m['source_frame_id']].timestamp != m['rgb_timestamp']):
                    raise ValueError('feature/frame/calibration mismatch')
                dataset = Path(paths['dataset']).resolve(); dp = dataset/m['images'][1]
                if not dp.resolve().is_relative_to(dataset): raise ValueError('unsafe depth path')
                with Image.open(dp) as image: image = np.array(image)
                if image.dtype != np.uint16 or image.shape != (m['height'], m['width']):
                    raise ValueError('invalid native depth')
                hashes[str(fp)], hashes[str(dp)] = file_sha256(fp), file_sha256(dp)
                features.append(feature); depth.append(image)
            fa, fb = features; pixels_a, pixels_b, za, ids = [], [], [], []
            target_ids = []
            for row in proposal['records']:
                i, j = row['source_feature'], row['target_feature']
                if not 0 <= i < len(fa.keypoints) or not 0 <= j < len(fb.keypoints):
                    raise ValueError('invalid feature row')
                u, v = fa.keypoints[i, :2], fb.keypoints[j, :2]
                source_z = sample_depth(depth[0], u, 5000.)
                target_z = sample_depth(depth[1], v, 5000.)
                if source_z is None or target_z is None: raise ValueError('frozen eligibility changed')
                pixels_a.append(u); pixels_b.append(v); za.append(source_z); ids.append(i)
                target_ids.append(j)
            if (len(ids) != proposal['depth_valid_matches'] or len(ids) != expected['records']
                    or len(set(ids)) != len(ids) or len(set(target_ids)) != len(target_ids)
                    or sum(i % 2 == 0 for i in ids) != expected['even_fit_rows']
                    or sum(i % 2 == 1 for i in ids) != expected['odd_check_rows']):
                raise ValueError('frozen row identity/split changed')
            raw = inverse(frames[fb.metadata['source_frame_id']].t_wc) @ frames[fa.metadata['source_frame_id']].t_wc
            cases.append(dict(scene=name, pair=pair, ids=ids,
                              source_pixels=np.asarray(pixels_a).reshape(-1, 2).tolist(),
                              target_pixels=np.asarray(pixels_b).reshape(-1, 2).tolist(),
                              source_depth_m=za, raw_baseline=raw.tolist(),
                              source_timestamp=fa.metadata['rgb_timestamp'],
                              target_timestamp=fb.metadata['rgb_timestamp']))
    return cases, hashes


def check_hashes(hashes):
    for path, expected in hashes.items():
        if file_sha256(Path(path)) != expected: raise ValueError('input changed: '+path)


def fit_worker(args):
    item = json.loads(args.input.read_text())
    source, pixels = np.array(item['source']).reshape(-1, 3), np.array(item['pixels']).reshape(-1, 2)
    initial = check_pose(item['initial']); ids = np.array(item['ids'], dtype=np.int64)
    if args.estimator == LOCAL:
        args.output.mkdir()
        result = refine_pair(source, pixels, item['camera'], initial, model='3d2d')
        write_json(args.output/'fit.json', result)
    else:
        native_fit(args.native_binary, source, pixels, item['camera'], ids, initial,
                   args.seed, args.output)


def arm_process(command, output, log, rss):
    """Independent 4 GiB / 60 s process budget, kill the whole arm on timeout."""
    def budget(): resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    start = time.perf_counter()
    process = subprocess.Popen(['/usr/bin/time', '-f', '%M', '-o', str(rss), *command],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, preexec_fn=budget, start_new_session=True)
    timed_out = False
    try: text, _ = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        timed_out = True; os.killpg(process.pid, signal.SIGKILL); text, _ = process.communicate()
    log.write_text(text)
    result = dict(process_returncode=process.returncode, timed_out=timed_out,
                  process_wall_s=time.perf_counter()-start,
                  peak_rss_kib=None)
    try: result['peak_rss_kib'] = int(rss.read_text().splitlines()[-1])
    except (ValueError, OSError, IndexError): pass
    if process.returncode == 0 and (output/'fit.json').is_file():
        result['fit'] = json.loads((output/'fit.json').read_text())
    else:
        result['fit'] = dict(applied=False, reason='arm timeout' if timed_out else 'arm process failure')
    return result


def run_fit(args):
    if args.output.exists(): raise ValueError('output exists; refuse overwrite')
    protocol = json.loads(args.protocol.read_text()); sources = json.loads(args.sources.read_text())
    repo = Path(__file__).resolve().parents[1]
    for rel, digest in protocol['source_sha256'].items():
        if file_sha256(repo/rel) != digest: raise ValueError('frozen implementation changed: '+rel)
    if (CONFIG['minimum_fit'], CONFIG['pixel_sigma'], CONFIG['huber_delta'], CONFIG['maximum_nfev'], CONFIG['tolerance']) != (6, 2., 2., 100, 1e-10):
        raise ValueError('local estimator config changed')
    if not args.native_binary.is_file(): raise ValueError('native binary missing')
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        if os.environ.get(key) != '1': raise ValueError(key+' must be 1')
    cases, hashes = prepare(protocol, sources)
    hashes[str(args.protocol.resolve())] = file_sha256(args.protocol)
    hashes[str(args.sources.resolve())] = file_sha256(args.sources)
    hashes[str(args.native_binary.resolve())] = file_sha256(args.native_binary)
    hashes[str(repo/'CMakeLists.txt')] = file_sha256(repo/'CMakeLists.txt')
    for base in ['python', 'include', 'src', 'app']:
        for p in (repo/base).rglob('*'):
            if p.is_file() and p.suffix in ['.py', '.h', '.cpp']: hashes[str(p)] = file_sha256(p)
    args.output.mkdir()
    write_json(args.output/'execution-protocol.json', dict(
        parent_protocol_sha256=file_sha256(args.protocol), input_sha256=hashes,
        camera_profiles=protocol['camera_profiles'], estimators=protocol['comparison_plan']['estimators'],
        native_binary_sha256=file_sha256(args.native_binary), truth_read=False,
        planned_cases=len(cases), planned_estimator_arms=len(cases)*2*6,
        resource_limits=protocol['comparison_plan']['resources']))
    (args.output/'inputs').mkdir(); (args.output/'arms').mkdir()
    for case in cases:
        case['arms'] = []; ids = np.array(case['ids'], dtype=np.int64); fit = ids % 2 == 0
        for profile, calibration in protocol['camera_profiles'].items():
            camera = calibration['camera']
            source = np.array([unproject(camera, uv, z) for uv, z in
                               zip(case['source_pixels'], case['source_depth_m'])]).reshape(-1, 3)
            pixels = np.array(case['target_pixels']).reshape(-1, 2)
            prefix = f"{case['scene']}_{case['pair']}_{profile}"
            input_path = args.output/'inputs'/f'{prefix}.json'
            write_json(input_path, dict(camera=camera, ids=ids[fit].tolist(), source=source[fit].tolist(),
                                        pixels=pixels[fit].tolist(), initial=case['raw_baseline']))
            for estimator in [LOCAL, NATIVE]:
                seeds = [None] if estimator == LOCAL else protocol['comparison_plan']['estimators'][NATIVE]['seeds']
                for seed in seeds:
                    name = f'{prefix}_{estimator}_{seed}'
                    output = args.output/'arms'/name
                    command = [sys.executable, '-B', str(Path(__file__).resolve()), 'fit-arm',
                               '--input', str(input_path), '--output', str(output),
                               '--estimator', estimator, '--native-binary', str(args.native_binary)]
                    if seed is not None: command += ['--seed', str(seed)]
                    arm = arm_process(command, output, args.output/'arms'/f'{name}.log',
                                      args.output/'arms'/f'{name}.rss.txt')
                    arm.update(camera_profile=profile, estimator=estimator, seed=seed)
                    arm['fit'].setdefault('t_target_source', case['raw_baseline'])
                    returned = check_pose(np.array(arm['fit']['t_target_source']))
                    if not arm['fit']['applied'] and not np.array_equal(returned, case['raw_baseline']):
                        raise ValueError('failed arm did not preserve exact raw fallback')
                    arm['check_pixels'] = pixel_check(returned, source[~fit], pixels[~fit], camera)
                    case['arms'].append(arm)
        write_json(args.output/'inputs'/f"{case['scene']}_{case['pair']}_completed.json", case)
        print(case['scene'], case['pair'], '12 arms recorded', flush=True)
    check_hashes(hashes)
    write_json(args.output/'fit-only.json', dict(cases=cases, input_sha256=hashes, truth_read=False,
                                               protocol_sha256=file_sha256(args.protocol)))
    seal = {str(p.relative_to(args.output)): file_sha256(p)
            for p in args.output.rglob('*') if p.is_file()}
    write_json(args.output/'fit-seal.json', seal)


def pixel_check(transform, source, pixels, camera):
    if not len(source): return dict(count=0, positive_count=0, median_px=None, p95_px=None)
    p = source@transform[:3, :3].T+transform[:3, 3]
    good = np.isfinite(p).all(1) & (p[:, 2] > 1e-6)
    error = np.linalg.norm(p[good, :2]/p[good, 2, None]*camera[:2]+camera[2:]-pixels[good], axis=1)
    return dict(count=len(source), positive_count=int(good.sum()),
                median_px=float(np.median(error)) if len(error) else None,
                p95_px=float(np.quantile(error, .95)) if len(error) else None)


def evaluate(args):
    """Only this command opens GT, after verifying the entire fit seal."""
    from adaptivefusion.candidate_validation import truth_endpoints, relative_error
    from adaptivefusion.evaluation import read_tum
    if args.output.exists(): raise ValueError('output exists; refuse overwrite')
    seal = json.loads((args.study/'fit-seal.json').read_text())
    for rel, digest in seal.items():
        if not (args.study/rel).resolve().is_relative_to(args.study.resolve()):
            raise ValueError('unsafe fit seal path')
        if file_sha256(args.study/rel) != digest: raise ValueError('fit seal mismatch: '+rel)
    frozen = json.loads((args.study/'fit-only.json').read_text()); check_hashes(frozen['input_sha256'])
    sources = json.loads(args.sources.read_text()); truth = {}; unavailable = {}
    for scene in {c['scene'] for c in frozen['cases']}:
        try:
            path = Path(sources[scene]['truth']); truth[scene] = read_tum(path)
        except (OSError, ValueError, KeyError) as error: unavailable[scene] = str(error)
    rows = []
    for case in frozen['cases']:
        endpoints = truth_endpoints(*truth[case['scene']], case['source_timestamp'], case['target_timestamp']) if case['scene'] in truth else {}
        for mode in ['nearest', 'interpolated']:
            poses = endpoints.get(mode)
            raw = relative_error(np.array(case['raw_baseline']), *poses) if poses is not None else None
            for arm in case['arms']:
                value = relative_error(np.array(arm['fit']['t_target_source']), *poses) if poses is not None else None
                rows.append(dict(scene=case['scene'], pair=case['pair'], camera_profile=arm['camera_profile'],
                                 estimator=arm['estimator'], seed=arm['seed'], truth_mode=mode,
                                 applied=arm['fit']['applied'], reason=arm['fit']['reason'],
                                 process_returncode=arm['process_returncode'], timed_out=arm['timed_out'],
                                 native_process_returncode=arm['fit'].get('process_returncode'),
                                 process_wall_s=arm['process_wall_s'], peak_rss_kib=arm['peak_rss_kib'],
                                 solver_wall_s=arm['fit'].get('solver_wall_s', arm['fit'].get('wall_s')),
                                 translation_m=value['translation_m'] if value else None,
                                 rotation_deg=value['rotation_deg'] if value else None,
                                 gt_fidelity_pass=value['gt_fidelity_pass'] if value else None,
                                 raw_translation_m=raw['translation_m'] if raw else None,
                                 raw_rotation_deg=raw['rotation_deg'] if raw else None,
                                 raw_gt_fidelity_pass=raw['gt_fidelity_pass'] if raw else None,
                                 raw_accurate_became_inaccurate=bool(raw['gt_fidelity_pass'] and not value['gt_fidelity_pass']) if raw and value else None,
                                 check_count=arm['check_pixels']['count'],
                                 check_positive_count=arm['check_pixels']['positive_count'],
                                 check_median_px=arm['check_pixels']['median_px']))
    args.output.mkdir()
    write_json(args.output/'report.json', dict(rows=rows, unavailable_truth=unavailable,
                                              fit_seal_sha256=file_sha256(args.study/'fit-seal.json'),
                                              truth_sha256={s:file_sha256(Path(sources[s]['truth'])) for s in truth},
                                              scope='reused selected pairs only; no BA, trajectory or online claim'))
    with (args.output/'results.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    for rel, digest in seal.items():
        if file_sha256(args.study/rel) != digest: raise ValueError('fit changed during evaluation')
    print(json.dumps(dict(rows=len(rows), unavailable_truth=unavailable)))


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit')
    for key in ['protocol', 'sources', 'native-binary', 'output']: fit.add_argument('--'+key, type=Path, required=True)
    worker = commands.add_parser('fit-arm')
    for key in ['input', 'native-binary', 'output']: worker.add_argument('--'+key, type=Path, required=True)
    worker.add_argument('--estimator', choices=[LOCAL, NATIVE], required=True); worker.add_argument('--seed', type=int, default=17)
    ev = commands.add_parser('evaluate')
    for key in ['study', 'sources', 'output']: ev.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    { 'fit': run_fit, 'fit-arm': fit_worker, 'evaluate': evaluate }[args.command](args)


if __name__ == '__main__': main()
