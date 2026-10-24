#!/usr/bin/env python3
"""V19 common-observation depth-time control. Fitting never opens ground truth."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from PIL import Image
from adaptivefusion.bridge_candidates import sample_depth
from adaptivefusion.depth_time_alignment import sensor_motion, register_source_depth
from adaptivefusion.frontend_features import read_features
from adaptivefusion.geometry import inverse, unproject
from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames
from run_camera_pnp_comparison import write_json, arm_process, check_hashes, NATIVE
from run_orb_observation_controls import support

CAMERA = [517.3, 516.5, 318.6, 255.3]
ARMS = ('unwarped_common', 'time_aligned_common')


def run(args):
    if args.output.exists(): raise ValueError('output exists; refuse overwrite')
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        if os.environ.get(key) != '1': raise ValueError(key+' must be 1')
    sources = json.loads(args.sources.read_text())
    # This command's manifest intentionally has no truth, dataset, or evaluation paths.
    if any(set(p) != {'frames', 'features'} for p in sources.values()):
        raise ValueError('estimated-motion sources must contain only frames and features')
    parent_path = args.baseline_study/'fit-only.json'
    seal_path = args.baseline_study/'fit-seal.json'
    parent = json.loads(parent_path.read_text()); parent_seal = json.loads(seal_path.read_text())
    if file_sha256(parent_path) != parent_seal['fit-only.json']:
        raise ValueError('V19 baseline changed')
    trajectory_path = args.trajectory_study/'fit-only.json'
    if parent['input_sha256'].get(str(trajectory_path.resolve())) != file_sha256(trajectory_path):
        raise ValueError('trajectory provenance differs from V19')
    trajectory_provenance = json.loads(trajectory_path.read_text())['input_sha256']
    binary = args.native_binary.resolve()
    if parent['input_sha256'].get(str(binary)) != file_sha256(binary):
        raise ValueError('native binary differs from V19')
    if len(parent['cases']) != 39: raise ValueError('frozen case inventory changed')
    reused = None
    if args.reuse_study is not None:
        reused = json.loads((args.reuse_study/'fit-only.json').read_text())
        reuse_seal = json.loads((args.reuse_study/'fit-seal.json').read_text())
        if file_sha256(args.reuse_study/'fit-only.json') != reuse_seal['fit-only.json']:
            raise ValueError('cached fit record changed')
        if reused['input_sha256'].get(str(binary)) != file_sha256(binary):
            raise ValueError('cached fit binary differs')
    hashes = {str(p.resolve()):file_sha256(p) for p in [parent_path, seal_path, trajectory_path, args.sources, binary,
        Path(__file__), Path(__file__).parent/'adaptivefusion/depth_time_alignment.py']}
    if reused is not None:
        for p in [args.reuse_study/'fit-only.json', args.reuse_study/'fit-seal.json']:
            hashes[str(p.resolve())] = file_sha256(p)
    frames = {}; metadata = {}
    for scene, paths in sources.items():
        fp = Path(paths['frames']); frames[scene] = load_frames(fp)
        if trajectory_provenance.get(str(fp)) != file_sha256(fp):
            raise ValueError('estimated trajectory differs from V19')
        hashes[str(fp)] = file_sha256(fp)
    args.output.mkdir(); (args.output/'depths').mkdir(); (args.output/'inputs').mkdir(); (args.output/'arms').mkdir()
    write_json(args.output/'execution-protocol.json', dict(
        cases=39, planned_pose_arms=78, camera=CAMERA, arms=ARMS, native_seed=17,
        source='V19 cached LightGlue rows and the original estimated frames.txt history',
        interpolation='linear T_wc translation + SO(3) Slerp; adjacent valid recorded frames only; no extrapolation or lost-frame interpolation; report actual bracket span',
        transform='inverse(T_wc(rgb_time)) @ T_wc(depth_time)',
        registration='metres, forward projection, nearest pixel, nearest-Z buffer, uint16 scale5000; no dilation or filling',
        common_observations='original eligible V19 rows intersect warped sample_depth eligibility; identical IDs/order/target pixels in both arms',
        failures='both arms exact raw fallback if motion unavailable; retain all cases',
        truth_read=False, input_sha256=hashes))
    prepared = {}; cases = []; begin = time.perf_counter()

    def get_metadata(scene, kid):
        key = scene, kid
        if key not in metadata:
            path = Path(sources[scene]['features'])/f'{kid}.features'
            if parent['input_sha256'].get(str(path)) != file_sha256(path):
                raise ValueError('native timestamp metadata differs from V19')
            hashes[str(path)] = file_sha256(path)
            m = read_features(path).metadata
            f = frames[scene][m['source_frame_id']]
            if not f.valid or f.timestamp != m['rgb_timestamp']:
                raise ValueError('feature/estimated RGB frame mismatch')
            metadata[key] = m
        return metadata[key]

    for old in parent['cases']:
        scene = old['scene']; source_id, target_id = map(int, old['pair'].split('-'))
        case = {k:old[k] for k in ['scene', 'pair', 'source_timestamp', 'target_timestamp', 'raw_baseline']}
        m0, m1 = get_metadata(scene, source_id), get_metadata(scene, target_id)
        raw = inverse(frames[scene][m1['source_frame_id']].t_wc)@frames[scene][m0['source_frame_id']].t_wc
        if not np.array_equal(raw, case['raw_baseline']): raise ValueError('baseline history changed')
        if (m0['rgb_timestamp'], m1['rgb_timestamp']) != (case['source_timestamp'], case['target_timestamp']):
            raise ValueError('endpoint timestamp changed')
        baseline_arm = next(a for a in old['arms'] if a['camera_profile']=='rootsift_lightglue')
        rows = baseline_arm['rows']
        case['v19_reference'] = dict(applied=baseline_arm['fit']['applied'], input_rows=len(rows),
                                     t_target_source=baseline_arm['fit']['t_target_source'])
        key = scene, source_id
        if key not in prepared:
            start = time.perf_counter(); path = Path(old['features'][0]['depth_path'])
            if parent['input_sha256'].get(str(path)) != file_sha256(path): raise ValueError('source depth changed')
            hashes[str(path)] = file_sha256(path)
            with Image.open(path) as image: depth = np.array(image)
            if depth.dtype != np.uint16 or depth.shape != (480, 640): raise ValueError('invalid source depth')
            item = dict(depth=depth)
            try:
                motion, evidence = sensor_motion(frames[scene], m0['rgb_timestamp'], m0['depth_timestamp'])
                warped, stats = register_source_depth(depth, CAMERA, motion)
                item.update(warped=warped, motion=motion.tolist(), evidence=evidence, registration=stats)
                np.savez_compressed(args.output/'depths'/f'{scene}_{source_id}.npz', depth=warped)
            except ValueError as error:
                item['failure'] = str(error)
            item['preparation_s'] = time.perf_counter()-start
            prepared[key] = item
        item = prepared[key]
        case['alignment'] = {k:v for k,v in item.items() if k not in ['depth', 'warped']}
        selected, aligned_z = [], []
        for row in rows:
            original = sample_depth(item['depth'], np.array(row['source_pixel']), 5000.)
            if original != row['source_depth_m']: raise ValueError('V19 original depth eligibility changed')
            if 'failure' not in item:
                z = sample_depth(item['warped'], np.array(row['source_pixel']), 5000.)
                if z is not None: selected.append(row); aligned_z.append(z)
        case['common_rows'] = len(selected)
        case['common_source_ids'] = [r['source_feature'] for r in selected]
        case['arms'] = []
        for name in ARMS:
            arm = dict(camera_profile=name, estimator=NATIVE, seed=17)
            if 'failure' in item:
                arm.update(process_returncode=None, timed_out=False, process_wall_s=0., peak_rss_kib=None,
                    fit=dict(applied=False, reason='estimated motion unavailable: '+item['failure'],
                             t_target_source=case['raw_baseline']))
                arm['input_support'] = support([], [], [640, 480])
            else:
                zs = [r['source_depth_m'] for r in selected] if name=='unwarped_common' else aligned_z
                source = [unproject(CAMERA, r['source_pixel'], z).tolist() for r,z in zip(selected, zs)]
                prefix = f'{scene}_{case["pair"]}_{name}'
                inp = args.output/'inputs'/f'{prefix}.json'; output = args.output/'arms'/prefix
                payload = dict(camera=CAMERA, ids=case['common_source_ids'], source=source,
                    pixels=[r['target_pixel'] for r in selected], initial=case['raw_baseline'])
                write_json(inp, payload)
                cached_input = args.reuse_study/'inputs'/f'{prefix}.json' if reused is not None else None
                if cached_input is not None and cached_input.is_file() and json.loads(cached_input.read_text()) == payload:
                    if file_sha256(cached_input) != reuse_seal[str(cached_input.relative_to(args.reuse_study))]:
                        raise ValueError('cached arm input changed')
                    cached_case = next(c for c in reused['cases'] if (c['scene'], c['pair'])==(scene, case['pair']))
                    cached_arm = next(a for a in cached_case['arms'] if a['camera_profile']==name)
                    for key in ['process_returncode', 'timed_out', 'process_wall_s', 'peak_rss_kib', 'fit']:
                        arm[key] = cached_arm[key]
                    arm['reused_from'] = str(args.reuse_study/'arms'/prefix)
                    shutil.copytree(args.reuse_study/'arms'/prefix, output)
                    for suffix in ['.log', '.rss.txt']:
                        shutil.copy2(args.reuse_study/'arms'/f'{prefix}{suffix}', args.output/'arms'/f'{prefix}{suffix}')
                    hashes[str(cached_input)] = file_sha256(cached_input)
                else:
                    arm.update(arm_process([sys.executable, '-B', str(Path(__file__).with_name('run_camera_pnp_comparison.py')),
                        'fit-arm', '--input', str(inp), '--output', str(output), '--native-binary', str(binary),
                        '--estimator', NATIVE, '--seed', '17'], output, args.output/'arms'/f'{prefix}.log',
                        args.output/'arms'/f'{prefix}.rss.txt'))
                arm['input_support'] = support([r['source_pixel'] for r in selected], zs, [640,480])
            arm['fit'].setdefault('t_target_source', case['raw_baseline'])
            if not arm['fit']['applied'] and not np.array_equal(arm['fit']['t_target_source'], case['raw_baseline']):
                raise ValueError('exact raw fallback changed')
            arm['check_pixels'] = dict(count=0, positive_count=0, median_px=None, p95_px=None)
            case['arms'].append(arm)
        cases.append(case)
        write_json(args.output/'inputs'/f'{scene}_{case["pair"]}_completed.json', case)
        print(scene, case['pair'], 'V19/common', len(rows), len(selected),
              [a['fit']['applied'] for a in case['arms']], item.get('failure',''), flush=True)
    check_hashes(hashes)
    write_json(args.output/'fit-only.json', dict(cases=cases, input_sha256=hashes, truth_read=False,
        fit_phase_wall_s=time.perf_counter()-begin, unique_sources=len(prepared),
        source_preparation_s=sum(p['preparation_s'] for p in prepared.values())))
    write_json(args.output/'fit-seal.json', {str(p.relative_to(args.output)):file_sha256(p)
               for p in args.output.rglob('*') if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline-study', 'trajectory-study', 'sources', 'native-binary', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--reuse-study', type=Path)
    run(parser.parse_args())


if __name__ == '__main__': main()
