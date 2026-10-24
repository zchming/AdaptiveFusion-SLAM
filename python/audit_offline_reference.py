#!/usr/bin/env python3
"""Audit full-cache offline BA output consistency. No optimization or GT input."""
import argparse
import json
from pathlib import Path
import numpy as np
from adaptivefusion.evaluation import read_tum
from adaptivefusion.reference_audit import audit_geometry, audit_history
from adaptivefusion.state import load_snapshot, file_sha256
from adaptivefusion.trajectory import load_frames


def read_shared_map(path, snapshot):
    lines = path.read_text().splitlines()
    expected = ['ply', 'format ascii 1.0', 'comment units metres; optimized shared landmarks',
                f'element vertex {len(snapshot.points)}', 'property double x',
                'property double y', 'property double z', 'end_header']
    if lines[:8] != expected or len(lines) != 8+len(snapshot.points):
        raise ValueError('shared map schema/count mismatch')
    points = np.array([list(map(float, line.split())) for line in lines[8:]])
    if points.shape != (len(snapshot.points), 3) or not np.array_equal(points, [p.xyz for p in snapshot.points.values()]):
        raise ValueError('shared map differs from exported state')
    return dict(points=len(points), units='metres', scope='shared landmark subset; excluded points may be unoptimized, not a dense reconstruction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'run', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('output exists; refuse overwrite')
    args.output.mkdir(parents=True)
    report = dict(status='failed', scope='output consistency relative to supplied snapshots/frame cache; not observation identity or pose accuracy',
                  ground_truth_used=False, optimization_run=False, online=False, checks={}, input_sha256={})
    required = {'frames': args.cache/'frames.txt', 'cache_completion': args.cache/'complete.json',
                **{name: args.run/name for name in ['input.snapshot', 'output.snapshot', 'protocol.json', 'report.json', 'corrected.tum', 'shared_map.ply']}}
    try:
        report['input_sha256'] = {name: file_sha256(p) for name, p in required.items()}
        marker = json.loads(required['cache_completion'].read_text())
        if marker.get('risk_enabled') is not False or marker.get('online_writeback') is not False:
            raise ValueError('not an explicitly baseline offline cache')
        frames = load_frames(required['frames'])
        if marker.get('processed_frames') != len(frames): raise ValueError('incomplete frame cache')
        before, after = [load_snapshot(required[name]) for name in ['input.snapshot', 'output.snapshot']]
        protocol = json.loads(required['protocol.json'].read_text())
        backend = json.loads(required['report.json'].read_text())
        report['checks']['geometry'] = audit_geometry(before, after, protocol, backend)
        times, poses = read_tum(required['corrected.tum'])
        report['checks']['history'] = audit_history(before, after, frames, times, poses,
            accepted=backend['accepted'], active=set(protocol['active_keyframe_ids']))
        report['checks']['map'] = read_shared_map(required['shared_map.ply'], after)
        if report['input_sha256'] != {name: file_sha256(p) for name, p in required.items()}:
            raise ValueError('input changed during audit')
        report['status'] = 'consistency_pass'
    except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
        report['error'] = str(error)
    report['audit_source_sha256'] = {p.name: file_sha256(p) for p in [Path(__file__), Path(__file__).parent/'adaptivefusion/reference_audit.py']}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'error': report.get('error')}))
    return 0 if report['status'] == 'consistency_pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
