#!/usr/bin/env python3
"""Replay one frozen BA window; optionally correct its anchored history."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import scipy
from adaptivefusion.state import load_snapshot, write_snapshot, file_sha256
from adaptivefusion.local_ba import Problem, solve, reanchor_singletons
from adaptivefusion.geometry import to_cw_vector
from adaptivefusion.trajectory import load_frames, correct_history, write_tum


def candidate_metrics(problem, candidate):
    original = problem.s
    candidate.validate()
    if (candidate.state_version != original.state_version or
            candidate.source_revision != original.source_revision or
            candidate.config != original.config or
            not np.array_equal(candidate.camera, original.camera) or
            set(candidate.points) != set(original.points) or
            set(candidate.observations) != set(original.observations) or
            len(candidate.keyframes) != len(original.keyframes)):
        raise ValueError('Ceres output does not describe the same problem')
    for a, b in zip(original.keyframes, candidate.keyframes):
        if (a.id, a.source_frame_id, a.timestamp, a.width, a.height) != (
                b.id, b.source_frame_id, b.timestamp, b.width, b.height):
            raise ValueError('Ceres keyframe metadata mismatch')
        if a.id <= problem.begin and not np.allclose(a.t_wc, b.t_wc, atol=1e-12, rtol=0):
            raise ValueError('Ceres changed fixed or external pose')
    for pid in original.points:
        if original.points[pid].host != candidate.points[pid].host:
            raise ValueError('Ceres changed point ownership')
        if pid not in problem.pi and not np.array_equal(original.points[pid].xyz, candidate.points[pid].xyz):
            raise ValueError('Ceres changed excluded point')
    vector = np.r_[*[to_cw_vector(candidate.keyframes[k].t_wc) for k in problem.free],
                   *[candidate.points[p].xyz for p in problem.pids]]
    return problem.metrics(vector)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new directory; refuses overwrite')
    parser.add_argument('--frames', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ceres-executable', type=Path)
    group.add_argument('--ceres-snapshot', type=Path, help='existing matching Ceres output')
    parser.add_argument('--max-nfev', type=int, default=200)
    parser.add_argument('--time-limit-s', type=float, default=30.)
    parser.add_argument('--jacobian', choices=['sparse-fd','analytic'], default='sparse-fd')
    parser.add_argument('--lsmr-tolerance', type=float)
    parser.add_argument('--termination-profile',choices=['strict','ceres-cost'],default='strict')
    parser.add_argument('--method',choices=['scipy-trf','schur-lm'],default='scipy-trf')
    parser.add_argument('--acceptance-policy',choices=['converged','usable'],default='converged')
    parser.add_argument('--repair-singletons', action='store_true', help='separate map consistency ablation')
    args = parser.parse_args()
    if args.output.exists(): parser.error('output already exists')
    read_start = time.perf_counter()
    s = load_snapshot(args.snapshot)
    frames = load_frames(args.frames) if args.frames else None
    input_read_ms = (time.perf_counter() - read_start) * 1000
    args.output.mkdir()  # Atomic reservation against another run.
    report = {'schema_version': 1, 'mode': 'independent_frozen_window',
              'online_writeback': False, 'input_sha256': file_sha256(args.snapshot),
              'source_revision': s.source_revision, 'state_version': s.state_version,
              'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__},
              'input_read_ms': input_read_ms,
              'python_max_nfev': args.max_nfev, 'ceres_maximum_iterations': s.config.maximum_iterations,
              'python_time_limit_s': args.time_limit_s,
              'thread_environment': {name: os.environ.get(name) for name in
                                     ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
              'rank_check': 'dense for <=200 variables; structural only above',
              'scale_gauge': 'fixed window first pose plus active metric depth',
              'window_boundary': 'outside observations excluded, matching existing Ceres',
              'trajectory_scope': 'only anchors in this snapshot window', 'status': 'started'}
    exit_code = 0
    try:
        result = solve(s, max_nfev=args.max_nfev, time_limit_s=args.time_limit_s,
                       jacobian=args.jacobian,lsmr_tolerance=args.lsmr_tolerance,
                       termination_profile=args.termination_profile,method=args.method,
                       acceptance_policy=args.acceptance_policy)
        report['python'] = result.report()
        write_snapshot(args.output / 'python.snapshot', result.snapshot)
        problem = Problem(s)
        ceres = None
        if args.ceres_executable:
            report['ceres_executable_sha256'] = file_sha256(args.ceres_executable)
            start = time.perf_counter()
            command = [str(args.ceres_executable.resolve()), str(args.snapshot.resolve()),
                       str((args.output / 'ceres').resolve())]
            with (args.output / 'ceres.log').open('x') as log:
                run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=args.time_limit_s + 5)
            report['ceres_process'] = {'returncode': run.returncode,
                                      'exchange_and_process_ms': (time.perf_counter()-start)*1000}
            if run.returncode != 0: raise ValueError('Ceres replay failed; see ceres.log')
            report['ceres_solver'] = json.loads((args.output / 'ceres' / 'result.json').read_text())
            ceres = load_snapshot(args.output / 'ceres' / 'optimized.snapshot')
        elif args.ceres_snapshot:
            report['ceres_output_sha256'] = file_sha256(args.ceres_snapshot)
            ceres = load_snapshot(args.ceres_snapshot)
        else:
            report['ceres_comparison'] = 'not run: no Ceres executable/output supplied'
        if ceres is not None:
            report['ceres_metrics'] = candidate_metrics(problem, ceres)
            report['backend_difference'] = {
                'python_state':'accepted_solution' if result.accepted else 'original_after_rollback',
                'max_point_distance_m': max((float(np.linalg.norm(ceres.points[p].xyz-result.snapshot.points[p].xyz))
                                              for p in problem.pids), default=0.),
                'max_pose_translation_m': max((float(np.linalg.norm(ceres.keyframes[k].t_wc[:3, 3]
                                                   - result.snapshot.keyframes[k].t_wc[:3, 3]))
                                               for k in problem.kids), default=0.)}
        if args.repair_singletons and result.accepted:
            repaired, changed = reanchor_singletons(s, result.snapshot)
            write_snapshot(args.output / 'python_singletons.snapshot', repaired)
            report['singleton_ablation'] = {'changed_point_ids': changed,
                'limits': 'does not enforce multi-view boundary observations'}
        if frames is not None:
            write_tum(args.output / 'original.tum', [(f.timestamp, f.t_wc) for f in frames if f.valid])
            # A rejected update must reproduce the entire original trajectory.
            corrected = {k: result.snapshot.keyframes[k].t_wc for k in problem.kids} if result.accepted else {}
            write_tum(args.output / 'python_corrected.tum', correct_history(frames, s.keyframes, corrected))
            if ceres is not None:
                write_tum(args.output / 'ceres_corrected.tum', correct_history(
                    frames, s.keyframes, {k: ceres.keyframes[k].t_wc for k in problem.kids}))
        report['status'] = 'complete' if result.accepted else 'rejected'
        exit_code = 0 if result.accepted else 2
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        report['status'], report['error'] = 'failed', str(error)
        exit_code = 1
    finally:
        with (args.output / 'report.json').open('x') as out:
            json.dump(report, out, indent=2, allow_nan=False)
    print(json.dumps({'status': report['status'], 'report': str(args.output / 'report.json')}))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
