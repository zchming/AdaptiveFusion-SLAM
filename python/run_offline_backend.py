#!/usr/bin/env python3
"""Replay all frozen windows with explicit state and solver controls. No truth input."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from adaptivefusion.offline import CachedState,full_history
from adaptivefusion.state import load_snapshot,write_snapshot,file_sha256
from adaptivefusion.local_ba import solve,Problem
from adaptivefusion.trajectory import load_frames,write_tum
from run_geometry_replay import candidate_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=['python-schur','ceres'],required=True)
    parser.add_argument('--state-mode',choices=['consistent','independent'],required=True)
    parser.add_argument('--ceres-executable',type=Path)
    parser.add_argument('--no-singleton-repair',action='store_true')
    parser.add_argument('--step-history',action='store_true')
    parser.add_argument('--boundary',choices=['window','fixed-external'],default='window')
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists')
    marker=json.loads((args.cache/'complete.json').read_text())
    if marker.get('risk_enabled') or marker.get('online_writeback'):parser.error('cache is not baseline offline export')
    if args.backend=='ceres' and args.ceres_executable is None:parser.error('Ceres executable required')
    if args.backend=='ceres' and args.boundary!='window':parser.error('Ceres parity backend supports the original window boundary only')
    frames=load_frames(args.cache/'frames.txt')
    paths=sorted(args.cache.glob('*.before.snapshot'),key=lambda p:int(p.name.split('.')[0]))
    if not paths:parser.error('no windows')
    if marker.get('processed_frames')!=len(frames):parser.error('incomplete frame stream')
    args.output.mkdir()
    state=CachedState(frames);events=[];start=time.perf_counter()
    report={'mode':'fixed-observation offline replay','backend':args.backend,'state_mode':args.state_mode,
            'online_tracking_feedback':False,'singleton_repair':args.state_mode=='consistent' and not args.no_singleton_repair,
            'history_interpolation':not args.step_history,'boundary':args.boundary,'frames_sha256':file_sha256(args.cache/'frames.txt'),
            'windows':[int(p.name.split('.')[0]) for p in paths],
            'solver':{'method':'schur-lm' if args.backend=='python-schur' else 'Ceres DENSE_SCHUR',
                      'max_iterations':'unchanged snapshot config','acceptance':'finite usable decreasing cost',
                      'termination_profile':'ceres-cost'},'status':'running'}
    report['python_source_sha256']={str(p.relative_to(Path(__file__).parent)):file_sha256(p)
                                   for p in Path(__file__).parent.rglob('*.py')}
    (args.output/'protocol.json').write_text(json.dumps(report,indent=2))
    if args.ceres_executable:report['ceres_executable_sha256']=file_sha256(args.ceres_executable)
    try:
        for path in paths:
            window_start=time.perf_counter();snapshot=load_snapshot(path)
            before=state.prepare(snapshot) if args.state_mode=='consistent' else snapshot
            out=args.output/str(snapshot.state_version);out.mkdir()
            write_snapshot(out/'input.snapshot',before)
            if args.backend=='python-schur':
                fit=solve(before,method='schur-lm',jacobian='analytic',termination_profile='ceres-cost',acceptance_policy='usable',boundary=args.boundary)
                after=fit.snapshot;accepted=fit.accepted;details=fit.report()
            else:
                with (out/'ceres.log').open('x') as log:
                    process=subprocess.run([str(args.ceres_executable.resolve()),str((out/'input.snapshot').resolve()),
                                           str((out/'ceres').resolve())],stdout=log,stderr=subprocess.STDOUT,timeout=35)
                if process.returncode==0:
                    after=load_snapshot(out/'ceres/optimized.snapshot')
                    metrics=candidate_metrics(Problem(before),after)
                    accepted=True;details=json.loads((out/'ceres/result.json').read_text())
                    details['objective']=metrics
                else:
                    after=before.clone();accepted=False;details={'reason':f'Ceres exit {process.returncode}'}
            # Always preserve inserted/raw-predicted state; rejection must not
            # erase an already processed frame. Optimization itself is atomic.
            if args.state_mode=='consistent':after=state.commit(before,after,repair=not args.no_singleton_repair)
            else:
                for k in after.keyframes[max(0,len(after.keyframes)-after.config.window_size):]:state.poses[k.id]=k.t_wc.copy()
            write_snapshot(out/'output.snapshot',after)
            event={'frame_id':snapshot.state_version,'accepted':accepted,'details':details,
                   'input_sha256':file_sha256(out/'input.snapshot'),
                   'window_wall_ms':(time.perf_counter()-window_start)*1000}
            events.append(event)
            (out/'report.json').write_text(json.dumps(event,indent=2))
            if len(events)%10==0:print(json.dumps({'processed':len(events),'accepted':sum(e['accepted'] for e in events)}),flush=True)
        template=load_snapshot(args.cache/'final.snapshot')
        write_tum(args.output/'corrected.tum',full_history(frames,template,state.poses,interpolate=not args.step_history))
        report['status']='complete'
    except (ValueError,OSError,subprocess.TimeoutExpired) as error:
        report['status']='failed';report['error']=str(error)
    finally:
        report.update({'events':events,'accepted':sum(e['accepted'] for e in events),
                       'rejected':sum(not e['accepted'] for e in events),'wall_seconds':time.perf_counter()-start})
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'accepted':report['accepted'],'rejected':report['rejected'],
                      'wall_seconds':report['wall_seconds']}),flush=True)
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
