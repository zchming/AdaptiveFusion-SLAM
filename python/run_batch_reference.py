#!/usr/bin/env python3
"""Offline full-cache BA diagnostic. Larger resource profile, not an online backend."""
import argparse
import json
from pathlib import Path
import time
from adaptivefusion.state import write_snapshot,file_sha256,write_sparse_ply
from adaptivefusion.trajectory import load_frames,write_tum
from adaptivefusion.offline import merge_cached_problem,full_history,component_history,split_landmark_holdout,heldout_geometry
from adaptivefusion.local_ba import solve,Problem


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--time-limit-s',type=float,default=180.)
    parser.add_argument('--holdout-modulus',type=int,default=0,help='0: fit all; >=2: exclude whole landmark IDs divisible by modulus')
    parser.add_argument('--component-policy',choices=['all','anchor'],default='all',
                        help='anchor: optimize only the component containing fixed keyframe 0')
    parser.add_argument('--observability-check',choices=['structural','bridge-audit'],default='structural',
                        help='optional full-batch separator/collinearity rejection; passing is not a rank certificate')
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists')
    marker=json.loads((args.cache/'complete.json').read_text())
    if marker.get('risk_enabled') or marker.get('online_writeback'):parser.error('not baseline offline cache')
    frames=load_frames(args.cache/'frames.txt')
    if len(frames)!=marker.get('processed_frames'):parser.error('incomplete frame stream')
    paths=sorted(args.cache.glob('*.before.snapshot'),key=lambda p:int(p.name.split('.')[0]))
    args.output.mkdir();start=time.perf_counter()
    s=merge_cached_problem(paths,frames)
    original=s
    held=set()
    if args.holdout_modulus:s,held=split_landmark_holdout(s,args.holdout_modulus)
    write_snapshot(args.output/'input.snapshot',s)
    p=Problem(s,component_policy=args.component_policy)
    protocol={'mode':'full-cache offline diagnostic','keyframes':len(s.keyframes),
              'points':len(s.points),'observations':len(s.observations),'variables':len(p.x0),
              'max_iterations':s.config.maximum_iterations,'scope':'same supplied cache, no new matching',
              'initialization':'original raw poses and original landmark birth positions',
              'truth_used':False,'online':False,'time_limit_s':args.time_limit_s}
    protocol['holdout_modulus']=args.holdout_modulus
    protocol['component_policy']=args.component_policy
    protocol['observability_check']=args.observability_check
    protocol['active_keyframe_ids']=p.kids
    protocol['active_points']=len(p.pids)
    protocol['active_observations']=len(p.obs)
    protocol['map_scope']='selected shared landmarks optimized; excluded landmarks retain input coordinates'
    protocol['heldout_point_ids']=sorted(held)
    protocol['python_source_sha256']={str(p.relative_to(Path(__file__).parent)):file_sha256(p)
                                     for p in Path(__file__).parent.rglob('*.py')}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps({k:v for k,v in protocol.items() if k not in ['heldout_point_ids','python_source_sha256']}),flush=True)
    fit=solve(s,method='schur-lm',jacobian='analytic',termination_profile='ceres-cost',
              acceptance_policy='usable',time_limit_s=args.time_limit_s,
              component_policy=args.component_policy,observability_check=args.observability_check)
    write_snapshot(args.output/'output.snapshot',fit.snapshot)
    write_sparse_ply(args.output/'shared_map.ply',fit.snapshot)
    poses={k.id:k.t_wc for k in fit.snapshot.keyframes}
    if args.component_policy=='anchor':
        history=(component_history(frames,s,poses,p.kids) if fit.accepted
                 else [(f.timestamp,f.t_wc.copy()) for f in frames if f.valid])
    else:
        history=full_history(frames,s,poses)
    write_tum(args.output/'corrected.tum',history)
    partial=len(p.kids)<len(s.keyframes)
    status=('accepted_partial' if partial else 'accepted') if fit.accepted else 'rejected'
    report={**fit.report(),'status':status,'partial_scope':partial,
            'total_wall_s':time.perf_counter()-start}
    if held:
        report['heldout_before']=heldout_geometry(original,{k.id:k.t_wc for k in original.keyframes},held)
        report['heldout_after']=heldout_geometry(original,poses,held)
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return 0 if fit.accepted else 2


if __name__=='__main__':raise SystemExit(main())
