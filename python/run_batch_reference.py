#!/usr/bin/env python3
"""Offline full-cache BA diagnostic. Larger resource profile, not an online backend."""
import argparse
import json
from pathlib import Path
import time
from adaptivefusion.state import write_snapshot,file_sha256,write_sparse_ply
from adaptivefusion.trajectory import load_frames,write_tum
from adaptivefusion.offline import merge_cached_problem,full_history,split_landmark_holdout,heldout_geometry
from adaptivefusion.local_ba import solve,Problem


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--time-limit-s',type=float,default=180.)
    parser.add_argument('--holdout-modulus',type=int,default=0,help='0: fit all; >=2: exclude whole landmark IDs divisible by modulus')
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
    p=Problem(s)
    protocol={'mode':'full-cache offline diagnostic','keyframes':len(s.keyframes),
              'points':len(s.points),'observations':len(s.observations),'variables':len(p.x0),
              'max_iterations':s.config.maximum_iterations,'scope':'same supplied cache, no new matching',
              'initialization':'original raw poses and original landmark birth positions',
              'truth_used':False,'online':False,'time_limit_s':args.time_limit_s}
    protocol['holdout_modulus']=args.holdout_modulus
    protocol['heldout_point_ids']=sorted(held)
    protocol['python_source_sha256']={str(p.relative_to(Path(__file__).parent)):file_sha256(p)
                                     for p in Path(__file__).parent.rglob('*.py')}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps({k:v for k,v in protocol.items() if k not in ['heldout_point_ids','python_source_sha256']}),flush=True)
    fit=solve(s,method='schur-lm',jacobian='analytic',termination_profile='ceres-cost',
              acceptance_policy='usable',time_limit_s=args.time_limit_s)
    write_snapshot(args.output/'output.snapshot',fit.snapshot)
    write_sparse_ply(args.output/'shared_map.ply',fit.snapshot)
    poses={k.id:k.t_wc for k in fit.snapshot.keyframes}
    write_tum(args.output/'corrected.tum',full_history(frames,s,poses))
    report={**fit.report(),'status':'accepted' if fit.accepted else 'rejected','total_wall_s':time.perf_counter()-start}
    if held:
        report['heldout_before']=heldout_geometry(original,{k.id:k.t_wc for k in original.keyframes},held)
        report['heldout_after']=heldout_geometry(original,poses,held)
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return 0 if fit.accepted else 2


if __name__=='__main__':raise SystemExit(main())
