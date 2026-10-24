#!/usr/bin/env python3
"""Read-only short-window duplicate-landmark audit; never merges or optimizes."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import time
import cv2
from audit_bridge_candidates import associations, select_cuts
from adaptivefusion.landmark_identity import CONFIG, audit_pair, audit_cycles, identity_evidence
from adaptivefusion.state import load_snapshot, file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['snapshot','dataset','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists; refuse overwrite')
    start=time.perf_counter();before=file_sha256(args.snapshot);s=load_snapshot(args.snapshot)
    associated=associations(args.dataset);cuts=select_cuts(s);windows={}
    for cut,_ in cuts:
        windows[cut]=list(range(max(0,cut-CONFIG['window_each_side']),min(len(s.keyframes),cut+CONFIG['window_each_side'])))
    requested=sorted({pair for kids in windows.values() for pair in combinations(kids,2)})
    args.output.mkdir(parents=True);images={};evidence={};reports={}
    protocol={'config':CONFIG,'selection':'same cut selection as prior audit; two keyframes on each side, all unordered pairs, deduplicated across windows',
              'snapshot_sha256':before,'cuts':cuts,'windows':windows,'requested_pairs':requested,
              'scope':'LK reciprocal feature identity, native RGB-D and split rigid verification, triangle consistency; no pose prior, truth, merge, BA or online feedback',
              'opencv_version':cv2.__version__,'source_sha256':{str(p.relative_to(Path(__file__).parent)):file_sha256(p) for p in [Path(__file__),Path(__file__).parent/'audit_bridge_candidates.py',Path(__file__).parent/'adaptivefusion/landmark_identity.py',Path(__file__).parent/'adaptivefusion/bridge_candidates.py']}}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for kid in sorted({k for pair in requested for k in pair}):
        stamp=s.keyframes[kid].timestamp
        if stamp not in associated:raise ValueError('keyframe timestamp absent from associated stream')
        rgb,depth=associated[stamp]
        for p in [rgb,depth]:evidence[str(p.relative_to(args.dataset))]=file_sha256(p)
        gray=cv2.imread(str(rgb),cv2.IMREAD_GRAYSCALE);native=cv2.imread(str(depth),cv2.IMREAD_UNCHANGED)
        if gray is None or native is None:raise ValueError('unreadable image')
        images[kid]=(gray,native)
    for a,b in requested:
        ga,da=images[a];gb,db=images[b]
        reports[a,b]=audit_pair(s,a,b,ga,gb,da,db)
        (args.output/f'{a}-{b}.json').write_text(json.dumps(reports[a,b],indent=2)+'\n')
    summaries=[]
    for cut,count in cuts:
        kids=windows[cut];selected=[reports[pair] for pair in combinations(kids,2)]
        cycles=audit_cycles(selected,kids);qualified={tuple(p) for p in cycles['qualified_pairs']}
        supported=[p for p in selected if p['source_keyframe']<cut<=p['target_keyframe']
                   and (p['source_keyframe'],p['target_keyframe']) in qualified]
        identity=identity_evidence(s,supported,kids)
        summaries.append({'cut':cut,'original_crossing_landmarks':count,'keyframes':kids,
                          'pair_count':len(selected),'accepted_rigid_pairs':sum(p['verification']['accepted'] for p in selected),
                          'cycles':cycles,'identity':identity})
    assert file_sha256(args.snapshot)==before
    summary={'windows':summaries,'unique_pairs':len(requested),'accepted_rigid_pairs':sum(p['verification']['accepted'] for p in reports.values()),
             'images_sha256':evidence,'snapshot_unchanged':True,'map_updated':False,'accuracy_evaluated':False,'online':False,
             'wall_s':time.perf_counter()-start}
    (args.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['windows','images_sha256']}))
    return 0


if __name__=='__main__':raise SystemExit(main())
