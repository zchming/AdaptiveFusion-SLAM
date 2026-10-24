#!/usr/bin/env python3
"""Compare direct-keyframe and ordinary-image tracks; never merges or optimizes."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import time
import cv2
from audit_bridge_candidates import associations, select_cuts
from adaptivefusion.landmark_identity import CONFIG, audit_pair, audit_cycles, identity_evidence
from adaptivefusion.state import load_snapshot, file_sha256
from adaptivefusion.trajectory import load_frames
from adaptivefusion.ordinary_tracks import chain_endpoints
from adaptivefusion.observation_pool import observation_pool


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['snapshot','frames','dataset','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--observation-cache',type=Path,help='optional original cache: include singleton observations for candidate audit only')
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists; refuse overwrite')
    start=time.perf_counter();before=file_sha256(args.snapshot);s=load_snapshot(args.snapshot)
    frames=load_frames(args.frames);frame_sha=file_sha256(args.frames)
    for k in s.keyframes:
        if k.source_frame_id>=len(frames) or frames[k.source_frame_id].timestamp!=k.timestamp:
            raise ValueError('keyframe/frame-stream timestamp mismatch')
    cuts=select_cuts(s);pool_report=None
    if args.observation_cache:s,pool_report=observation_pool(args.observation_cache,s)
    associated=associations(args.dataset);windows={}
    for cut,_ in cuts:
        windows[cut]=list(range(max(0,cut-CONFIG['window_each_side']),min(len(s.keyframes),cut+CONFIG['window_each_side'])))
    requested=sorted({pair for kids in windows.values() for pair in combinations(kids,2)})
    args.output.mkdir(parents=True);images={};evidence={};reports={}
    protocol={'config':CONFIG,'selection':'same cut selection as prior audit; two keyframes on each side, all unordered pairs, deduplicated across windows',
              'snapshot_sha256':before,'frames_sha256':frame_sha,'cuts':cuts,'windows':windows,'requested_pairs':requested,
              'scope':'paired direct and ordinary-frame transport with identical endpoint/geometry/identity gates; no inference pose, truth, merge, BA or online feedback',
              'opencv_version':cv2.__version__,'source_sha256':{str(p.relative_to(Path(__file__).parent)):file_sha256(p) for p in [Path(__file__),Path(__file__).parent/'audit_bridge_candidates.py',Path(__file__).parent/'adaptivefusion/landmark_identity.py',Path(__file__).parent/'adaptivefusion/bridge_candidates.py',Path(__file__).parent/'adaptivefusion/ordinary_tracks.py',Path(__file__).parent/'adaptivefusion/observation_pool.py']}}
    protocol['observation_pool']='cached_all_including_singletons' if pool_report else 'shared_baseline'
    if pool_report:(args.output/'observation-pool.json').write_text(json.dumps(pool_report,indent=2)+'\n')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    needed_frames=sorted({fid for a,b in requested for fid in range(s.keyframes[a].source_frame_id,s.keyframes[b].source_frame_id+1)})
    for fid in needed_frames:
        stamp=frames[fid].timestamp
        if stamp not in associated:raise ValueError('ordinary timestamp absent from associated stream')
        rgb,_=associated[stamp];evidence[str(rgb.relative_to(args.dataset))]=file_sha256(rgb)
        gray=cv2.imread(str(rgb),cv2.IMREAD_GRAYSCALE)
        if gray is None:raise ValueError('unreadable ordinary image')
        images[fid]=gray
    depths={}
    for kid in sorted({k for pair in requested for k in pair}):
        _,depth=associated[s.keyframes[kid].timestamp]
        evidence[str(depth.relative_to(args.dataset))]=file_sha256(depth)
        native=cv2.imread(str(depth),cv2.IMREAD_UNCHANGED)
        if native is None:raise ValueError('unreadable native depth')
        depths[kid]=native
    reports={'direct':{},'chain':{}};pair_timing={}
    for a,b in requested:
        first=s.keyframes[a].source_frame_id;last=s.keyframes[b].source_frame_id
        sequence=[images[fid] for fid in range(first,last+1)]
        tracking_start=time.perf_counter()
        endpoint_flows,transport=chain_endpoints(s,a,b,sequence)
        pair_timing[f'{a}-{b}']={'chain_transport_s':time.perf_counter()-tracking_start}
        for arm,flow in [('direct',None),('chain',endpoint_flows)]:
            pair_start=time.perf_counter()
            reports[arm][a,b]=audit_pair(s,a,b,sequence[0],sequence[-1],depths[a],depths[b],endpoint_flows=flow)
            pair_timing[f'{a}-{b}'][arm+'_endpoint_audit_s']=time.perf_counter()-pair_start
            if arm=='chain':reports[arm][a,b]['transport']=transport
            directory=args.output/arm;directory.mkdir(exist_ok=True)
            (directory/f'{a}-{b}.json').write_text(json.dumps(reports[arm][a,b],indent=2)+'\n')
    summaries={}
    for arm,arm_reports in reports.items():
        summaries[arm]=[]
        for cut,count in cuts:
            kids=windows[cut];selected=[arm_reports[pair] for pair in combinations(kids,2)]
            cycles=audit_cycles(selected,kids);qualified={tuple(p) for p in cycles['qualified_pairs']}
            supported=[p for p in selected if p['source_keyframe']<cut<=p['target_keyframe']
                       and (p['source_keyframe'],p['target_keyframe']) in qualified]
            identity=identity_evidence(s,supported,kids)
            summaries[arm].append({'cut':cut,'original_crossing_landmarks':count,'keyframes':kids,
                                  'pair_count':len(selected),'accepted_rigid_pairs':sum(p['verification']['accepted'] for p in selected),
                                  'cycles':cycles,'identity':identity})
    assert file_sha256(args.snapshot)==before and file_sha256(args.frames)==frame_sha
    summary={'windows':summaries,'unique_pairs':len(requested),'accepted_rigid_pairs':{arm:sum(p['verification']['accepted'] for p in rows.values()) for arm,rows in reports.items()},
             'images_sha256':evidence,'pair_timing':pair_timing,'snapshot_unchanged':True,'map_updated':False,'accuracy_evaluated':False,'online':False,
             'wall_s':time.perf_counter()-start}
    (args.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['windows','images_sha256','pair_timing']}))
    return 0


if __name__=='__main__':raise SystemExit(main())
