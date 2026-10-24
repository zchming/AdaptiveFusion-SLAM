#!/usr/bin/env python3
"""Read-only weak-cut correspondence audit. Outputs proposals, never optimized state."""
import argparse
import bisect
import json
import itertools
from pathlib import Path
import time
import cv2
from adaptivefusion.bridge_candidates import CONFIG, propose_pair
from adaptivefusion.state import load_snapshot, file_sha256


def associations(root):
    def index(name):
        rows=[]
        for line in (root/name).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):continue
            stamp,relative=line.split();p=root/relative
            if not p.resolve().is_relative_to(root.resolve()):raise ValueError('index escapes dataset root')
            rows.append((float(stamp),p))
        rows.sort()
        if len({t for t,_ in rows})!=len(rows):raise ValueError('duplicate timestamps')
        return rows
    rgb,depth=index('rgb.txt'),index('depth.txt');times=[t for t,_ in depth];candidates=[]
    for i,(t,_) in enumerate(rgb):
        for j in range(bisect.bisect_left(times,t-.02),bisect.bisect_right(times,t+.02)):
            candidates.append((abs(t-times[j]),i,j))
    candidates.sort()
    # Equal-cost disjoint edges commute under C++'s unspecified tie ordering.
    # Only conflicting, still-eligible edges would make the stream ambiguous.
    ri=set();di=set();out={}
    for _,group in itertools.groupby(candidates,key=lambda row:row[0]):
        eligible=[(i,j) for _,i,j in group if i not in ri and j not in di]
        if (len({i for i,j in eligible})!=len(eligible)
                or len({j for i,j in eligible})!=len(eligible)):
            raise ValueError('ambiguous association tie requires explicit exported association')
        for i,j in eligible:
            ri.add(i);di.add(j);out[rgb[i][0]]=(rgb[i][1],depth[j][1])
    return out


def select_cuts(snapshot):
    ranges={}
    for o in snapshot.observations:
        lo,hi=ranges.get(o.point_id,(o.keyframe_id,o.keyframe_id))
        ranges[o.point_id]=(min(lo,o.keyframe_id),max(hi,o.keyframe_id))
    counts=[sum(lo<i<=hi for lo,hi in ranges.values()) for i in range(1,len(snapshot.keyframes))]
    if not counts:raise ValueError('at least two keyframes required')
    cuts=[i+1 for i,n in enumerate(counts) if n<3]
    if not cuts:cuts=[min(range(1,len(counts)+1),key=lambda i:(counts[i-1],i))]
    if len(cuts)>32:raise ValueError('more than 32 weak cuts; refuse rather than silently truncate')
    return [(i,counts[i-1]) for i in cuts]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ['snapshot','dataset','output']:parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists; refuse overwrite')
    start=time.perf_counter();s=load_snapshot(args.snapshot);pairs=associations(args.dataset)
    cuts=select_cuts(s);args.output.mkdir(parents=True)
    before=file_sha256(args.snapshot);evidence={};results=[]
    protocol={'config':CONFIG,'selection':'all cuts with <3 crossing landmarks; otherwise first minimum-support cut',
              'scope':'bidirectional adjacent-keyframe proposals only; no BA, map mutation, pose prior or truth',
              'snapshot_sha256':before,'source_sha256':{p.name:file_sha256(p) for p in [Path(__file__),Path(__file__).parent/'adaptivefusion/bridge_candidates.py']},
              'opencv_version':cv2.__version__,'cuts':cuts}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for cut,count in cuts:
        for source,target in [(cut-1,cut),(cut,cut-1)]:
            paths=[]
            for kid in [source,target]:
                stamp=s.keyframes[kid].timestamp
                if stamp not in pairs:raise ValueError('keyframe timestamp absent from associated RGB stream')
                paths.extend(pairs[stamp])
            images=[]
            for n,path in enumerate(paths):
                evidence[str(path.relative_to(args.dataset))]=file_sha256(path)
                image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE if n%2==0 else cv2.IMREAD_UNCHANGED)
                if image is None:raise ValueError('unreadable image')
                images.append(image)
            row=propose_pair(s,source,target,images[0],images[2],images[1],images[3])
            row['cut_crossing_landmarks']=count;results.append(row)
            (args.output/f'{source}-{target}.json').write_text(json.dumps(row,indent=2)+'\n')
    assert file_sha256(args.snapshot)==before
    summary={'pairs':results,'images_sha256':evidence,'wall_s':time.perf_counter()-start,
             'snapshot_unchanged':True,'online':False,'map_updated':False,'accuracy_evaluated':False}
    (args.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['pairs','images_sha256']}))
    return 0


if __name__=='__main__':raise SystemExit(main())
