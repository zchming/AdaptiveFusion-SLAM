#!/usr/bin/env python3
"""Evaluate sealed causal replay against original baseline/offline confidence."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames
from adaptivefusion.evaluation import read_tum,associate
from evaluate_full_replay import metric

def main():
    p=argparse.ArgumentParser()
    for name in ["sources","runs","output"]:p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError("refuse overwrite")
    seal=json.loads((a.runs/"fit-seal.json").read_text())
    def check():
        for path,sha in seal.items():
            if file_sha256(a.runs/path)!=sha:raise ValueError("sealed replay changed")
    check()
    sources=json.loads(a.sources.read_text());records=[];errors=[];failures=[];gt_hashes={}
    for src in sources:
        scene=src["scene"]
        frames=load_frames(src["baseline_frames"])
        times={f.id:f.timestamp for f in frames}
        gt=Path(src["dataset"])/"groundtruth.txt";tt,g=read_tum(gt);gt_hashes[scene]=file_sha256(gt)
        pairs=associate(tt,np.array([f.timestamp for f in frames]),.02)
        ground={i:g[j] for j,i in pairs};gts={i:tt[j] for j,i in pairs}
        descriptions=dict(baseline=Path(src["baseline_trajectory"]),
            offline_confidence=Path(src["offline_confidence_trajectory"]),
            causal_instant=a.runs/scene/"causal_instant/trajectory.tum",
            causal_queued=a.runs/scene/"causal_queued/trajectory.tum")
        methods={}
        for name,path in descriptions.items():
            if not path.exists():
                failures.append(dict(scene=scene,method=name,reason="missing trajectory"));continue
            ts,poses=read_tum(path);valid=[f for f in frames if f.valid]
            if len(ts)!=len(valid) or not np.allclose(ts,[f.timestamp for f in valid],atol=1e-6,rtol=0):raise ValueError("frame inventory changed")
            methods[name]={f.id:pose for f,pose in zip(valid,poses)}
        common=set(ground)
        for poses in methods.values():common&=set(poses)
        for name,poses in methods.items():
            for scope,ids in [("own",sorted(set(ground)&set(poses))),("common",sorted(common))]:
                value,perframe=metric(ids,poses,ground,gts,times,[f.valid for f in frames]);extra={}
                if len(ids)>=3:
                    e=np.array([poses[i] for i in ids]);gg=np.array([ground[i] for i in ids])
                    delta=np.linalg.inv(gg)@((gg[0]@np.linalg.inv(e[0]))@e)
                    trans=np.linalg.norm(delta[:,:3,3],axis=1)*100
                    angle=np.rad2deg(Rotation.from_matrix(delta[:,:3,:3]).magnitude())
                    extra=dict(anchor_RMSE_cm=float(np.sqrt(np.mean(trans**2))),endpoint_cm=float(trans[-1]),
                        endpoint_rotation_deg=float(angle[-1]),last_evaluated_frame=ids[-1])
                    errors.extend(dict(scene=scene,method=name,scope=scope,
                        anchor_translation_cm=float(x),anchor_rotation_deg=float(y),**r)
                        for r,x,y in zip(perframe,trans,angle))
                # Consecutive pose motion includes ordinary camera motion; report separately from correction jumps.
                vv=np.array([poses[i] for i in sorted(poses)])
                relative=np.linalg.inv(vv[:-1])@vv[1:]
                extra["adjacent_motion_translation_P95_cm"]=float(np.quantile(np.linalg.norm(relative[:,:3,3],axis=1)*100,.95))
                extra["adjacent_motion_rotation_P95_deg"]=float(np.quantile(np.rad2deg(Rotation.from_matrix(relative[:,:3,:3]).magnitude()),.95))
                records.append(dict(scene=scene,split=src["split"],method=name,scope=scope,
                    input_frames=len(frames),valid_frames=len(poses),lost_frames=len(frames)-len(poses),
                    coverage=len(poses)/len(frames),common_methods=len(methods),**value,**extra))
    a.output.mkdir()
    payload=dict(records=records,failures=failures,GT_sha256=gt_hashes,
        fit_seal_sha256=file_sha256(a.runs/"fit-seal.json"),
        scope="Five already evaluated recordings; cached source-time causal simulation, not fresh blind evaluation or online tracking")
    (a.output/"summary.json").write_text(json.dumps(payload,indent=2)+"\n")
    for name,rows in [("summary.csv",records),("frame_errors.csv",errors)]:
        if rows:
            with (a.output/name).open("x",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
                w.writeheader();w.writerows(rows)
    check()
    print(json.dumps({"rows":len(records),"failures":failures}),flush=True)
if __name__=="__main__":main()
