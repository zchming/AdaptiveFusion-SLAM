#!/usr/bin/env python3
"""Evaluate sealed independent-service output; no inference/GT feedback."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from adaptivefusion.evaluation import read_tum,associate
from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames
from evaluate_full_replay import metric

def main():
    p=argparse.ArgumentParser()
    for name in ["protocol","runs","prior-runs","output"]:p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError("refuse overwrite")
    seal=json.loads((a.runs/"fit-seal.json").read_text())
    def check():
        for path,sha in seal.items():
            if file_sha256(a.runs/path)!=sha:raise ValueError("sealed output changed")
    check();plan=json.loads(a.protocol.read_text());sources=json.loads(Path(plan["sources"]).read_text())
    records=[];errors=[];failures=[];gt_hashes={};raw_differences={};cache_differences={}
    for src in sources:
        scene=src["scene"];frames=load_frames(src["baseline_frames"])
        tt,g=read_tum(Path(src["dataset"])/"groundtruth.txt")
        gt_hashes[scene]=file_sha256(Path(src["dataset"])/"groundtruth.txt")
        pairs=associate(tt,np.array([f.timestamp for f in frames]),.02)
        ground={i:g[j] for j,i in pairs};gts={i:tt[j] for j,i in pairs};times={f.id:f.timestamp for f in frames}
        descriptions=dict(baseline=Path(src["baseline_trajectory"]),
            offline_confidence=Path(src["offline_confidence_trajectory"]),
            v35_cached_queued=a.prior_runs/scene/"causal_queued/trajectory.tum",
            service_raw=a.runs/scene/"raw.tum",service_corrected=a.runs/scene/"corrected.tum")
        methods={}
        for name,path in descriptions.items():
            if not path.exists():
                failures.append(dict(scene=scene,method=name,reason="missing trajectory"));continue
            ts,poses=read_tum(path);valid=[f for f in frames if f.valid]
            if len(ts)!=len(valid) or not np.allclose(ts,[f.timestamp for f in valid],rtol=0,atol=1e-6):
                failures.append(dict(scene=scene,method=name,reason="incomplete/changed frame inventory",rows=len(ts)))
                continue
            methods[name]={f.id:pose for f,pose in zip(valid,poses)}
        if "baseline" in methods and "service_raw" in methods:
            raw_differences[scene]=float(max(np.max(np.abs(methods["baseline"][i]-methods["service_raw"][i])) for i in methods["baseline"]))
            cache={f.id:f.t_wc for f in frames if f.valid}
            cache_differences[scene]=float(max(np.max(np.abs(cache[i]-methods["service_raw"][i])) for i in cache))
            if cache_differences[scene]>1e-12:raise ValueError("raw output differs from original full-precision frame cache")
        common=set(ground)
        for poses in methods.values():common&=set(poses)
        for name,poses in methods.items():
            for scope,ids in [("own",sorted(set(poses)&set(ground))),("common",sorted(common))]:
                value,perframe=metric(ids,poses,ground,gts,times,[f.valid for f in frames]);extra={}
                if len(ids)>=3:
                    e=np.array([poses[i] for i in ids]);gg=np.array([ground[i] for i in ids])
                    delta=np.linalg.inv(gg)@((gg[0]@np.linalg.inv(e[0]))@e)
                    trans=np.linalg.norm(delta[:,:3,3],axis=1)*100;angle=np.rad2deg(Rotation.from_matrix(delta[:,:3,:3]).magnitude())
                    extra=dict(anchor_RMSE_cm=float(np.sqrt(np.mean(trans**2))),endpoint_cm=float(trans[-1]),endpoint_rotation_deg=float(angle[-1]))
                    errors.extend(dict(scene=scene,method=name,scope=scope,anchor_translation_cm=float(x),anchor_rotation_deg=float(y),**r)
                        for r,x,y in zip(perframe,trans,angle))
                records.append(dict(scene=scene,method=name,scope=scope,input_frames=len(frames),
                    valid_frames=len(poses),lost_frames=len(frames)-len(poses),coverage=len(poses)/len(frames),
                    common_methods=len(methods),**value,**extra))
    a.output.mkdir()
    payload=dict(records=records,failures=failures,GT_sha256=gt_hashes,raw_pose_matrix_difference=raw_differences,raw_pose_vs_cache_matrix_difference=cache_differences,
        fit_seal_sha256=file_sha256(a.runs/"fit-seal.json"),
        scope=plan["scope"],production_tracking_integration=False,commit_push=False)
    (a.output/"summary.json").write_text(json.dumps(payload,indent=2)+"\n")
    for name,rows in [("summary.csv",records),("frame_errors.csv",errors)]:
        if rows:
            with (a.output/name).open("x",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
    check();print(json.dumps(dict(records=len(records),failures=failures,raw_differences=raw_differences)))
if __name__=="__main__":main()
