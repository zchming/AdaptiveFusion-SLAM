#!/usr/bin/env python3
"""Sealed real-C++ integration evaluation, with separate fault-prefix scopes."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from adaptivefusion.state import file_sha256
from adaptivefusion.evaluation import read_tum,associate
from adaptivefusion.trajectory import load_frames
from evaluate_full_replay import metric

def mapped(path,frames):
    if not path.exists():return None
    ts,poses=read_tum(path);st=np.array([f.timestamp for f in frames]);out={}
    for t,pose in zip(ts,poses):
        j=int(np.argmin(np.abs(st-t)))
        if abs(st[j]-t)>1e-6 or j in out:raise ValueError("trajectory inventory mismatch")
        out[j]=pose
    return out

def evaluate_group(scene,group,frames,descriptions,dataset,records,errors,failures):
    tt,gt=read_tum(Path(dataset)/"groundtruth.txt")
    pairs=associate(tt,np.array([f.timestamp for f in frames]),.02)
    ground={i:gt[j] for j,i in pairs};gts={i:tt[j] for j,i in pairs};times={f.id:f.timestamp for f in frames}
    methods={}
    for method,path in descriptions.items():
        if not path.exists():
            failures.append(dict(scene=scene,group=group,method=method,reason="missing trajectory"));continue
        methods[method]=mapped(path,frames)
    common=set(ground)
    for poses in methods.values():common&=set(poses)
    for method,poses in methods.items():
        valid=[i in poses for i in range(len(frames))]
        for scope,ids in [("own",sorted(set(ground)&set(poses))),("common",sorted(common))]:
            value,perframe=metric(ids,poses,ground,gts,times,valid);extra={}
            if len(ids)>=3:
                e=np.array([poses[i] for i in ids]);g=np.array([ground[i] for i in ids])
                delta=np.linalg.inv(g)@((g[0]@np.linalg.inv(e[0]))@e)
                trans=np.linalg.norm(delta[:,:3,3],axis=1)*100;angle=np.rad2deg(Rotation.from_matrix(delta[:,:3,:3]).magnitude())
                extra=dict(anchor_RMSE_cm=float(np.sqrt(np.mean(trans**2))),endpoint_cm=float(trans[-1]),endpoint_rotation_deg=float(angle[-1]))
                errors.extend(dict(scene=scene,group=group,method=method,scope=scope,anchor_translation_cm=float(x),**r)
                    for r,x in zip(perframe,trans))
            records.append(dict(scene=scene,group=group,method=method,scope=scope,input_frames=len(frames),
                valid_frames=len(poses),lost_or_missing_frames=len(frames)-len(poses),
                coverage=len(poses)/len(frames),common_methods=len(methods),**value,**extra))

def main():
    p=argparse.ArgumentParser()
    for name in ["protocol","runs","output"]:p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError("refuse overwrite")
    seal=json.loads((a.runs/"fit-seal.json").read_text())
    def check():
        for path,sha in seal.items():
            if file_sha256(a.runs/path)!=sha:raise ValueError("sealed integration output changed")
    check();plan=json.loads(a.protocol.read_text());sources=json.loads(Path(plan["sources"]).read_text())
    records=[];errors=[];failures=[];parity=[];runtimes=json.loads((a.runs/"status.json").read_text())
    for src in sources:
        scene=src["scene"];disabled=a.runs/(scene+"-disabled");enabled=a.runs/(scene+"-enabled")
        if not (disabled/"cache/frames.txt").exists():
            failures.append(dict(scene=scene,reason="disabled live producer missing"));continue
        frames=load_frames(disabled/"cache/frames.txt")
        evaluate_group(scene,"full",frames,dict(baseline_live=disabled/"raw.tum",
            enabled_raw=enabled/"raw.tum",enabled_corrected=enabled/"corrected.tum",
            v36_cached_producer=Path(plan["v36_runs"])/scene/"corrected.tum"),src["dataset"],records,errors,failures)
        base=mapped(disabled/"raw.tum",frames)
        old=load_frames(src["baseline_frames"])
        for arm in [x for x in plan["arms"] if x["scene"]==scene]:
            folder=a.runs/(scene+"-"+arm["mode"])
            if not (folder/"cache/frames.txt").exists():continue
            ff=load_frames(folder/"cache/frames.txt");raw=mapped(folder/"raw.tum",ff)
            raw_vs_live=max((float(np.max(np.abs(pose-base[i]))) for i,pose in raw.items()),default=None)
            raw_vs_input=max((float(np.max(np.abs(pose-ff[i].t_wc))) for i,pose in raw.items()),default=None)
            raw_vs_old=max((float(np.max(np.abs(pose-old[i].t_wc))) for i,pose in raw.items()),default=None)
            final=folder/"cache/final.snapshot";reference=disabled/"cache/final.snapshot"
            map_same=file_sha256(final)==file_sha256(reference) if len(ff)==len(frames) and final.exists() and reference.exists() else None
            parity.append(dict(scene=scene,mode=arm["mode"],frames=len(ff),raw_vs_disabled_max_element=raw_vs_live,
                raw_vs_live_input_cache_max_element=raw_vs_input,raw_vs_prior_cached_baseline_max_element=raw_vs_old,
                final_map_snapshot_identical=map_same,
                final_map_snapshot_sha256=file_sha256(final) if final.exists() else None))
            if arm["mode"] in ["disconnect","timeout"]:
                prefix_file=folder/"baseline_prefix.tum"
                # This is a post-seal evaluation-only control projection.
                from adaptivefusion.trajectory import write_tum
                write_tum(a.output.parent/(scene+"-"+arm["mode"]+"-baseline-prefix.tum"),
                          [(f.timestamp,base[f.id]) for f in ff if f.id in base])
                prefix_file=a.output.parent/(scene+"-"+arm["mode"]+"-baseline-prefix.tum")
                evaluate_group(scene,arm["mode"]+"_prefix",ff,dict(baseline_prefix=prefix_file,
                    fault_raw=folder/"raw.tum",fault_corrected=folder/"corrected.tum"),src["dataset"],records,errors,failures)
    a.output.mkdir()
    data=dict(records=records,failures=failures,parity=parity,runtime=runtimes,
        fit_seal_sha256=file_sha256(a.runs/"fit-seal.json"),
        scope="Full live C++ RPY/360 controls; two separately declared 192-frame RPY fault prefixes",
        correction_feedback=False,commit_push=False)
    (a.output/"summary.json").write_text(json.dumps(data,indent=2)+"\n")
    for name,rows in [("summary.csv",records),("frame_errors.csv",errors)]:
        if rows:
            with (a.output/name).open("x",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)
    check();print(json.dumps(dict(records=len(records),failures=failures,parity=parity),indent=2))
if __name__=="__main__":main()
