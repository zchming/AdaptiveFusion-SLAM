#!/usr/bin/env python3
"""Evaluate sealed independent live integration and clearly named offline references."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames
from evaluate_live_cpp_auxiliary import mapped,evaluate_group

def main():
    parser=argparse.ArgumentParser()
    for name in ["protocol","runs","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError("refuse overwrite")
    plan=json.loads(args.protocol.read_text());sources=json.loads(Path(plan["sources"]).read_text())
    seal=json.loads((args.runs/"fit-seal.json").read_text())
    def check():
        for path,sha in seal.items():
            if file_sha256(args.runs/path)!=sha:raise ValueError("sealed run changed: "+path)
    check();records=[];errors=[];failures=[];parity=[]
    for source in sources:
        scene=source["scene"];disabled=args.runs/(scene+"-disabled");enabled=args.runs/(scene+"-enabled")
        if not (disabled/"cache/frames.txt").exists():
            failures.append(dict(scene=scene,reason="missing baseline frame inventory"));continue
        frames=load_frames(disabled/"cache/frames.txt")
        descriptions=dict(baseline_live=disabled/"raw.tum",enabled_raw=enabled/"raw.tum",
            enabled_corrected=enabled/"corrected.tum",
            historical_fixed_offline=Path(source["historical_fixed_trajectory"]),
            historical_every_frame_offline=Path(source["historical_every_frame_trajectory"]),
            historical_confidence_offline=Path(source["offline_confidence_trajectory"]))
        evaluate_group(scene,"independent_live",frames,descriptions,source["dataset"],records,errors,failures)
        raw=mapped(enabled/"raw.tum",frames)
        base=mapped(disabled/"raw.tum",frames)
        old=load_frames(source["baseline_frames"])
        if raw is None:
            parity.append(dict(scene=scene,status="missing enabled raw trajectory"));continue
        direct=max((float(np.max(np.abs(pose-base[i]))) for i,pose in raw.items()),default=None)
        old_diff=max((float(np.max(np.abs(pose-old[i].t_wc))) for i,pose in raw.items()),default=None)
        snapshots=[disabled/"cache/final.snapshot",enabled/"cache/final.snapshot"]
        same=file_sha256(snapshots[0])==file_sha256(snapshots[1]) if all(p.exists() for p in snapshots) else None
        expected=[k["frame_id"] for k in source["keyframes"]]
        inventory={}
        for mode in ["disabled","enabled"]:
            p=args.runs/(scene+"-"+mode)/"emissions.jsonl"
            rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
            actual=[x["frame_id"] for x in rows if x["is_keyframe"]]
            inventory[mode]=dict(expected_keyframes=len(expected),actual_keyframes=len(actual),
                frozen_frame_ids_identical=actual==expected)
        parity.append(dict(scene=scene,status="evaluated",raw_vs_disabled_max_element=direct,
            raw_vs_historical_input_cache_max_element=old_diff,final_map_snapshot_identical=same,
            snapshot_sha256={p.parent.parent.name:file_sha256(p) for p in snapshots if p.exists()},
            keyframe_inventory=inventory))
    args.output.mkdir()
    payload=dict(records=records,failures=failures,parity=parity,
        runtime=json.loads((args.runs/"status.json").read_text()),
        fit_seal_sha256=file_sha256(args.runs/"fit-seal.json"),
        scope=plan["scope"],historical_reference_scope="same input timestamps, cached offline trajectories; no current per-frame/fixed matching runs or runtime comparison",
        correction_feedback=False,commit_push=False)
    (args.output/"summary.json").write_text(json.dumps(payload,indent=2)+"\n")
    for name,rows in [("summary.csv",records),("frame_errors.csv",errors)]:
        if rows:
            with (args.output/name).open("x",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
                writer.writeheader();writer.writerows(rows)
    check();print(json.dumps(dict(records=len(records),failures=failures,parity=parity),indent=2))
if __name__=="__main__":main()
