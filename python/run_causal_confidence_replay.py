#!/usr/bin/env python3
"""Causal cached RGB-D replay; no matcher/PnP calls, GT, or tracking feedback."""
import argparse
import csv
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
from adaptivefusion.edge_confidence import confidence_pair
from adaptivefusion.geometry import check_pose, inverse
from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames, write_tum
from run_keyframe_auxiliary import read_nodes

def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")

def prefix_graph(path, original, target, available, weights):
    vertices, edges = [], []
    for line in original.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "V" and int(t[1]) <= target:
            vertices.append(line)
        elif t[0] == "E":
            i, j, kind = map(int, t[1:4])
            if j > target or (kind and j not in available):
                continue
            w = weights[j]["auxiliary" if kind else "baseline"]["weight"] if j in available else 1.
            assert 0 < w <= 1
            t[-2] = format(float(t[-2])/np.sqrt(w), ".17g")
            t[-1] = format(float(t[-1])/np.sqrt(w), ".17g")
            edges.append(" ".join(t))
    assert len(vertices) == target+1
    text = ["AFS_KF_GRAPH 1", f"NODES {len(vertices)}", *vertices,
            f"EDGES {len(edges)}", *edges, "END"]
    with path.open("x") as f:
        f.write("\n".join(text)+"\n")

def replay_scene(src, plan, output, env):
    scene = src["scene"]
    frames = load_frames(src["baseline_frames"])
    keys = src["keyframes"]
    constraints = [json.loads(x) for x in (Path(src["evidence"])/"constraints.jsonl").read_text().splitlines()]
    old_weights = json.loads(Path(src["confidence_weights"]).read_text())
    original = (Path(src["evidence"])/"auxiliary.graph").read_text()
    assert len(constraints) == src["request_budget"] == len(old_weights)
    assert all(x["target_keyframe"] == x["source_keyframe"]+1 for x in constraints)
    weights, compute_cost, confidence_records = {}, {}, []
    for index, c in enumerate(constraints):
        j = c["target_keyframe"]
        begin = time.perf_counter()
        if c["accepted"]:
            with np.load(Path(src["evidence"])/"service.matches"/f"{index:06d}.npz") as f:
                rows = f["rows"].copy()
            score = confidence_pair(
                inverse(frames[c["target_frame"]].t_wc)@frames[c["source_frame"]].t_wc,
                np.asarray(c["t_target_source"]), rows, keys[j]["depth"], plan["config"])
        else:
            score = dict(status="native_rejected", baseline=dict(weight=1.),
                         auxiliary=dict(weight=plan["config"]["minimum_information_weight"]))
        compute_cost[j] = time.perf_counter()-begin
        for name in ["baseline", "auxiliary"]:
            assert abs(score[name]["weight"]-old_weights[index][name]["weight"]) <= 1e-12
        weights[j] = score
        confidence_records.append(dict(source_keyframe=c["source_keyframe"], target_keyframe=j,
            confidence_cpu_s=compute_cost[j], old_weights_unchanged=True, **score))
    save(output/"confidence-before-replay.json", confidence_records)
    summaries = []
    for variant in plan["variants"]:
        dest = output/variant
        dest.mkdir()
        queued = variant == "causal_queued"
        jobs, emissions, trajectory, completed_weights, fits, failures = [], [], [], set(), [], []
        active = pending = None
        correction = np.eye(4)
        last_target = 0
        last_completed_s = None
        max_waiting = 0
        target_by_frame = {c["target_frame"]:c for c in constraints}
        clock_zero = frames[0].timestamp
        wait = plan["synchronization_wait_s"]
        def start_job(c, now):
            nonlocal completed_weights
            j = c["target_keyframe"]
            available = completed_weights | {j}
            graph = dest/f"prefix-{j:04d}.graph"
            prefix_graph(graph, original, j, available, weights)
            native_dest = dest/f"prefix-{j:04d}"
            beg = time.perf_counter()
            record = dict(target_keyframe=j, target_frame=c["target_frame"],
                          arrival_s=frames[c["target_frame"]].timestamp-clock_zero+wait,
                          start_s=now, status="process_failed", native_accepted=c["accepted"])
            candidate = None
            try:
                run = subprocess.run([plan["binary"], str(graph), str(native_dest), "quadratic"],
                    env=env, capture_output=True, text=True,
                    timeout=plan["budget"]["per_solve_timeout_s"])
                (dest/f"prefix-{j:04d}.log").write_text(run.stdout+run.stderr)
                record["returncode"] = run.returncode
                report = json.loads((native_dest/"report.json").read_text()) if (native_dest/"report.json").exists() else {}
                record["solver"] = report
                if run.returncode != 0 or not report.get("usable"):
                    raise ValueError("prefix graph unusable")
                poses = read_nodes(native_dest/"nodes.txt")
                assert len(poses) == j+1
                candidate = check_pose(poses[-1]@inverse(frames[c["target_frame"]].t_wc))
                record["status"] = "complete"
            except Exception as error:
                record.update(status="failed", reason=str(error), traceback=traceback.format_exc())
                failures.append(record)
            native_wall = time.perf_counter()-beg
            stored = float(c["wall_s"])
            assert np.isfinite(stored) and stored > 0
            duration = stored+compute_cost[j]+native_wall if queued else 0.
            record.update(prefix_process_wall_s=native_wall, cached_matching_PnP_wall_s=stored,
                confidence_cpu_s=compute_cost[j], service_s=duration, finish_s=now+duration,
                wait_s=now-record["arrival_s"], graph_nodes=j+1,
                max_constraint_frame=c["target_frame"])
            jobs.append(record)
            fits.append(record)
            return dict(record=record, candidate=candidate, available=available, target=j)
        def advance(now):
            nonlocal active, pending, correction, last_target, last_completed_s, completed_weights
            while active is not None and active["record"]["finish_s"] <= now:
                done = active
                finish = done["record"]["finish_s"]
                done["record"]["delivered_in_stream"] = True
                if done["candidate"] is not None:
                    correction = done["candidate"]
                    completed_weights = done["available"]
                    last_target = done["target"]
                    last_completed_s = finish
                active = None
                if pending is not None:
                    c = pending
                    pending = None
                    active = start_job(c, finish)
        for f in frames:
            now = f.timestamp-clock_zero+wait
            advance(now)
            if f.id in target_by_frame:
                c = target_by_frame[f.id]
                if active is None:
                    active = start_job(c, now)
                elif pending is None:
                    pending = c
                    max_waiting = max(max_waiting, 1)
                else:
                    jobs.append(dict(target_keyframe=c["target_keyframe"],target_frame=f.id,
                                     arrival_s=now,status="dropped_queue_full",delivered_in_stream=False))
                advance(now)
            if f.valid:
                assert keys[last_target]["frame_id"] <= f.id
                assert last_completed_s is None or last_completed_s <= now
                pose = check_pose(correction@f.t_wc)
                trajectory.append((f.timestamp,pose))
            emissions.append(dict(frame_id=f.id, timestamp=f.timestamp, valid=f.valid,
                available_s=now, used_target_keyframe=last_target,
                used_target_frame=keys[last_target]["frame_id"],
                correction_completed_s=last_completed_s,
                output_written=f.valid, uses_future_constraint=False))
        stream_end = frames[-1].timestamp-clock_zero+wait
        if active is not None:
            active["record"]["delivered_in_stream"] = False
            active["record"]["delivery_status"] = "unfinished_at_stream_end"
        if pending is not None:
            jobs.append(dict(target_keyframe=pending["target_keyframe"],target_frame=pending["target_frame"],
                arrival_s=frames[pending["target_frame"]].timestamp-clock_zero+wait,
                status="not_started_at_stream_end",delivered_in_stream=False))
        assert len(jobs) == src["request_budget"]
        assert len(fits) <= src["request_budget"]
        assert len(trajectory) == sum(f.valid for f in frames)
        write_tum(dest/"trajectory.tum",trajectory)
        for name,rows in [("jobs",jobs),("emissions",emissions)]:
            with (dest/(name+".jsonl")).open("x") as f:
                for row in rows: f.write(json.dumps(row,allow_nan=False)+"\n")
        delivered = [x for x in fits if x.get("delivered_in_stream")]
        delays = [x["finish_s"]-x["arrival_s"] for x in delivered]
        result = dict(scene=scene,variant=variant,status="complete",input_frames=len(frames),
            valid_frames=len(trajectory),lost_frames=len(frames)-len(trajectory),
            request_budget=src["request_budget"],prefix_solves=len(fits),solver_failures=len(failures),
            delivered_jobs=len(delivered),dropped_queue_full=sum(x["status"]=="dropped_queue_full" for x in jobs),
            unfinished_at_stream_end=sum(x.get("delivery_status")=="unfinished_at_stream_end" or x["status"]=="not_started_at_stream_end" for x in jobs),
            maximum_waiting_jobs=max_waiting,causality_violations=0,
            past_output_revisions=0,stream_duration_s=stream_end-wait,
            delivery_latency_s=dict(median=float(np.median(delays)),P95=float(np.quantile(delays,.95)),max=max(delays)) if delays else None,
            native_process_total_s=sum(x["prefix_process_wall_s"] for x in fits),
            native_solver_total_s=sum(x.get("solver",{}).get("solve_s",0) for x in fits),
            confidence_cpu_total_s=sum(compute_cost.values()),
            new_matcher_calls=0,new_native_PnP_calls=0,
            timing_scope="cached previous matching/PnP timings plus current confidence/prefix CPU; no online baseline contention or model startup")
        save(dest/"summary.json",result)
        summaries.append(result)
        print(json.dumps(result),flush=True)
    return summaries

def main():
    p=argparse.ArgumentParser()
    for name in ["protocol","sources","output"]:
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    plan=json.loads(a.protocol.read_text())
    sources=json.loads(a.sources.read_text())
    if a.output.exists(): raise ValueError("refuse overwrite")
    for path,sha in plan["input_sha256"].items():
        if file_sha256(path)!=sha:raise ValueError("frozen input changed: "+path)
    a.output.mkdir()
    env=os.environ.copy()
    lib=plan["native_library_path"]
    env["LD_LIBRARY_PATH"]=":".join([lib,lib+"/blas",lib+"/lapack",str(Path(lib).parent),env.get("LD_LIBRARY_PATH","")])
    start=time.perf_counter()
    summary=[]
    for src in sources:
        dest=a.output/src["scene"]
        dest.mkdir()
        try:
            summary.extend(replay_scene(src,plan,dest,env))
        except Exception as error:
            item=dict(scene=src["scene"],status="failed",reason=str(error),traceback=traceback.format_exc())
            save(dest/"failure.json",item)
            summary.append(item)
            print(json.dumps(item),flush=True)
        (a.output/"status.json").write_text(json.dumps(summary,indent=2)+"\n")
    for path,sha in plan["input_sha256"].items():
        if file_sha256(path)!=sha:raise ValueError("frozen input changed during replay: "+path)
    save(a.output/"wall.json",dict(seconds=time.perf_counter()-start,GT_read=False,new_matcher_calls=0,new_PnP_calls=0))
    save(a.output/"fit-seal.json",{str(p.relative_to(a.output)):file_sha256(p) for p in a.output.rglob("*") if p.is_file()})
    print("CAUSAL_REPLAY_SEALED_WITHOUT_GT",flush=True)
if __name__=="__main__":main()
