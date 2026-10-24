#!/usr/bin/env python3
"""Pace frozen baseline poses while a separate worker computes real auxiliaries."""
import argparse,csv,hashlib,json,os,select,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from adaptivefusion.auxiliary_output import BoundedAuxiliaryClient,CorrectionOutput
from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames

def save(path,value):
    with Path(path).open("x") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")

def terminate(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()

def pose_line(timestamp,pose):
    q=Rotation.from_matrix(pose[:3,:3]).as_quat()
    return f"{timestamp:.9f} "+" ".join(f"{x:.17g}" for x in [*pose[:3,3],*q])+"\n"

def run_scene(src,plan,protocol,output):
    output.mkdir()
    frames=load_frames(src["baseline_frames"]);keys=src["keyframes"]
    raw_digest=hashlib.sha256(np.array([f.t_wc for f in frames]).tobytes()).hexdigest()
    for f in frames:f.t_wc.setflags(write=False)
    socket_path=output.parent/(src["scene"]+".sock")
    ready=output/"ready.json";log=(output/"worker.log").open("x")
    command=[sys.executable,"-B",str(Path(__file__).parent/"auxiliary_pose_service.py"),
        "--protocol",str(protocol),"--scene",src["scene"],"--socket",str(socket_path),
        "--output",str(output/"worker"),"--ready",str(ready)]
    begin=time.perf_counter()
    worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while not ready.exists() and worker.poll() is None and time.perf_counter()-begin<plan["budget"]["worker_start_timeout_s"]:
        time.sleep(.02)
    if not ready.exists():
        terminate(worker);log.close()
        raise RuntimeError("auxiliary worker failed to become ready")
    startup=time.perf_counter()-begin
    connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);connection.connect(str(socket_path))
    adapter=CorrectionOutput();events=[];emissions=[];requests=[]
    timeline=(output/"events.jsonl").open("x")
    epoch=time.perf_counter()
    def event(record):
        events.append(record);timeline.write(json.dumps(record,allow_nan=False)+"\n");timeline.flush()
        if record["status"]=="service_disabled" and worker.poll() is None:
            os.killpg(worker.pid,signal.SIGTERM)
    client=BoundedAuxiliaryClient(connection,adapter,plan["budget"]["request_timeout_s"],event)
    by_frame={k["frame_id"]:j for j,k in enumerate(keys) if j>0}
    raw_max=0.;visible=-1;sent_budget=0
    raw_file=(output/"raw.tum").open("x");corrected_file=(output/"corrected.tum").open("x")
    try:
        for frame in frames:
            due=frame.timestamp-frames[0].timestamp+plan["budget"]["synchronization_wait_s"]
            while True:
                now=time.perf_counter()-epoch
                client.poll(now,visible)
                remaining=due-(time.perf_counter()-epoch)
                if remaining<=0:break
                if client.enabled:
                    select.select([connection],[],[],min(.005,remaining))
                else:time.sleep(min(.005,remaining))
            visible=frame.id;now=time.perf_counter()-epoch
            client.poll(now,visible)
            if frame.id in by_frame:
                j=by_frame[frame.id];a,b=keys[j-1],keys[j]
                request=dict(schema=1,request_id=src["scene"]+":"+str(j),scene=src["scene"],
                    source_keyframe=j-1,target_keyframe=j,source_frame=a["frame_id"],target_frame=b["frame_id"],
                    source_timestamp=a["timestamp"],target_timestamp=b["timestamp"])
                request["arrival_s"]=due
                requests.append(request);sent_budget+=1
                if sent_budget>src["request_budget"]:raise ValueError("request budget exceeded")
                client.submit(request,time.perf_counter()-epoch,visible)
                client.poll(time.perf_counter()-epoch,visible)
            emitted=adapter.emit(frame.id,frame.timestamp,frame.t_wc,frame.valid)
            if emitted:
                if not np.array_equal(emitted.raw,frame.t_wc):raise ValueError("raw pose changed")
                raw_max=max(raw_max,float(np.max(np.abs(emitted.raw-frame.t_wc))))
                raw_file.write(pose_line(frame.timestamp,emitted.raw));raw_file.flush()
                corrected_file.write(pose_line(frame.timestamp,emitted.corrected));corrected_file.flush()
            emissions.append(dict(frame_id=frame.id,timestamp=frame.timestamp,valid=frame.valid,
                arrival_s=due,emitted_s=time.perf_counter()-epoch,
                used_correction_request=adapter.request_id,used_target_frame=adapter.target_frame,
                raw_pose_element_difference=0.,service_enabled=client.enabled))
        end=time.perf_counter()-epoch
        client.finish(end)
    finally:
        if client.enabled:client.finish(time.perf_counter()-epoch)
        raw_file.close();corrected_file.close();timeline.close()
        try:worker.wait(timeout=2)
        except subprocess.TimeoutExpired:terminate(worker)
        log.close()
    assert raw_digest==hashlib.sha256(np.array([f.t_wc for f in frames]).tobytes()).hexdigest()
    assert all(x["used_target_frame"]<=x["frame_id"] for x in emissions)
    for name,rows in [("requests",requests),("emissions",emissions)]:
        with (output/(name+".jsonl")).open("x") as f:
            for row in rows:f.write(json.dumps(row)+"\n")
    worker_jobs=[json.loads(line) for line in (output/"worker/jobs.jsonl").read_text().splitlines()] if (output/"worker/jobs.jsonl").exists() else []
    delivered=[e for e in events if e["status"]=="delivered"]
    arrivals={x["request_id"]:x["arrival_s"] for x in requests}
    latencies=[x["at_s"]-arrivals[x["request_id"]] for x in delivered]
    dispatch=[x["at_s"]-x["dispatch_s"] for x in delivered]
    source_lag=[x["emitted_s"]-x["arrival_s"] for x in emissions]
    service=[x for x in worker_jobs if x["status"]=="ok"]
    record=dict(scene=src["scene"],status="complete",input_frames=len(frames),
        valid_frames=sum(f.valid for f in frames),request_budget=src["request_budget"],
        started_jobs=sum(e["status"]=="dispatched" for e in events),delivered_jobs=len(delivered),
        dropped_queue_full=sum(e["status"]=="dropped_queue_full" for e in events),
        unfinished_at_stream_end=sum(e["status"]=="unfinished_at_stream_end" for e in events),
        service_disabled_events=sum(e["status"]=="service_disabled" for e in events),
        worker_failed_jobs=sum(x["status"]!="ok" for x in worker_jobs),
        F_failures=sum(x.get("matching_status")!="ok" for x in service),
        native_PnP_rejections=sum(not x.get("native_accepted") for x in service),
        confidence_errors=sum(x.get("confidence_status")=="confidence_error" for x in service),
        worker_returncode=worker.returncode,maximum_waiting_jobs=client.maximum_waiting,
        raw_pose_max_element_change=raw_max,raw_inputs_unchanged=True,past_output_revisions=0,
        causality_violations=0,model_startup_wall_s=startup,model_ready=json.loads(ready.read_text()),
        total_wall_including_startup_s=time.perf_counter()-begin,stream_wall_s=end,
        delivery_latency_ms=dict(median=float(np.median(latencies))*1000,P95=float(np.quantile(latencies,.95))*1000,max=max(latencies)*1000) if latencies else None,
        dispatch_to_receive_ms=dict(median=float(np.median(dispatch))*1000,P95=float(np.quantile(dispatch,.95))*1000) if dispatch else None,
        frame_emission_lateness_ms=dict(median=float(np.median(source_lag))*1000,P95=float(np.quantile(source_lag,.95))*1000,max=max(source_lag)*1000),
        peak_GPU_allocated_MiB=max([x.get("peak_GPU_allocated_bytes",0) for x in service],default=0)/1024**2,
        peak_GPU_reserved_MiB=max([x.get("peak_GPU_reserved_bytes",0) for x in service],default=0)/1024**2,
        component_total_s={key:sum(x.get(key,0) for x in service) for key in ["matching_wall_s","native_PnP_wall_s","confidence_wall_s","graph_wall_s"]},
        scope="real auxiliary worker + timestamp-paced cached baseline producer; original tracker/map/BA not running")
    save(output/"summary.json",record)
    return record

def main():
    parser=argparse.ArgumentParser()
    for name in ["protocol","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError("refuse overwrite")
    plan=json.loads(args.protocol.read_text())
    for path,sha in plan["input_sha256"].items():
        if file_sha256(path)!=sha:raise ValueError("frozen input changed: "+path)
    args.output.mkdir();status=[];begin=time.perf_counter()
    for src in json.loads(Path(plan["sources"]).read_text()):
        try:record=run_scene(src,plan,args.protocol,args.output/src["scene"])
        except Exception as error:record=dict(scene=src["scene"],status="failed",reason=str(error),traceback=traceback.format_exc())
        status.append(record);(args.output/"status.json").write_text(json.dumps(status,indent=2)+"\n")
        print(json.dumps(record),flush=True)
    for path,sha in plan["input_sha256"].items():
        if file_sha256(path)!=sha:raise ValueError("frozen input changed during run: "+path)
    save(args.output/"wall.json",dict(seconds=time.perf_counter()-begin,GT_read=False))
    save(args.output/"fit-seal.json",{str(p.relative_to(args.output)):file_sha256(p) for p in args.output.rglob("*") if p.is_file()})
    print("ASYNC_OUTPUT_SEALED_WITHOUT_GT",flush=True)
if __name__=="__main__":main()
