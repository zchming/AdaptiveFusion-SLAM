#!/usr/bin/env python3
"""Frozen CPU-budget control; isolated original orchestrator with a resource wrapper."""
import argparse,hashlib,json,os,select,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from adaptivefusion.auxiliary_output import BoundedAuxiliaryClient,CorrectionOutput
from adaptivefusion.geometry import check_pose
from adaptivefusion.state import file_sha256
from run_async_auxiliary_output import pose_line,save,terminate

def run_arm(source,arm,plan,protocol,output):
    output.mkdir()
    mode=arm["mode"];scene=arm["scene"];keys=source["keyframes"]
    enabled=mode!="disabled";events=[];emissions=[];requests=[];faults=[];client=None;worker=None
    event_file=(output/"events.jsonl").open("x")
    raw_file=(output/"raw.tum").open("x");corrected_file=(output/"corrected.tum").open("x")
    epoch=time.perf_counter();visible=-1;delivery_count=0
    def event(record):
        nonlocal delivery_count
        events.append(record);event_file.write(json.dumps(record,allow_nan=False)+"\n");event_file.flush()
        if record["status"]=="delivered":
            delivery_count+=1
            if mode in ["disconnect","timeout"] and not faults and delivery_count==arm["inject_after_deliveries"]:
                item=dict(status="fault_injected",mode=mode,at_s=time.perf_counter()-epoch,
                          visible_frame=visible,after_deliveries=delivery_count)
                faults.append(item);events.append(item)
                event_file.write(json.dumps(item)+"\n");event_file.flush()
                os.killpg(worker.pid,signal.SIGKILL if mode=="disconnect" else signal.SIGSTOP)
        if record["status"]=="service_disabled" and worker and worker.poll() is None:
            os.killpg(worker.pid,signal.SIGKILL)
    adapter=CorrectionOutput();startup=0.;worker_log=None;cpp=None
    lib=plan["paths"]["native_library_path"];env=os.environ.copy()
    env["LD_LIBRARY_PATH"]=":".join([lib,lib+"/blas",lib+"/lapack",str(Path(lib).parent),"/usr/lib/wsl/lib",env.get("LD_LIBRARY_PATH","")])
    producer_log=(output/"producer.log").open("x")
    a,b=socket.socketpair(socket.AF_UNIX,socket.SOCK_DGRAM)
    a.setblocking(False);a.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,1024*1024)
    commands=[]
    expected=min(arm["max_frames"],plan["frame_counts"][scene])
    try:
        if enabled:
            socket_path=output.parent/(scene+"-"+mode+".sock")
            # Linux Unix socket paths are limited to about 108 bytes.
            socket_path=Path("/tmp")/("afs37-"+hashlib.sha256(str(output).encode()).hexdigest()[:12]+".sock")
            ready=output/"ready.json";worker_log=(output/"worker.log").open("x")
            cmd=[sys.executable,"-B",str(Path(__file__).parent/"cpu_budget_pose_service.py"),
                "--protocol",str(protocol),"--scene",scene,"--socket",str(socket_path),
                "--output",str(output/"worker"),"--ready",str(ready),"--request-cap",str(arm["request_cap"]),"--cpu-profile",arm["profile"]]
            commands.append(cmd);begin=time.perf_counter()
            worker=subprocess.Popen(cmd,stdout=worker_log,stderr=subprocess.STDOUT,start_new_session=True)
            while not ready.exists() and worker.poll() is None and time.perf_counter()-begin<plan["budget"]["worker_start_timeout_s"]:
                time.sleep(.02)
            startup=time.perf_counter()-begin
            if ready.exists():
                connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);connection.connect(str(socket_path))
                client=BoundedAuxiliaryClient(connection,adapter,plan["budget"]["request_timeout_s"],event)
            else:event(dict(status="worker_start_failed",at_s=time.perf_counter()-epoch))
        cmd=[plan["producer_binary"],"--pose-fd",str(b.fileno()),"--stream-stats",str(output/"cpp-stream.csv"),"--",
            source["dataset"],str(output/"cpp.tum"),str(arm["max_frames"]),"-","none",str(output/"cache"),"--export-keyframe-features"]
        commands.append(cmd);save(output/"commands.json",commands)
        cpp=subprocess.Popen(cmd,pass_fds=[b.fileno()],stdout=producer_log,stderr=subprocess.STDOUT,
            env=env,start_new_session=True)
        b.close();producer_begin=time.perf_counter();by_frame={k["frame_id"]:j for j,k in enumerate(keys)}
        prefix={};max_raw_difference=0.;last_received=-1;received=0;gaps=[]
        while True:
            now=time.perf_counter()-epoch
            if client:client.poll(now,visible)
            readable,_,_=select.select([a],[],[],.002)
            if not readable:
                if cpp.poll() is not None:break
                if time.perf_counter()-producer_begin>plan["budget"]["arm_wall_timeout_s"]:
                    raise TimeoutError("live producer arm wall budget")
                continue
            packet=a.recv(65536);f=json.loads(packet)
            if f["schema"]!=1 or f["frame_id"]<=last_received:raise ValueError("invalid/duplicate producer frame")
            if f["frame_id"]!=last_received+1:gaps.append([last_received+1,f["frame_id"]-1])
            last_received=f["frame_id"];visible=last_received;received+=1
            pose=check_pose(np.asarray(f["t_wc"],float));pose.setflags(write=False)
            now=time.perf_counter()-epoch
            if client:client.poll(now,visible)
            j=by_frame.get(visible)
            if j is not None and f["valid"]:
                key=keys[j]
                if key["timestamp"]!=f["timestamp"]:raise ValueError("frozen keyframe timestamp changed")
                prefix[j]=dict(frame_id=visible,timestamp=f["timestamp"],t_wc=pose.tolist())
                if enabled and j>0:
                    request=dict(schema=1,request_id=scene+"-"+mode+":"+str(j),scene=scene,
                        source_keyframe=j-1,target_keyframe=j,source_frame=keys[j-1]["frame_id"],target_frame=visible,
                        source_timestamp=keys[j-1]["timestamp"],target_timestamp=f["timestamp"],
                        arrival_s=f["published_monotonic_s"]-epoch,sensor_due_s=f["due_monotonic_s"]-epoch)
                    if all(k in prefix for k in range(j+1)):
                        request["live_prefix"]=[prefix[k] for k in range(j+1)]
                        requests.append({k:v for k,v in request.items() if k!="live_prefix"})
                        if client:client.submit(request,time.perf_counter()-epoch,visible)
                        else:event(dict(status="not_submitted_service_unavailable",request_id=request["request_id"],at_s=time.perf_counter()-epoch))
                    else:event(dict(status="missing_live_prefix",target_frame=visible,at_s=time.perf_counter()-epoch))
            emitted=adapter.emit(visible,f["timestamp"],pose,f["valid"])
            if emitted:
                max_raw_difference=max(max_raw_difference,float(np.max(np.abs(emitted.raw-pose))))
                raw_file.write(pose_line(f["timestamp"],emitted.raw));raw_file.flush()
                corrected_file.write(pose_line(f["timestamp"],emitted.corrected));corrected_file.flush()
            record=dict(f,received_monotonic_s=time.perf_counter(),used_correction_request=adapter.request_id,
                used_target_frame=adapter.target_frame,service_enabled=bool(client and client.enabled))
            assert adapter.target_frame<=visible
            emissions.append(record)
            if visible==expected-1:break
        end=time.perf_counter()
        if client:client.finish(end-epoch)
        try:cpp.wait(timeout=15)
        except subprocess.TimeoutExpired:terminate(cpp)
        if worker:
            try:worker.wait(timeout=2)
            except subprocess.TimeoutExpired:terminate(worker)
        for name,rows in [("requests",requests),("emissions",emissions)]:
            with (output/(name+".jsonl")).open("x") as stream:
                for row in rows:stream.write(json.dumps(row,allow_nan=False)+"\n")
        delivered=[e for e in events if e["status"]=="delivered"]
        lookup={r["request_id"]:r for r in requests}
        latencies=[1000*(e["at_s"]-lookup[e["request_id"]]["arrival_s"]) for e in delivered]
        sensor_latency=[1000*(e["at_s"]-lookup[e["request_id"]]["sensor_due_s"]) for e in delivered]
        lateness=[1000*(f["received_monotonic_s"]-f["due_monotonic_s"]) for f in emissions]
        handoff=[1000*(f["received_monotonic_s"]-f["published_monotonic_s"]) for f in emissions]
        def quantile(values):
            return dict(median=float(np.median(values)),P95=float(np.quantile(values,.95)),max=max(values)) if values else None
        jobs=[json.loads(x) for x in (output/"worker/jobs.jsonl").read_text().splitlines()] if (output/"worker/jobs.jsonl").exists() else []
        matching=[json.loads(x) for x in (output/"worker/matching.jsonl").read_text().splitlines()] if (output/"worker/matching.jsonl").exists() else []
        result=dict(scene=scene,mode=mode,status="complete" if cpp.returncode==0 and received==expected else "incomplete",
            expected_frames=expected,received_frames=received,valid_frames=sum(f["valid"] for f in emissions),
            producer_returncode=cpp.returncode,worker_returncode=worker.returncode if worker else None,
            request_slots=len(requests),request_cap=arm["request_cap"] if enabled else 0,
            matching_calls=len(list((output/"worker").glob("job-*/live-request.json"))),matching_completed=len(matching),
            native_PnP_calls=len(list((output/"worker").glob("job-*/pnp/input.pnp"))),
            prefix_solves=len(list((output/"worker").glob("job-*/prefix.graph"))),
            completed_prefix_solves=sum("solver" in j for j in jobs),delivered_jobs=len(delivered),
            dropped_queue_full=sum(e["status"]=="dropped_queue_full" for e in events),
            unfinished_at_stream_end=sum(e["status"]=="unfinished_at_stream_end" for e in events),
            service_disabled_events=sum(e["status"]=="service_disabled" for e in events),
            not_submitted_service_unavailable=sum(e["status"]=="not_submitted_service_unavailable" for e in events),
            worker_failed_jobs=sum(j["status"]!="ok" for j in jobs),transport_gaps=gaps,
            raw_pose_max_element_change=max_raw_difference,past_output_revisions=0,causality_violations=0,
            auxiliary_delivery_latency_ms=quantile(latencies),auxiliary_from_sensor_due_ms=quantile(sensor_latency),
            raw_end_to_end_latency_ms=quantile(lateness),raw_cpp_to_adapter_ms=quantile(handoff),
            worker_startup_wall_s=startup,producer_wall_s=end-producer_begin,
            peak_GPU_allocated_MiB=max([j.get("peak_GPU_allocated_bytes",0) for j in jobs],default=0)/1024**2,
            peak_GPU_reserved_MiB=max([j.get("peak_GPU_reserved_bytes",0) for j in jobs],default=0)/1024**2,
            injected_faults=faults,live_pose_source=True,correction_feedback=False)
        if mode in ["disconnect","timeout"]:
            result["fault_triggered"]=bool(faults)
            fault_time=faults[0]["at_s"]+epoch if faults else end
            result["frames_after_fault"]=sum(f["received_monotonic_s"]>=fault_time for f in emissions)
            result["fallback_observed"]=any(e["status"]=="service_disabled" for e in events)
        result["profile"]=arm["profile"]
        result["CPU_resources"]=json.loads((output/"worker-resources.json").read_text()) if (output/"worker-resources.json").exists() else None
        result["CPU_usage"]=json.loads((output/"worker-cpu-total.json").read_text()) if (output/"worker-cpu-total.json").exists() else None
        save(output/"summary.json",result)
        return result
    finally:
        if client and client.enabled:client.finish(time.perf_counter()-epoch)
        if cpp and cpp.poll() is None:terminate(cpp)
        if worker and worker.poll() is None:terminate(worker)
        a.close();b.close();producer_log.close()
        if worker_log:worker_log.close()
        raw_file.close();corrected_file.close();event_file.close()

def main():
    p=argparse.ArgumentParser()
    for name in ["protocol","output"]:p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.protocol.read_text())
    if a.output.exists():raise ValueError("refuse overwrite")
    for path,sha in plan["input_sha256"].items():
        if file_sha256(path)!=sha:raise ValueError("frozen input changed: "+path)
    sources={s["scene"]:s for s in json.loads(Path(plan["sources"]).read_text())}
    a.output.mkdir();status=[];begin=time.perf_counter()
    for arm in plan["arms"]:
        dest=a.output/(arm["scene"]+"-"+arm["mode"])
        try:r=run_arm(sources[arm["scene"]],arm,plan,a.protocol,dest)
        except Exception as error:r=dict(scene=arm["scene"],mode=arm["mode"],profile=arm["profile"],status="failed",reason=str(error),traceback=traceback.format_exc())
        status.append(r);(a.output/"status.json").write_text(json.dumps(status,indent=2)+"\n");print(json.dumps(r),flush=True)
    for path,sha in plan["input_sha256"].items():
        if file_sha256(path)!=sha:raise ValueError("frozen input changed during run: "+path)
    save(a.output/"wall.json",dict(seconds=time.perf_counter()-begin,GT_read=False))
    save(a.output/"fit-seal.json",{str(p.relative_to(a.output)):file_sha256(p) for p in a.output.rglob("*") if p.is_file()})
    print("CPU_BUDGET_LIVE_SEALED_WITHOUT_GT",flush=True)
if __name__=="__main__":main()
