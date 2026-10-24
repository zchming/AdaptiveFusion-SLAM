#!/usr/bin/env python3
"""Independent GPU auxiliary worker, operating only on admitted prefix requests."""
import argparse,json,os,shlex,socket,subprocess,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from adaptivefusion.edge_confidence import confidence_pair
from adaptivefusion.geometry import inverse,check_pose
from adaptivefusion.native_pnp import native_fit
from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames
from offline_matching_service import Matcher
from run_keyframe_auxiliary import write_graph,read_nodes

def save(path,value):
    with Path(path).open("x") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")

def main():
    parser=argparse.ArgumentParser()
    for name in ["protocol","scene","socket","output","ready"]:
        parser.add_argument("--"+name,type=Path if name!="scene" else str,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.socket.exists() or args.ready.exists():raise ValueError("refuse overwrite")
    plan=json.loads(args.protocol.read_text())
    sources=json.loads(Path(plan["sources"]).read_text())
    source=next(s for s in sources if s["scene"]==args.scene)
    paths=plan["paths"];config=plan["config"]
    if file_sha256(Path(__file__).parent/"adaptivefusion/edge_confidence.py")!=plan["confidence_source_sha256"]:
        raise ValueError("confidence source changed")
    frames=load_frames(source["baseline_frames"]);keys=source["keyframes"]
    args.output.mkdir()
    matcher=Matcher(SimpleNamespace(dataset=Path(source["dataset"]),lightglue_source=Path(paths["lightglue_source"]),
        weights=Path(paths["weights"]),log=args.output/"matching.jsonl"))
    lib=paths["native_library_path"]
    os.environ["LD_LIBRARY_PATH"]=":".join([lib,lib+"/blas",lib+"/lapack",str(Path(lib).parent),"/usr/lib/wsl/lib",os.environ.get("LD_LIBRARY_PATH","")])
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    server.bind(str(args.socket));server.listen(1)
    save(args.ready,dict(model_load_s=matcher.load_s,gpu=matcher.torch.cuda.get_device_name(0),
        confidence_source_sha256=plan["confidence_source_sha256"],GT_read=False))
    weights={};auxiliary=[];seen=set();count=0;last_target=0
    log=(args.output/"jobs.jsonl").open("x")
    connection,_=server.accept()
    try:
        with connection,connection.makefile("rwb",buffering=0) as stream:
            while True:
                line=stream.readline()
                if not line:break
                request=json.loads(line)
                start=time.perf_counter()
                result=dict(request,status="failed",worker_start_s=start)
                try:
                    i,j=request["source_keyframe"],request["target_keyframe"]
                    if request.get("schema")!=1 or request["scene"]!=args.scene or j!=i+1 or not 0<=i<j<len(keys):
                        raise ValueError("invalid request")
                    if request["request_id"] in seen or j<=last_target:raise ValueError("duplicate/stale request")
                    for prefix,key in [("source",keys[i]),("target",keys[j])]:
                        if request[prefix+"_frame"]!=key["frame_id"] or request[prefix+"_timestamp"]!=key["timestamp"]:
                            raise ValueError("request metadata differs from frozen inputs")
                    if count>=source["request_budget"]:raise ValueError("scene request budget exceeded")
                    seen.add(request["request_id"]);last_target=j;count+=1
                    dest=args.output/f"job-{j:04d}";dest.mkdir()
                    request_index=matcher.requests
                    service_start=time.perf_counter()
                    matcher.request(shlex.join(["magsac",str(keys[i]["frame_id"]),str(keys[j]["frame_id"]),
                        keys[i]["rgb"],keys[i]["depth"],keys[j]["rgb"]]))
                    with np.load(matcher.evidence/f"{request_index:06d}.npz") as data:rows=data["rows"].copy()
                    matching_wall=time.perf_counter()-service_start
                    service=json.loads((args.output/"matching.jsonl").read_text().splitlines()[-1])
                    fit=None;score=dict(status="native_rejected",baseline=dict(weight=1.),
                        auxiliary=dict(weight=config["minimum_information_weight"]))
                    initial=inverse(frames[keys[j]["frame_id"]].t_wc)@frames[keys[i]["frame_id"]].t_wc
                    if service["status"]=="ok":
                        fit=native_fit(paths["native_binary"],rows[:,3:6],rows[:,6:8],config["camera"],
                            rows[:,0].astype(np.int64),initial,17,dest/"pnp")
                    confidence_start=time.perf_counter()
                    if fit and fit["applied"]:
                        try:score=confidence_pair(initial,np.asarray(fit["t_target_source"]),rows,keys[j]["depth"],config)
                        except Exception as error:
                            score.update(status="confidence_error",reason=str(error),traceback=traceback.format_exc())
                    confidence_wall=time.perf_counter()-confidence_start
                    save(dest/"confidence-before-fit.json",score)
                    candidate_aux=auxiliary+([(i,j,np.asarray(fit["t_target_source"]))] if fit and fit["applied"] else [])
                    candidate_weights=dict(weights);candidate_weights[j]=score
                    raw=np.array([frames[k["frame_id"]].t_wc for k in keys[:j+1]])
                    graph=dest/"prefix.graph";write_graph(graph,raw,candidate_aux,plan["graph"])
                    lines=[]
                    for row in graph.read_text().splitlines():
                        tok=row.split()
                        if tok and tok[0]=="E":
                            target,kind=int(tok[2]),int(tok[3])
                            w=candidate_weights[target]["auxiliary" if kind else "baseline"]["weight"] if target in candidate_weights else 1.
                            tok[-2]=format(float(tok[-2])/np.sqrt(w),".17g")
                            tok[-1]=format(float(tok[-1])/np.sqrt(w),".17g")
                            row=" ".join(tok)
                        lines.append(row)
                    # This graph is new, local to this request; no retained file is overwritten.
                    graph.write_text("\n".join(lines)+"\n")
                    graph_start=time.perf_counter()
                    run=subprocess.run([paths["confidence_binary"],str(graph),str(dest/"native"),"quadratic"],
                        capture_output=True,text=True,timeout=120)
                    (dest/"graph.log").write_text(run.stdout+run.stderr)
                    report=json.loads((dest/"native/report.json").read_text()) if (dest/"native/report.json").exists() else {}
                    if run.returncode or not report.get("usable"):raise ValueError("prefix graph unusable")
                    optimized=read_nodes(dest/"native/nodes.txt")
                    correction=check_pose(optimized[-1]@inverse(raw[-1]))
                    auxiliary=candidate_aux;weights=candidate_weights
                    result.update(status="ok",t_corrected_raw=correction.tolist(),matching_status=service["status"],
                        native_accepted=bool(fit and fit["applied"]),native_reason=fit["reason"] if fit else service.get("failure"),
                        confidence_status=score["status"],LK_weight=score["baseline"]["weight"],
                        auxiliary_weight=score["auxiliary"]["weight"],graph_nodes=len(raw),
                        max_constraint_frame=keys[j]["frame_id"],matching_wall_s=matching_wall,
                        native_PnP_wall_s=fit["wall_s"] if fit else 0.,
                        confidence_wall_s=confidence_wall,graph_wall_s=time.perf_counter()-graph_start,
                        solver=report,peak_GPU_allocated_bytes=service["peak_allocated_bytes"],
                        peak_GPU_reserved_bytes=service["peak_reserved_bytes"])
                except Exception as error:
                    result.update(reason=str(error),traceback=traceback.format_exc())
                result.update(worker_finish_s=time.perf_counter(),worker_wall_s=time.perf_counter()-start)
                log.write(json.dumps(result,allow_nan=False)+"\n");log.flush()
                try:stream.write((json.dumps(result,allow_nan=False)+"\n").encode())
                except (BrokenPipeError,ConnectionResetError):break
    finally:
        log.close();matcher.log.close();server.close();args.socket.unlink(missing_ok=True)
if __name__=="__main__":main()
