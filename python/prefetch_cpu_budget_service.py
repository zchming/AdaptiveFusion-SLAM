#!/usr/bin/env python3
"""Same frozen CPU profiles with visibility-checked source prefetch and live matching."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu-profile", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    plan = json.loads(args.protocol.read_text())
    profile = plan["resource_profiles"][args.cpu_profile]
    folder = args.output.parent
    begin = time.perf_counter()
    cpu_begin = time.process_time()
    if profile["apply"]:
        wanted = set(profile["affinity"])
        if not wanted or not wanted.issubset(os.sched_getaffinity(0)):
            raise ValueError("frozen CPU affinity unavailable")
        os.sched_setaffinity(0, wanted)
        current_nice = os.nice(0)
        if current_nice > profile["nice"]:
            raise ValueError("cannot restore frozen worker priority")
        os.nice(profile["nice"] - current_nice)
        os.environ.update(profile["environment"])
    # Set resources before importing numerical libraries; the original matcher
    # initializes at eight threads, then the wrapper sets the request-time cap.
    import live_prefetch_auxiliary_service as service

    original = service.Matcher
    if profile["apply"]:
        import torch
        torch.set_num_interop_threads(profile["torch_interop_threads"])

    source=next(s for s in json.loads(Path(plan["sources"]).read_text()) if s["scene"]==sys.argv[sys.argv.index("--scene")+1])
    registry={str(Path(k["rgb"]).resolve()):(k["frame_id"],k["timestamp"]) for k in source["keyframes"]}

    class BudgetMatcher(original):
        def __init__(self, arguments):
            super().__init__(arguments)
            if profile["apply"]:
                self.cv2.setNumThreads(profile["opencv_threads"])
                self.torch.set_num_threads(profile["torch_threads"])
            actual = dict(profile=args.cpu_profile, affinity=sorted(os.sched_getaffinity(0)),
                nice=os.nice(0), opencv_threads=self.cv2.getNumThreads(),
                torch_threads=self.torch.get_num_threads(),
                torch_interop_threads=self.torch.get_num_interop_threads(),
                environment={k: os.environ.get(k) for k in plan["resource_profiles"]["budget2"]["environment"]},
                measurement_start_monotonic_s=begin, GT_read=False,
                child_resource_policy="native PnP and graph inherit affinity, nice and environment",
                affinity_scope="bounded worker concurrency; not exclusive cores or CPU time quota")
            for key in ["affinity", "nice", "opencv_threads", "torch_threads"]:
                if actual[key] != profile[key]:
                    raise ValueError("frozen worker resource mismatch: " + key)
            save(folder / "worker-resources.json", actual)

        def feature(self, path):
            begin=time.perf_counter();cpu_begin=time.process_time();hit=path in self.features
            frame,timestamp=registry[str(Path(path).resolve())]
            record=dict(request_id=self.read_request_id,kind=self.read_kind,frame_id=frame,timestamp=timestamp,
                visible_frame=self.visible_frame,visible_timestamp=self.visible_timestamp,
                visible_published_monotonic_s=self.visible_published_s,read_start_s=begin,cache_hit=hit,status="failed")
            try:
                if frame>self.visible_frame or timestamp>self.visible_timestamp or begin<self.visible_published_s:
                    raise ValueError("future image access blocked")
                result=super().feature(path)
                record.update(status="ok",features=len(result[0]["keypoints"]),reported_extraction_s=result[1])
                return result
            finally:
                record.update(wall_s=time.perf_counter()-begin,CPU_s=time.process_time()-cpu_begin)
                with (folder/"image-access.jsonl").open("a") as stream:
                    stream.write(json.dumps(record,allow_nan=False)+"\n")

        def request(self, line):
            cpu0, wall0, index = time.process_time(), time.perf_counter(), self.requests
            try:
                return super().request(line)
            finally:
                value = dict(request_index=index, worker_CPU_s=time.process_time() - cpu0,
                    wall_s=time.perf_counter() - wall0, affinity=sorted(os.sched_getaffinity(0)),
                    opencv_threads=self.cv2.getNumThreads(), torch_threads=self.torch.get_num_threads())
                with (folder / "matcher-cpu.jsonl").open("a") as stream:
                    stream.write(json.dumps(value, allow_nan=False) + "\n")

    service.Matcher = BudgetMatcher
    # Only remove the wrapper option; the original service parses every
    # original argument and still owns all matching, PnP and graph computation.
    index = sys.argv.index("--cpu-profile")
    del sys.argv[index:index + 2]
    try:
        service.main()
    finally:
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        own = resource.getrusage(resource.RUSAGE_SELF)
        save(folder / "worker-cpu-total.json", dict(profile=args.cpu_profile,
            wall_s=time.perf_counter() - begin, worker_CPU_s=time.process_time() - cpu_begin,
            children_CPU_s=children.ru_utime + children.ru_stime,
            worker_peak_RSS_KiB=own.ru_maxrss, voluntary_switches=own.ru_nvcsw,
            involuntary_switches=own.ru_nivcsw, affinity=sorted(os.sched_getaffinity(0)),
            nice=os.nice(0), GT_read=False))


if __name__ == "__main__":
    main()
