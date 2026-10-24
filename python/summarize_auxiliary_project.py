#!/usr/bin/env python3
"""Summarize every frozen study, including capped tails and all negative results."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from adaptivefusion.state import file_sha256
from evaluate_prefetch_auxiliary import distribution, rows


def main():
    parser = argparse.ArgumentParser()
    for name in ["manifest", "runs", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse overwrite")
    manifest = json.loads(args.manifest.read_text())
    records, parity, visibility, checks, failures, jobs_out, events_out, prefetch_out = [], [], [], [], [], [], [], []
    seals = []
    for study in manifest["studies"]:
        scene = study["scene"]
        plan = json.loads(Path(study["protocol"]).read_text())
        source = json.loads(Path(plan["sources"]).read_text())[0]
        folder = args.runs / scene
        seal = json.loads((folder / "runs/fit-seal.json").read_text())
        seals.append((folder / "runs", seal))
        for name, sha in seal.items():
            if file_sha256(folder / "runs" / name) != sha:
                raise ValueError("sealed independent output changed")
        evaluation = json.loads((folder / "evaluation/summary.json").read_text())
        parity.extend(evaluation["parity"])
        visibility.extend(dict(scene=scene, **x) for x in evaluation["visibility_audit"])
        failures.extend(evaluation["failures"])
        timestamps = {k["frame_id"]: k["timestamp"] for k in source["keyframes"]}
        matching = {}
        all_time_eligible, last = [], None
        for key in source["keyframes"][1:]:
            if last is None or key["timestamp"]-last >= plan["schedules"]["sparse250"]["minimum_source_interval_s"]:
                all_time_eligible.append(key["id"])
                last = key["timestamp"]
        for arm in plan["arms"]:
            mode = arm["mode"]
            root = folder / "runs" / (scene + "-" + mode)
            runtime = json.loads((root / "summary.json").read_text())
            jobs = rows(root / "worker/jobs.jsonl")
            attempts = rows(root / "worker/matching.jsonl")
            images = rows(root / "image-access.jsonl")
            events = rows(root / "events.jsonl")
            emissions = rows(root / "emissions.jsonl")
            requests = {x["request_id"]: x for x in rows(root / "requests.jsonl")}
            pf = rows(root / "worker/prefetch.jsonl")
            candidates = rows(root / "candidates.jsonl")
            publication = {x["frame_id"]: x["published_monotonic_s"] for x in emissions}
            epoch = next(publication[x["target_frame"]]-x["arrival_s"] for x in requests.values())
            age = [1000*(x["timestamp"]-timestamps[x["used_target_frame"]]) for x in emissions if x["used_target_frame"] >= 0]
            wait = [1000*(x["worker_start_s"]-epoch-requests[x["request_id"]]["arrival_s"]) for x in jobs]
            raw = next(x for x in evaluation["timings"] if x["mode"] == mode and x["scope"] == "all" and x["measure"] == "rgb_to_adapter_ms")
            eligible = plan["schedules"][arm["schedule"]]["eligible_keyframe_ids"]
            missing_for_cap = set(all_time_eligible)-set(eligible) if arm["schedule"] != "full" else set()
            records.append(dict(scene=scene, profile=arm["profile"], mode=mode,
                input_frames=runtime["expected_frames"], valid_frames=runtime["valid_frames"],
                matching_budget=arm["request_cap"], prefetch_budget=arm["prefetch_cap"],
                matching_calls=runtime["matching_calls"], delivered=runtime["delivered_jobs"],
                skipped_schedule=runtime["skipped_schedule"], skipped_by_64_cap=sum(x["target_keyframe"] in missing_for_cap for x in candidates),
                dropped_busy=runtime["dropped_busy_no_wait"], dropped_queue=runtime["dropped_queue_full"], unfinished=runtime["unfinished_at_stream_end"],
                prefetch_completed=runtime["prefetch_completed"], prefetch_skipped_busy=runtime["prefetch_skipped_busy"],
                prefetch_failed=runtime["prefetch_failed"], unfinished_prefetch=runtime["unfinished_prefetch"],
                matching_drops_during_prefetch=sum(x["status"] == "dropped_busy_no_wait" and x.get("busy_kind") == "prefetch" for x in events),
                CPU_s=runtime["CPU_usage"]["worker_CPU_s"]+runtime["CPU_usage"]["children_CPU_s"],
                prefetch_CPU_s=sum(x["worker_CPU_s"] for x in pf),
                delivery_median_ms=runtime["auxiliary_delivery_latency_ms"]["median"],
                delivery_P95_ms=runtime["auxiliary_delivery_latency_ms"]["P95"], delivery_max_ms=runtime["auxiliary_delivery_latency_ms"]["max"],
                output_P95_ms=raw["P95"], output_P99_ms=raw["P99"], output_max_ms=raw["max"],
                start_wait_P95_ms=distribution(wait)["P95"], extraction_mean_ms=distribution([1000*x["extraction_s"] for x in attempts])["mean"],
                correction_age_P95_ms=distribution(age)["P95"], correction_age_max_ms=max(age),
                frames_without_correction=len(emissions)-len(age),
                output_tail_after_last_budget_target_s=emissions[-1]["timestamp"]-source["keyframes"][eligible[-1]]["timestamp"],
                feature_cache_misses=sum(not x["cache_hit"] for x in images if x["status"] == "ok"),
                GPU_reserved_MiB=runtime["peak_GPU_reserved_MiB"],
                worker_returncode=runtime["worker_returncode"], producer_returncode=runtime["producer_returncode"],
                maximum_waiting=runtime["maximum_waiting_observed"], GT_read=False, absolute_accuracy_evaluated=False))
            by_frame = {x["target_frame"]: x for x in jobs}
            matching[mode] = {x["frame_id"]: (root / "worker/matching.matches" / f"{i:06d}.npz",
                root / "worker" / f"job-{by_frame[x['frame_id']]['target_keyframe']:04d}/pnp/fit.json",
                by_frame[x["frame_id"]]) for i, x in enumerate(attempts)}
            jobs_out.extend(dict(x, scene=scene, mode=mode) for x in jobs)
            prefetch_out.extend(dict(x, scene=scene, mode=mode) for x in pf)
            events_out.extend(dict(x, scene=scene, mode=mode) for x in events if x["status"] not in ["delivered", "dispatched"])
            failures.extend(dict(x, scene=scene, mode=mode) for x in jobs if x["status"] != "ok" or not x.get("native_accepted") or x.get("confidence_status") != "scored")
            failures.extend(dict(x, scene=scene, mode=mode) for x in pf if x["status"] != "ok")
        left, right = matching["sparse"], matching["prefetch"]
        common = sorted(set(left)&set(right))
        different, pose_diff, weight_diff = [], 0., 0.
        for frame in common:
            with np.load(left[frame][0]) as a, np.load(right[frame][0]) as b:
                if not all(np.array_equal(a[k], b[k]) for k in ["matches", "F_mask", "rows"]):
                    different.append(frame)
            a, b = (json.loads(x[frame][1].read_text()) for x in [left, right])
            pose_diff = max(pose_diff, float(np.max(np.abs(np.array(a["t_target_source"])-np.array(b["t_target_source"])))))
            weight_diff = max(weight_diff, *(abs(left[frame][2][k]-right[frame][2][k]) for k in ["LK_weight", "auxiliary_weight"]))
        checks.append(dict(scene=scene, common_completed_pairs=len(common), matching_inventory_identical=list(left) == list(right),
                           nonidentical_matching_frames=different, native_pose_max_element_difference=pose_diff, confidence_weight_max_difference=weight_diff))
    comparisons = []
    for scene in [x["scene"] for x in manifest["studies"]]:
        sparse = next(x for x in records if x["scene"] == scene and x["mode"] == "sparse")
        candidate = next(x for x in records if x["scene"] == scene and x["mode"] == "prefetch")
        comparisons.append(dict(scene=scene, **{key+"_change_percent": 100*(candidate[key]/sparse[key]-1)
            for key in ["delivery_median_ms", "delivery_P95_ms", "CPU_s", "output_P95_ms", "correction_age_P95_ms"]}))
    args.output.mkdir()
    summary = dict(records=records, comparisons=comparisons, parity=parity, visibility_audit=visibility,
        common_observation_checks=checks, failures=failures, manifest_sha256=file_sha256(args.manifest),
        scope="three recordings independent of RPY prefetch development; one run per profile per recording;64-call capped full-length outputs retained",
        GT_read=False, absolute_accuracy_evaluated=False, correction_feedback=False, commit_push=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, data in [("summary.csv", records), ("comparisons.csv", comparisons),
                       ("jobs.csv", jobs_out), ("events.csv", events_out), ("prefetch.csv", prefetch_out)]:
        if data:
            with (args.output / name).open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for x in data for k in x)))
                writer.writeheader()
                writer.writerows(data)
    for folder, seal in seals:
        for name, sha in seal.items():
            if file_sha256(folder / name) != sha:
                raise ValueError("sealed output changed during aggregation")
    print(json.dumps(dict(records=len(records), comparisons=comparisons, failures=len(failures)), indent=2))


if __name__ == "__main__":
    main()
