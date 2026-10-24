#!/usr/bin/env python3
"""Evaluate every sealed timing-control arm without selecting a best repetition."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames
from evaluate_live_cpp_auxiliary import evaluate_group, mapped


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    return dict(n=len(a), mean=float(np.mean(a)), median=float(np.median(a)),
                P95=float(np.quantile(a, .95)), P99=float(np.quantile(a, .99)),
                max=float(np.max(a)))


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser()
    for name in ["protocol", "runs", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(args.protocol.read_text())
    sources = json.loads(Path(plan["sources"]).read_text())
    seal = json.loads((args.runs / "fit-seal.json").read_text())

    def check():
        for path, sha in seal.items():
            if file_sha256(args.runs / path) != sha:
                raise ValueError("sealed output changed: " + path)

    check()
    records, errors, failures, parity, timings, jobs = [], [], [], [], [], []
    sync_ms = 1000 * plan["budget"]["synchronization_wait_s"]
    for source in sources:
        scene = source["scene"]
        arms = [a for a in plan["arms"] if a["scene"] == scene]
        reference = Path(plan["raw_reference"])
        frames = load_frames(reference / "cache/frames.txt")
        baseline = mapped(reference / "raw.tum", frames)
        descriptions = dict(baseline_reference=reference / "raw.tum")
        for arm in arms:
            folder = args.runs / (scene + "-" + arm["mode"])
            descriptions[arm["mode"] + "_raw"] = folder / "raw.tum"
            descriptions[arm["mode"] + "_corrected"] = folder / "corrected.tum"
        evaluate_group(scene, "publication_development", frames, descriptions,
                       source["dataset"], records, errors, failures)
        for arm in arms:
            folder = args.runs / (scene + "-" + arm["mode"])
            if not (folder / "raw.tum").exists():
                failures.append(dict(scene=scene, mode=arm["mode"], reason="missing run"))
                continue
            raw = mapped(folder / "raw.tum", frames)
            own = load_frames(folder / "cache/frames.txt")
            emissions = lines(folder / "emissions.jsonl")
            final = folder / "cache/final.snapshot"
            parity.append(dict(scene=scene, mode=arm["mode"], profile=arm["profile"],
                frames=len(own), raw_frames=len(raw),
                raw_vs_reference_max_element=max((float(np.max(np.abs(t - baseline[i])))
                    for i, t in raw.items()), default=None),
                raw_vs_own_cache_max_element=max((float(np.max(np.abs(t - own[i].t_wc)))
                    for i, t in raw.items()), default=None),
                final_map_identical=file_sha256(final) == file_sha256(reference / "cache/final.snapshot"),
                final_map_sha256=file_sha256(final),
                frozen_keyframe_ids_identical=[r["frame_id"] for r in emissions if r["is_keyframe"]]
                    == [k["frame_id"] for k in source["keyframes"]]))
            with (folder / "cpp-stream.csv").open() as stream:
                cpp = {int(r["frame_id"]): r for r in csv.DictReader(stream)}
            for scope in ["all", "keyframe", "ordinary"]:
                selected = [r for r in emissions if scope == "all" or
                            bool(r["is_keyframe"]) == (scope == "keyframe")]
                fields = {
                    "rgb_to_adapter_ms": [1000 * (r["received_monotonic_s"] - r["due_monotonic_s"]) + sync_ms for r in selected],
                    "rgb_to_cpp_publication_ms": [1000 * (r["published_monotonic_s"] - r["due_monotonic_s"]) + sync_ms for r in selected],
                    "start_backlog_ms": [1000 * (r["start_monotonic_s"] - r["due_monotonic_s"]) for r in selected],
                    "full_frame_compute_ms": [float(cpp[r["frame_id"]]["tracking_compute_ms"]) for r in selected],
                    "cpp_to_adapter_ms": [1000 * (r["received_monotonic_s"] - r["published_monotonic_s"]) for r in selected],
                }
                for name in ["pose_ready_compute_ms", "backend_after_publish_ms"]:
                    if selected and name in cpp[selected[0]["frame_id"]]:
                        fields[name] = [float(cpp[r["frame_id"]][name]) for r in selected]
                for measure, values in fields.items():
                    timings.append(dict(scene=scene, mode=arm["mode"], profile=arm["profile"],
                                        scope=scope, measure=measure, **distribution(values)))
            jobs.extend(dict(mode=arm["mode"], profile=arm["profile"], **r)
                        for r in lines(folder / "worker/jobs.jsonl"))
    profile_medians = []
    for profile, scope, measure in sorted(set((r["profile"], r["scope"], r["measure"]) for r in timings)):
        rows = [r for r in timings if (r["profile"], r["scope"], r["measure"]) == (profile, scope, measure)]
        profile_medians.append(dict(profile=profile, scope=scope, measure=measure,
            repetitions=len(rows), median_of_run_mean=float(np.median([r["mean"] for r in rows])),
            median_of_run_P95=float(np.median([r["P95"] for r in rows])),
            median_of_run_max=float(np.median([r["max"] for r in rows])),
            worst_observed_max=max(r["max"] for r in rows)))
    args.output.mkdir()
    payload = dict(records=records, failures=failures, parity=parity, timings=timings,
        profile_medians=profile_medians, runtime=json.loads((args.runs / "status.json").read_text()),
        fit_seal_sha256=file_sha256(args.runs / "fit-seal.json"), scope=plan["scope"],
        time_origin="source RGB schedule including 20ms synchronization wait; no physical camera/network timestamp",
        limitations="two repetitions per profile on the development sequence; no independent generalization or hard realtime claim",
        correction_feedback=False, commit_push=False)
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    for name, rows in [("summary.csv", records), ("frame_errors.csv", errors),
                       ("runtime.csv", timings), ("jobs.csv", jobs)]:
        if rows:
            with (args.output / name).open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
                writer.writeheader()
                writer.writerows(rows)
    check()
    print(json.dumps(dict(failures=failures, parity=parity, profile_medians=profile_medians), indent=2))


if __name__ == "__main__":
    main()
