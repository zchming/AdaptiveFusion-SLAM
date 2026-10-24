#!/usr/bin/env python3
"""Sealed latency, visibility and raw-state evaluation without ground truth."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from adaptivefusion.evaluation import read_tum
from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    return dict(n=len(a), mean=float(np.mean(a)), median=float(np.median(a)),
                P95=float(np.quantile(a, .95)), P99=float(np.quantile(a, .99)), max=float(np.max(a)))


def mapped(path, frames):
    times, poses = read_tum(path)
    inventory = np.array([f.timestamp for f in frames])
    out = {}
    for timestamp, pose in zip(times, poses):
        j = int(np.argmin(np.abs(inventory - timestamp)))
        if abs(inventory[j] - timestamp) > 1e-6 or j in out:
            raise ValueError("trajectory inventory mismatch")
        out[j] = pose
    return out


def main():
    parser = argparse.ArgumentParser()
    for name in ["protocol", "runs", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(args.protocol.read_text())
    sources = {x["scene"]: x for x in json.loads(Path(plan["sources"]).read_text())}
    seal = json.loads((args.runs / "fit-seal.json").read_text())

    def check():
        for path, sha in seal.items():
            if file_sha256(args.runs / path) != sha:
                raise ValueError("sealed output changed: " + path)

    check()
    parity, timings, visibility, failures, records = [], [], [], [], []
    for arm in plan["arms"]:
        scene, mode = arm["scene"], arm["mode"]
        folder = args.runs / (scene + "-" + mode)
        reference = Path(plan["raw_reference"])
        if not (folder / "cache/frames.txt").exists():
            failures.append(dict(mode=mode, reason="missing producer inventory"))
            continue
        frames = load_frames(folder / "cache/frames.txt")
        base_frames = load_frames(reference / "cache/frames.txt")
        raw, base = mapped(folder / "raw.tum", frames), mapped(reference / "raw.tum", base_frames)
        corrected = mapped(folder / "corrected.tum", frames)
        emissions = rows(folder / "emissions.jsonl")
        final = folder / "cache/final.snapshot"
        parity.append(dict(scene=scene, mode=mode, profile=arm["profile"], frames=len(frames),
            raw_frames=len(raw), corrected_frames=len(corrected),
            raw_vs_reference_max_element=max((float(np.max(np.abs(t - base[i]))) for i, t in raw.items()), default=None),
            raw_vs_own_cache_max_element=max((float(np.max(np.abs(t - frames[i].t_wc))) for i, t in raw.items()), default=None),
            final_map_identical=file_sha256(final) == file_sha256(reference / "cache/final.snapshot"),
            final_map_sha256=file_sha256(final), frozen_keyframe_ids_identical=
                [x["frame_id"] for x in emissions if x["is_keyframe"]] == [k["frame_id"] for k in sources[scene]["keyframes"]]))
        with (folder / "cpp-stream.csv").open() as stream:
            cpp = {int(x["frame_id"]): x for x in csv.DictReader(stream)}
        for scope in ["all", "keyframe", "ordinary"]:
            selected = [x for x in emissions if scope == "all" or bool(x["is_keyframe"]) == (scope == "keyframe")]
            values = {
                "rgb_to_adapter_ms": [1000 * (x["received_monotonic_s"] - x["due_monotonic_s"]) + 20 for x in selected],
                "rgb_to_cpp_publication_ms": [1000 * (x["published_monotonic_s"] - x["due_monotonic_s"]) + 20 for x in selected],
                "start_backlog_ms": [1000 * (x["start_monotonic_s"] - x["due_monotonic_s"]) for x in selected],
                "full_frame_compute_ms": [float(cpp[x["frame_id"]]["tracking_compute_ms"]) for x in selected],
                "cpp_to_adapter_ms": [1000 * (x["received_monotonic_s"] - x["published_monotonic_s"]) for x in selected],
            }
            for measure, data in values.items():
                timings.append(dict(scene=scene, mode=mode, profile=arm["profile"], scope=scope,
                                    measure=measure, **distribution(data)))
        accesses = rows(folder / "image-access.jsonl")
        source_by_frame = {k["frame_id"]: k for k in sources[scene]["keyframes"]}
        emission_by_frame = {x["frame_id"]: x for x in emissions}
        bad = [x for x in accesses if x["frame_id"] > x["visible_frame"] or x["timestamp"] > x["visible_timestamp"]
               or x["read_start_s"] < x["visible_published_monotonic_s"]]
        unknown = [x for x in accesses if x["frame_id"] not in source_by_frame
                   or source_by_frame[x["frame_id"]]["timestamp"] != x["timestamp"]]
        incorrect_watermarks = [x for x in accesses if x["visible_frame"] not in emission_by_frame
            or x["visible_timestamp"] != emission_by_frame[x["visible_frame"]]["timestamp"]
            or x["visible_published_monotonic_s"] != emission_by_frame[x["visible_frame"]]["published_monotonic_s"]]
        visibility.append(dict(mode=mode, accesses=len(accesses), future_accesses=len(bad),
            unknown_image_accesses=len(unknown), incorrect_live_watermarks=len(incorrect_watermarks),
            failed_feature_accesses=sum(x["status"] != "ok" for x in accesses),
            prefetch_accesses=sum(x["kind"] == "prefetch" for x in accesses), GT_read=False))
        records.append(dict(scene=scene, mode=mode, profile=arm["profile"], input_frames=len(frames),
                            raw_frames=len(raw), corrected_frames=len(corrected), GT_read=False,
                            absolute_accuracy_evaluated=False))
    args.output.mkdir()
    payload = dict(records=records, parity=parity, timings=timings, visibility_audit=visibility,
        failures=failures, runtime=json.loads((args.runs / "status.json").read_text()),
        fit_seal_sha256=file_sha256(args.runs / "fit-seal.json"), scope=plan["scope"],
        GT_read=False, absolute_accuracy_evaluated=False, correction_feedback=False, commit_push=False,
        time_origin="RGB source schedule including20ms wait; not physical camera/network latency")
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    for name, data in [("summary.csv", records), ("runtime.csv", timings)]:
        with (args.output / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for x in data for k in x)))
            writer.writeheader()
            writer.writerows(data)
    check()
    print(json.dumps(dict(failures=failures, visibility=visibility, parity=parity), indent=2))


if __name__ == "__main__":
    main()
