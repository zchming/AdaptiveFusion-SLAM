#!/usr/bin/env python3
"""Freeze explicit source inventories and references for the delivered profiles."""
import argparse
import copy
import json
from pathlib import Path

from adaptivefusion.state import file_sha256


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    for name in ["template", "sources", "references", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse overwrite")
    template = json.loads(args.template.read_text())
    sources = json.loads(args.sources.read_text())
    references = json.loads(args.references.read_text())
    repo = Path(__file__).resolve().parent.parent
    helpers = [repo / "python" / name for name in ["freeze_auxiliary_project.py", "run_auxiliary_project.py"]]
    shared = {Path(x) for x in template["input_sha256"] if not x.endswith(".png")}
    shared.update(helpers)
    shared.add(args.template.resolve())
    args.output.mkdir()
    studies, caps = [], []
    for index, source in enumerate(sources):
        scene = source["scene"]
        keys = source["keyframes"]
        if [x["id"] for x in keys] != list(range(len(keys))):
            raise ValueError("keyframe identity inventory must be sequential")
        selected, last = [], None
        interval = template["schedules"]["sparse250"]["minimum_source_interval_s"]
        limit = template["prefetch"]["maximum_attempts_per_run"]
        eligible_all = []
        for key in keys[1:]:
            if last is None or key["timestamp"] - last >= interval:
                eligible_all.append(key["id"])
                last = key["timestamp"]
        selected = eligible_all[:limit]
        if not selected:
            raise ValueError("no eligible auxiliary observations")
        folder = args.output / scene
        folder.mkdir()
        source_path = folder / "sources.json"
        save(source_path, [source])
        plan = copy.deepcopy(template)
        plan.pop("input_sha256")
        plan.update(stage="Frozen auxiliary service comparison", scope="Explicit recording prefix, three frozen profiles; no GT; capped resource comparison",
                    sources=str(source_path.resolve()), frame_counts={scene: references[scene]["frames"]},
                    selected_sequences=[scene], raw_reference=references[scene]["root"],
                    parent_reference="Frozen matching, confidence and64-call upper bound unchanged",
                    evaluation="all three runs sealed; no GT; retain raw/map, visibility, latency, CPU, full capped tail and all failures")
        plan["profiles"] = dict(default="unchanged default CPU, all frozen keyframe pairs",
            sparse250="V42 low CPU, first at most64 eligible250ms positions, no waiting",
            prefetch250="same sparse positions/CPU plus at most one causal idle source prefetch per eligible target")
        plan["schedules"]["full"]["eligible_keyframe_ids"] = list(range(1, len(keys)))
        plan["schedules"]["sparse250"]["eligible_keyframe_ids"] = selected
        plan["prefetch"]["eligible_for_target_keyframe_ids"] = selected
        descriptions = [dict(mode="default", profile="default", resource_profile="default", schedule="full", request_cap=len(keys)-1, prefetch_cap=0),
                        dict(mode="sparse", profile="sparse250", resource_profile="budget2", schedule="sparse250", request_cap=len(selected), prefetch_cap=0),
                        dict(mode="prefetch", profile="prefetch250", resource_profile="budget2", schedule="sparse250", request_cap=len(selected), prefetch_cap=len(selected))]
        rotation = index % 3
        plan["arms"] = [dict(scene=scene, max_frames=references[scene]["frames"], **x) for x in descriptions[rotation:] + descriptions[:rotation]]
        call_cap = len(keys) - 1 + 2 * len(selected)
        plan["budget"].update(max_matching_calls=call_cap, max_PnP_calls=call_cap, max_prefix_solves=call_cap,
            normal_calls=call_cap, max_prefetch_tasks=len(selected), requests_per_scene={scene: dict(default=len(keys)-1, sparse=len(selected), prefetch=len(selected))})
        plan["independent_scope"] = "caller-supplied recording; evaluation independence must be documented separately"
        plan["budget_tail"] = dict(sparse_upper_bound=limit, all_time_eligible=len(eligible_all), admitted_frozen_positions=len(selected),
            last_frozen_target_timestamp=keys[selected[-1]]["timestamp"], last_recording_keyframe_timestamp=keys[-1]["timestamp"],
            behavior="no replenishment after cap; original and corrected outputs continue to end; old correction age retained")
        paths = set(shared)
        paths.update([source_path.resolve(), args.sources.resolve(), args.references.resolve()])
        dataset = Path(source["dataset"])
        paths.update(dataset.glob("rgb/*.png"))
        paths.update(dataset.glob("depth/*.png"))
        for name in ["rgb.txt", "depth.txt", "associations.txt"]:
            if (dataset / name).exists():
                paths.add(dataset / name)
        reference = Path(references[scene]["root"])
        paths.update([reference / "raw.tum", reference / "cache/frames.txt", reference / "cache/final.snapshot"])
        if any("groundtruth" in str(x) for x in paths):
            raise ValueError("truth must not enter this manifest")
        plan["input_sha256"] = {str(x): file_sha256(x) for x in sorted(paths)}
        protocol = folder / "protocol.json"
        save(protocol, plan)
        studies.append(dict(scene=scene, protocol=str(protocol.resolve()), protocol_sha256=file_sha256(protocol)))
        caps.append(dict(scene=scene, matching=call_cap, prefetch=len(selected), frames=3 * references[scene]["frames"], **plan["budget_tail"]))
    save(args.output / "manifest.json", dict(stage="Frozen auxiliary service replay", studies=studies, budgets=caps,
        template_sha256=file_sha256(args.template), GT_read=False, correction_feedback=False, commit_push=False,
        input_sha256={str(x): file_sha256(x) for x in helpers},
        rule="one run per profile per recording; rotate profile order; frozen64-call upper bound, no tuning or retries"))
    print(json.dumps(dict(studies=len(studies), matching_cap=sum(x["matching"] for x in caps),
                          prefetch_cap=sum(x["prefetch"] for x in caps), frame_outputs=sum(x["frames"] for x in caps)), indent=2))


if __name__ == "__main__":
    main()
