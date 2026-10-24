#!/usr/bin/env python3
"""Export one already frozen profile for a run without changing its parameters."""
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", choices=["default", "sparse", "prefetch"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse overwrite")
    parent = json.loads(args.manifest.read_text())
    for path, sha in parent["input_sha256"].items():
        if file_sha256(path) != sha:
            raise ValueError("frozen source changed")
    args.output.mkdir()
    studies, budgets = [], []
    own_source = Path(__file__).resolve()
    for study in parent["studies"]:
        if file_sha256(study["protocol"]) != study["protocol_sha256"]:
            raise ValueError("frozen study changed")
        original = json.loads(Path(study["protocol"]).read_text())
        selected = [x for x in original["arms"] if x["mode"] == args.profile]
        if len(selected) != 1:
            raise ValueError("profile missing or duplicated")
        plan = copy.deepcopy(original)
        plan["arms"] = selected
        calls, prefetch = selected[0]["request_cap"], selected[0]["prefetch_cap"]
        plan["budget"].update(max_matching_calls=calls, max_PnP_calls=calls, max_prefix_solves=calls,
                              normal_calls=calls, max_prefetch_tasks=prefetch)
        plan["input_sha256"][str(own_source)] = file_sha256(own_source)
        plan["input_sha256"][study["protocol"]] = study["protocol_sha256"]
        plan["scope"] = "Single explicit profile exported from the frozen validation inventory; no algorithm/resource changes"
        folder = args.output / study["scene"]
        folder.mkdir()
        protocol = folder / "protocol.json"
        save(protocol, plan)
        studies.append(dict(scene=study["scene"], protocol=str(protocol.resolve()), protocol_sha256=file_sha256(protocol)))
        budgets.append(dict(scene=study["scene"], matching=calls, prefetch=prefetch))
    hashes = dict(parent["input_sha256"])
    hashes[str(own_source)] = file_sha256(own_source)
    hashes[str(args.manifest.resolve())] = file_sha256(args.manifest)
    save(args.output / "manifest.json", dict(stage="Explicit selected auxiliary profile", profile=args.profile,
        studies=studies, budgets=budgets, input_sha256=hashes, parent_manifest_sha256=file_sha256(args.manifest),
        GT_read=False, correction_feedback=False, commit_push=False))
    print(json.dumps(dict(profile=args.profile, studies=len(studies), matching_cap=sum(x["matching"] for x in budgets))))


if __name__ == "__main__":
    main()
