#!/usr/bin/env python3
"""One explicit manifest runs the frozen comparison and evaluates without GT."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from adaptivefusion.state import file_sha256


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(args.manifest.read_text())
    for path, sha in plan["input_sha256"].items():
        if file_sha256(path) != sha:
            raise ValueError("delivery source changed: " + path)
    for study in plan["studies"]:
        if file_sha256(study["protocol"]) != study["protocol_sha256"]:
            raise ValueError("study protocol changed")
    args.output.mkdir()
    status, start = [], time.perf_counter()
    scripts = Path(__file__).resolve().parent
    for study in plan["studies"]:
        folder = args.output / study["scene"]
        folder.mkdir()
        record = dict(scene=study["scene"], status="started")
        with (folder / "run.log").open("x") as log:
            cmd = [sys.executable, "-B", str(scripts / "run_prefetch_auxiliary_live.py"),
                   "--protocol", study["protocol"], "--output", str(folder / "runs")]
            record["run_returncode"] = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        if record["run_returncode"] == 0 and (folder / "runs/fit-seal.json").exists():
            with (folder / "evaluation.log").open("x") as log:
                cmd = [sys.executable, "-B", str(scripts / "evaluate_prefetch_auxiliary.py"),
                       "--protocol", study["protocol"], "--runs", str(folder / "runs"), "--output", str(folder / "evaluation")]
                record["evaluation_returncode"] = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
            arms = json.loads((folder / "runs/status.json").read_text())
            record["status"] = "complete" if record.get("evaluation_returncode") == 0 and all(
                x.get("status") == "complete" and x.get("worker_returncode") == 0 and x.get("producer_returncode") == 0 for x in arms) else "completed_with_failures"
        else:
            record["status"] = "failed"
        status.append(record)
        (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    save(args.output / "completion.json", dict(studies=status, wall_s=time.perf_counter()-start, GT_read=False,
        scope=plan["stage"], manifest_sha256=file_sha256(args.manifest), correction_feedback=False, commit_push=False))
    print("AUXILIARY_PROJECT_RUNS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
