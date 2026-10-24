#!/usr/bin/env python3
"""Create a path-explicit frozen run from an exported native baseline cache."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from adaptivefusion.state import file_sha256
from adaptivefusion.trajectory import load_frames, write_tum


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    for name in ["dataset", "reference-cache", "binaries", "lightglue-source", "weights", "library-path", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--aux-cpus", required=True, help="two available logical CPU IDs, e.g.20,22")
    parser.add_argument("--scene", default="sequence")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse overwrite")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.scene):
        raise ValueError("scene must be one simple identifier")
    cpus = [int(x) for x in args.aux_cpus.split(",")]
    if len(set(cpus)) != 2 or len(cpus) != 2 or not set(cpus).issubset(os.sched_getaffinity(0)):
        raise ValueError("choose two available logical CPUs")
    repo = Path(__file__).resolve().parent.parent
    dataset, cache = args.dataset.resolve(), args.reference_cache.resolve()
    frames = load_frames(cache / "frames.txt")
    keys = []
    for line in (cache / "final.snapshot").read_text().splitlines():
        if not line.startswith("K "):
            continue
        values = line.split()
        kid, frame_id, timestamp = int(values[1]), int(values[2]), float(values[3])
        with (cache / "features" / f"{kid}.features").open() as stream:
            head = [next(stream) for _ in range(8)]
        meta, images = shlex.split(head[2]), shlex.split(head[7])
        if (int(meta[1]), int(meta[2]), float(meta[3])) != (kid, frame_id, timestamp) or not frames[frame_id].valid:
            raise ValueError("native keyframe metadata differs from raw frame inventory")
        keys.append(dict(id=kid, frame_id=frame_id, timestamp=timestamp, depth_timestamp=float(meta[4]),
                         rgb=str(dataset / images[1]), depth=str(dataset / images[2])))
    if len(keys) < 2:
        raise ValueError("export at least two keyframes")
    binaries = args.binaries.resolve()
    required = [binaries / name for name in ["run_streaming_odometry", "native_pnp_pair", "offline_confidence_pose_graph"]]
    for binary in required:
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("missing native binary: " + str(binary))
    template_path = repo / "configs/auxiliary.json"
    template = json.loads(template_path.read_text())
    template["producer_binary"] = str(required[0])
    template["paths"] = dict(native_binary=str(required[1]), confidence_binary=str(required[2]),
        lightglue_source=str(args.lightglue_source.resolve()), weights=str(args.weights.resolve()),
        native_library_path=str(args.library_path.resolve()))
    template["resource_profiles"]["default"].update(affinity=sorted(os.sched_getaffinity(0)), nice=os.nice(0))
    template["resource_profiles"]["budget2"].update(affinity=sorted(cpus), nice=min(19, os.nice(0) + 5))
    paths = list((repo / "python").glob("*.py")) + list((repo / "python/adaptivefusion").glob("*.py"))
    paths += required + [template_path, args.weights.resolve()]
    paths += [args.lightglue_source.resolve() / "lightglue" / name for name in ["sift.py", "lightglue.py", "utils.py"]]
    template["input_sha256"] = {str(x): file_sha256(x) for x in paths}
    args.output.mkdir()
    reference = args.output / "reference"
    (reference / "cache").mkdir(parents=True)
    for name in ["frames.txt", "final.snapshot"]:
        shutil.copyfile(cache / name, reference / "cache" / name)
    write_tum(reference / "raw.tum", [(f.timestamp, f.t_wc) for f in frames if f.valid])
    sources = [dict(scene=args.scene, dataset=str(dataset), keyframes=keys,
                    baseline_frames=str(reference.resolve() / "cache/frames.txt"), request_budget=len(keys)-1)]
    save(args.output / "sources.json", sources)
    save(args.output / "references.json", {args.scene: dict(root=str(reference.resolve()), frames=len(frames))})
    save(args.output / "template.json", template)
    subprocess.run([sys.executable, "-B", str(repo / "python/freeze_auxiliary_project.py"),
        "--template", str(args.output / "template.json"), "--sources", str(args.output / "sources.json"),
        "--references", str(args.output / "references.json"), "--output", str(args.output / "frozen")], check=True)
    print(str((args.output / "frozen/manifest.json").resolve()))


if __name__ == "__main__":
    main()
