# Auxiliary pose service

The native runner tracks each RGB-D frame. An independent Python worker matches
frozen adjacent keyframes with RootSIFT + LightGlue, verifies 2D geometry with
MAGSAC, reuses native PnP and combines LK/auxiliary constraints using the frozen
observation-confidence rule. This produces a causal corrected output alongside
the unchanged native trajectory.

## Dependencies

Build the repository as described in the README. The auxiliary build targets
are `run_streaming_odometry`, `native_pnp_pair` and
`offline_confidence_pose_graph`; `AFS_BUILD_AUXILIARY=OFF` disables them.
The stream observer is generated inside the build directory. It does not edit
the production tracking source.

Use a Python environment with a CUDA-enabled PyTorch build appropriate for the
machine. Install `python/requirements-auxiliary.txt`, then obtain the pinned
LightGlue source outside this repository:

```bash
python -m pip install -r python/requirements-auxiliary.txt
git clone https://github.com/cvg/LightGlue.git "$LIGHTGLUE_SOURCE"
git -C "$LIGHTGLUE_SOURCE" checkout eb42fee2d71449efb0aa5c10549752b5d75384d8
python -m pip install -e "$LIGHTGLUE_SOURCE"
```

Provide the official `sift_lightglue_v0-1_arxiv.pth` checkpoint explicitly as
`LIGHTGLUE_WEIGHTS`; the expected SHA256 is
`5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0`.
The source, weights and outputs are not bundled. Tested versions are recorded
in [provenance](../results/provenance.json).

## Prepare inputs

Set `DATASET` to a TUM RGB-D directory, `BUILD` to the absolute build directory,
`NATIVE_LIB` to the directory containing the native OpenCV/Ceres shared
libraries, and `AUX_CPUS` to two available logical CPU IDs. Use a new `WORK`
directory with an existing parent. These choices are written into the local
manifest rather than hard-coded into the repository.

```bash
mkdir "$WORK"
"$BUILD/run_rgbd_odometry" "$DATASET" "$WORK/baseline.tum" 1000000 \
    - none "$WORK/cache" --export-keyframe-features

python -B python/prepare_auxiliary_run.py \
    --dataset "$DATASET" --reference-cache "$WORK/cache" \
    --binaries "$BUILD" --lightglue-source "$LIGHTGLUE_SOURCE" \
    --weights "$LIGHTGLUE_WEIGHTS" --library-path "$NATIVE_LIB" \
    --aux-cpus "$AUX_CPUS" --scene sequence --output "$WORK/prepared"
```

The baseline export determines input IDs and the reference used for subsequent
raw-output comparison. The worker receives only the live C++ prefix during the
actual run. No ground truth enters preparation, matching or correction.

The supplied camera `[517.3, 516.5, 318.6, 255.3]` and depth scale `5000` match
the Freiburg1 evaluation setup. Other camera models need an explicit new
configuration and a matching native calibration; they are not validated by
these results. A partial baseline export limits replay to its exported frame
count. Export at least two keyframes.

## Choose one profile and run

```bash
python -B python/select_auxiliary_profile.py \
    --manifest "$WORK/prepared/frozen/manifest.json" \
    --profile default --output "$WORK/selected"
python -B python/run_auxiliary_project.py \
    --manifest "$WORK/selected/manifest.json" --output "$WORK/run"
```

All preparation and run directories must be new. Existing outputs are refused.
For a complete three-profile comparison, run the prepared manifest directly.
Use `summarize_auxiliary_project.py --manifest ... --runs ... --output ...`
for its combined table. This aggregator expects the three-profile comparison;
single-profile runs already contain their own evaluation output.

| Profile | CPU and scheduling | Intended use |
|---|---|---|
| `default` | 8 worker threads, all frozen adjacent keyframe pairs, one waiting request | Configuration with historical absolute-accuracy evidence |
| `sparse` | 2 worker threads on the selected CPUs, ≥250 ms spacing, ≤64 calls, no waiting | Lower auxiliary CPU use with capped coverage |
| `prefetch` | Same sparse budget plus idle extraction of an already-published source frame | Optional reduction in median handoff latency |

CPU affinity limits concurrency, not exclusive cores or a hard CPU-time quota.
The two low-CPU profiles inherit the configured priority and one-thread BLAS
settings. Matching, confidence and graph parameters stay fixed across profiles.

## Outputs and failure behavior

Each arm retains `raw.tum`, `corrected.tum`, `cache/final.snapshot`, stream
latency, worker events and every skipped, rejected, unfinished or failed call.
The map snapshot uses the raw trajectory coordinate system. Do not combine it
with corrected camera poses as though the map had also been optimized.

A correction applies to future output frames after receipt; old output rows are
immutable. Service failures leave the native trajectory available. Busy
prefetch opportunities are skipped without retries. A failed prefetch yields
no correction; later matching can extract source features on demand. The sparse
budget is not replenished, so correction age can grow after the final call.
