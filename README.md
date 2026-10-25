# AdaptiveFusion-SLAM

A C++17 RGB-D SLAM prototype with risk-adaptive mapping and an optional Python
auxiliary pose service. The native pipeline uses ORB, LK optical flow,
PnP-RANSAC and Ceres local bundle adjustment. The auxiliary service combines
RootSIFT, LightGlue, MAGSAC geometry verification and observation confidence
to produce a separate corrected trajectory.

## Core contributions and key gains

- **Original system — risk-adaptive RGB-D SLAM:** use temporal geometric-health
  signals to control keyframe and map updates. On TUM Freiburg1 XYZ (792 inputs),
  throughput rises **16.65 → 20.87 FPS (+25.4%)** and map points fall **19.8%**,
  with ATE RMSE **6.2081 → 6.2442 cm (+0.58%)**.
- **Added — learning-assisted trajectory correction:** combine
  RootSIFT + LightGlue + MAGSAC with native PnP and confidence-weighted pose
  constraints. Against native LK with risk control disabled, corrected-output
  ATE RMSE falls **24.4%–70.7% in four of five recordings**; floor improves
  **27.32 → 8.01 cm**.
- **Added — asynchronous keyframe assistance:** connect the Python worker to
  the C++ pose stream, retaining per-frame LK, dual trajectories and failure
  fallback. Across floor/plant/desk2, keyframe assistance uses **423 vs 2,970
  calls (85.8% fewer)** than the historical per-frame LightGlue + MAGSAC scheme.
- **Added — causal source-feature prefetch:** extract already-arrived source
  features only while the worker is idle. With unchanged low-CPU settings and
  call budgets, median auxiliary delivery latency falls **20.3% on plant** and
  **18.3% on desk2** relative to sparse assistance without prefetch.

These gains come from separate mapping, accuracy and latency evaluations;
auxiliary corrections remain independent outputs. [Full results, including
regressions](#measured-results), follow below.

## What it provides

- **Native RGB-D SLAM:** synchronized RGB/depth loading, frame tracking,
  keyframes, sparse landmarks, local BA, geometric health signals and TUM output.
- **Risk-adaptive mapping:** a temporal failure predictor controls feature,
  keyframe and map updates to reduce work under visual degradation.
- **Auxiliary pose correction:** low-frequency matching between frozen adjacent
  keyframes, using the existing native PnP estimator and confidence-weighted pose
  constraints while LK continues to process every frame.
- **Bounded auxiliary scheduling:** default, sparse and causal source-prefetch
  profiles, with explicit inputs, call budgets, failures and output freshness.

The auxiliary service receives the real C++ pose stream during dataset replay.
It writes `raw.tum` and `corrected.tum`; corrections are applied to the separate
output after receipt. Tracking, the map and local BA retain the original poses.

## Measured results

### Auxiliary trajectory output

ATE RMSE on common valid timestamps, in centimeters; lower is better. These
historical full-recording evaluations use the default auxiliary configuration
with the risk controller disabled. RPY and 360 are development recordings;
floor, plant and desk2 were held out from that service-development stage.

| TUM Freiburg1 sequence | Native LK baseline | Auxiliary output | ATE change |
|---|---:|---:|---:|
| RPY | 8.27 | 9.47 | +14.4% |
| 360 | 15.10 | 11.42 | −24.4% |
| floor | 27.32 | 8.01 | −70.7% |
| plant | 14.11 | 8.47 | −40.0% |
| desk2 | 13.73 | 7.86 | −42.7% |

The auxiliary output improves ATE in four of five evaluated recordings. The
RPY regression is retained. Keyframe assistance uses 423 calls across the three
held-out recordings versus 2,970 pair evaluations in the historical every-frame
comparison (85.8% fewer); this is a call-count reduction, not a frame-rate claim.
[Metrics and tradeoffs](docs/results.md) · [CSV](results/trajectory_summary.csv)

### Original risk-adaptive mapping

The original v0.1 XYZ evaluation processed all 792 RGB-D inputs: adaptive mode
reduced map points by 19.8% and mean frame time by 20.2%, with ATE RMSE changing
from 6.2081 to 6.2442 cm (+0.58%). Measured throughput was 16.65 → 20.87 FPS.
These are the original mapping results, separate from the auxiliary service.
[Original benchmark](results/stage_18_v0.1_release.md)

### Optional low-CPU scheduling

On the RPY development recording, source-prefetch reduced median auxiliary
handoff latency from 193.5 to 154.3 ms with the same 64-call budget. On three
recordings held out from prefetch development, median reductions were 5.0%,
20.3% and 18.3%; floor P95 increased 13.6%. Sparse budgets can leave a long
suffix without fresh corrections. The default profile remains available.
[Complete latency data](results/auxiliary_latency.csv)

## Build and run the native pipeline

Linux, a C++17 compiler, CMake ≥3.16, Eigen3, OpenCV ≥4.6 and Ceres ≥2.2 are
required. The published native evaluation used OpenCV 4.6.0 and Ceres 2.2.0.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure

# DATASET is a TUM RGB-D sequence directory.
./build/run_rgbd_odometry "$DATASET" "$NEW_OUTPUT/baseline.tum" 1000000

# Optional original risk controller.
./build/run_rgbd_odometry "$DATASET" "$NEW_OUTPUT/adaptive.tum" 1000000 \
    results/stage_16_simulated_risk_model.txt
```

Create a fresh output directory first. Supply an explicit frame limit instead
of `1000000` for a shorter run. Dataset associations use the existing native
loader. Data, weights, vendor sources and generated outputs stay outside Git.

## Run the auxiliary service

The optional service uses Python, PyTorch, SIFT/RootSIFT and LightGlue with a
CUDA GPU. The native tracking and graph components remain C++.

1. Install the Python dependencies and pinned LightGlue checkout.
2. Export an unmodified native baseline cache to freeze keyframe input IDs.
3. Prepare explicit data, weight, binary and CPU paths; choose a frozen profile.
4. Run the C++ stream and auxiliary worker together, retaining both trajectories.

[Setup and commands](docs/auxiliary_service.md) ·
[Configuration](configs/auxiliary.json) ·
[Evaluation and recovery](docs/reproducibility.md)

## Repository

| Path | Contents |
|---|---|
| `include/`, `src/`, `app/` | Native tracking, mapping, evaluation and pose tools |
| `python/adaptivefusion/` | Geometry, confidence, transport and evaluation helpers |
| `python/` | Auxiliary services and explicit freeze/select/run entry points |
| `configs/` | Portable frozen matching and resource configuration |
| `tests/` | Native and Python checks |
| `results/` | Compact measurements, protocols and representative failures |
| `docs/` | Usage, results and reproduction |

## Scope and references

The delivered extension provides independent pose output for recorded RGB-D
sequences. Its measured benefits concern trajectory accuracy, auxiliary work
and delivery latency. Corrected-map optimization, feedback into tracking/BA,
loop closure and a joint risk-controller-plus-auxiliary benchmark are future
integration work. Low-CPU/prefetch runs were evaluated without ground truth;
they do not establish a new absolute-accuracy result.

The implementation builds on [LightGlue](https://github.com/cvg/LightGlue),
[OpenCV](https://github.com/opencv/opencv),
[Ceres Solver](https://github.com/ceres-solver/ceres-solver) and
[TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset).
Upstream source and model terms apply to their respective components.
