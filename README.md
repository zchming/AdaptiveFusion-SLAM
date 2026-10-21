# AdaptiveFusion-SLAM

> Current engineering focus: **static-scene RGB-D geometry**, preserving the
> C++ frontend and adding a substantive Python offline optimization backend.
> Offline ATE gains on desk/room and safe rejection of disconnected teddy remain
> the supported behavior. The [experimental component subset](#experimental-anchor-component-subset)
> fails accuracy; a new opt-in [bridge guard](#python-geometric-bridge-guard)
> rejects its known teddy degeneracy while retaining desk/room outputs exactly.
> Safety acceptance passes, but teddy accuracy is not improved. A subsequent
> [cached-feature bridge audit](#cached-feature-bridge-candidates) produces no
> accepted new constraints; the subsequent [short-window identity audit](#short-window-landmark-identity-audit)
> also lacks sufficient support. [Ordinary-frame transport](#ordinary-frame-track-comparison)
> increases raw correspondence counts but still supplies no validated constraints.
> A [candidate-pool repair](#candidate-observation-pool-and-minimal-feature-export-proposal)
> recovers cached singletons, but its frozen comparison still fails geometry.
> The strict default
> remains unchanged. Online tracking benefit, reliable cross-scene operation and
> dynamic/4D remain unproven or unimplemented.

**Proactive Tracking-Failure Forecasting and Risk-Adaptive Map Updating for Robust RGB-D SLAM**

AdaptiveFusion-SLAM is a research-oriented RGB-D visual SLAM system designed to predict tracking degradation before complete failure and proactively protect the map from unreliable observations.

The project is implemented incrementally from a minimal SLAM pipeline instead of directly modifying an existing complete system. Its long-term goal is to provide a reproducible platform for studying tracking-failure forecasting, risk-aware frontend control, and selective map updating.

> Current status: v0.1 research prototype. The complete causal RGB-D pipeline,
> risk-adaptive policy, trajectory evaluation, runtime profiling, and controlled
> degradation benchmark are runnable. The predictor is still trained on
> synthetic episodes, so the research hypothesis is not yet validated across
> real datasets.

## Quantitative Results

Full TUM `rgbd_dataset_freiburg1_xyz` sequence:

| Metric | Baseline | Adaptive | Change |
|---|---:|---:|---:|
| Valid poses | 792 | 792 | maintained |
| Tracking success | 100% | 100% | maintained |
| ATE RMSE | 0.062081 m | 0.062442 m | +0.58% |
| RPE translation | 0.005518 m | 0.005537 m | +0.34% |
| Keyframes | 114 | 92 | -19.3% |
| Map points | 79,790 | 63,983 | -19.8% |
| Local BA runs | 113 | 90 | -20.4% |
| Mean frame time | 60.07 ms | 47.91 ms | -20.2% |
| Throughput | 16.65 FPS | 20.87 FPS | +25.4% |

Five controlled RGB degradation experiments are included: motion blur, low
light, central occlusion, Gaussian noise, and every-tenth-frame loss.

| Degradation | Mode | Success | ATE (m) | Keyframes | Map points | FPS |
|---|---|---:|---:|---:|---:|---:|
| Blur | baseline | 100% | 0.032742 | 17 | 10,046 | 14.66 |
| Blur | adaptive | 100% | 0.032108 | 17 | 9,252 | 14.24 |
| Dark | baseline | 100% | 0.025468 | 16 | 10,046 | 25.53 |
| Dark | adaptive | 100% | 0.025000 | 17 | 10,099 | 24.80 |
| Occlusion | baseline | 100% | 0.052648 | 22 | 17,623 | 23.70 |
| Occlusion | adaptive | 100% | 0.054455 | 20 | 15,289 | 24.78 |
| Noise | baseline | 100% | 0.025805 | 17 | 12,625 | 18.26 |
| Noise | adaptive | 100% | 0.026300 | 18 | 11,933 | 17.85 |
| Drop | baseline | 90% | 0.024499 | 17 | 12,015 | 23.38 |
| Drop | adaptive | 90% | 0.023557 | 16 | 10,710 | 23.55 |

See the Chinese resume-ready description, the
[full v0.1 report](results/stage_18_v0.1_release.md), and the
[machine-readable experiment table](results/stage_18_degradation_comparison.csv).

## Motivation

Conventional visual SLAM systems usually react after tracking quality has already deteriorated:

- the number of valid matches becomes insufficient;
- pose estimation fails;
- the tracker enters a lost state;
- unreliable observations have already contaminated the map.

AdaptiveFusion-SLAM investigates a proactive alternative:

> Can recent geometric health signals be used to predict tracking failure several frames in advance, allowing the system to intervene before unreliable observations damage the trajectory and map?

## Research Hypothesis

Compared with reactive decisions based only on the current frame, short-horizon failure forecasting from temporal geometric health signals can:

- reduce the number of tracking failures;
- prevent unreliable map updates;
- improve trajectory accuracy and recovery performance;
- retain real-time execution.

## Proposed Method

### 1. Geometric Health Monitoring

For every frame, the system records a geometric health vector:

\[
h_t =
[
r_{\text{inlier}},
e_{\text{reproj}},
e_{\text{fb}},
H_{\text{spatial}},
p_{\text{parallax}},
r_{\text{depth}},
\kappa
]
\]

where:

- \(r_{\text{inlier}}\): RANSAC inlier ratio;
- \(e_{\text{reproj}}\): reprojection error;
- \(e_{\text{fb}}\): forward-backward optical-flow error;
- \(H_{\text{spatial}}\): spatial coverage of tracked features;
- \(p_{\text{parallax}}\): frame-to-frame parallax;
- \(r_{\text{depth}}\): valid-depth ratio;
- \(\kappa\): geometric conditioning indicator.

### 2. Short-Horizon Failure Forecasting

Instead of examining only the current frame, the system uses a temporal window of recent health vectors to estimate:

\[
P(F_{t+1:t+H}=1 \mid h_{t-L+1:t})
\]

This represents the probability that tracking will fail within the next \(H\) frames, based on the previous \(L\) frames.

Failure labels will be generated automatically from future tracking states and trajectory errors, avoiding manual frame-by-frame annotation.

### 3. Risk-Adaptive Intervention

The predicted failure risk controls the frontend and mapping policies:

| Risk level | Planned intervention |
|---|---|
| Low | Continue efficient LK optical-flow tracking and normal map updates |
| Medium | Add local ORB verification and evaluate early keyframe insertion |
| High | Perform global ORB redetection and suspend unreliable map-point insertion |
| Critical | Preserve the latest trusted map state and initiate recovery from a reference keyframe |

The main purpose is not merely to report that tracking is unreliable, but to prevent uncertain observations from corrupting the map.

## System Pipeline

```text
RGB-D image sequence
        |
        v
Camera model and dataset synchronization
        |
        v
ORB features + LK optical-flow tracking
        |
        v
Pose estimation using RANSAC and PnP
        |
        v
Per-frame geometric health vector
        |
        v
Temporal tracking-failure predictor
        |
        v
Risk-adaptive frontend and map-update policy
        |
        v
Keyframes, map points, and local bundle adjustment
        |
        v
Trajectory, sparse map, and evaluation results
```

## Evaluation Plan

The system will be evaluated on real RGB-D sequences, including:

- TUM RGB-D;
- Bonn RGB-D Dynamic;
- OpenLORIS-Scene.

Controlled degradation experiments will additionally study:

- motion blur;
- reduced illumination;
- partial occlusion;
- dropped frames;
- image noise.

### SLAM Metrics

- Absolute Trajectory Error (ATE);
- Relative Pose Error (RPE);
- tracking success rate;
- number of tracking failures;
- recovery time;
- runtime per frame and FPS.

### Forecasting Metrics

- AUROC and AUPRC;
- false-positive and false-negative rates;
- Brier score;
- expected calibration error;
- average failure-warning lead time;
- cross-dataset generalization.

## Planned Ablation Study

The final evaluation will compare:

1. baseline RGB-D SLAM;
2. reactive current-frame thresholding;
3. temporal failure forecasting without intervention;
4. forecasting with adaptive frontend control;
5. forecasting with selective map updating;
6. the complete AdaptiveFusion-SLAM system.

## Build

The project currently requires:

- Ubuntu 24.04
- C++17
- CMake
- Eigen3
- OpenCV 4.6
- Ceres Solver 2.2

OpenCV is used for RGB-D image loading, ORB features, descriptor matching, and
LK optical flow. Ceres performs local bundle adjustment. g2o and additional
dependencies will be enabled when their corresponding modules are implemented.

```bash
cmake -S . -B build
cmake --build build
./build/run_slam
ctest --test-dir build --output-on-failure
```

Expected output at the current stage:

```text
Original pixel: 320 240
3D point in camera coordinates: 0.00541272 -0.0592449 2
Projected pixel: 320 240
Round-trip error: 0
Camera model test passed.
```

Inspect a TUM RGB-D sequence by passing the directory that contains `rgb.txt`
and `depth.txt`:

```bash
./build/inspect_tum_dataset /path/to/tum_sequence
```

The optional second argument sets the maximum RGB-to-depth timestamp difference
in seconds. Its default value is `0.02`.

Run the current feature frontend on two consecutive synchronized RGB-D frames:

```bash
./build/run_feature_frontend /path/to/tum_sequence 0
```

The final argument is the zero-based index of the first frame. The program
loads that frame and the following frame, extracts ORB features, and reports
ORB matches, LK tracks, and valid metric RGB-D correspondences. The current
demo uses the TUM Freiburg 1 RGB intrinsics `(517.3, 516.5, 318.6, 255.3)` and
the TUM depth scale `5000`; other sequences require their own calibration.

Run the current RGB-D visual odometry loop and write a TUM-format trajectory:

```bash
./build/run_rgbd_odometry /path/to/tum_sequence trajectory.txt
```

An optional third argument limits the number of processed frames. Only
initialized or successfully tracked frames enter the trajectory; lost frames
do not replace the latest trusted reference frame.

After training or obtaining a compatible risk model, enable the causal online
adaptive loop with:

```bash
./build/run_rgbd_odometry \
    /path/to/tum_sequence trajectory.txt 1000 risk_model.txt
```

Set `1000` to the desired maximum frame count; omitting the model keeps baseline
mode. The runner additionally writes `.health.csv`, `.failure_dataset.csv`,
`.risk.csv`, and `.summary.csv` files beside the trajectory. The summary
contains tracking failures and recoveries, map size, runtime percentiles, and
FPS. Add an optional degradation mode for a reproducible stress test:

```bash
./build/run_rgbd_odometry \
    /path/to/tum_sequence degraded.txt 200 - blur
```

The supported modes are `none`, `blur`, `dark`, `occlusion`, `noise`, and
`drop`. Passing `-` keeps baseline control while retaining the fifth argument
position.

## Current Runnable Pipeline

The implemented frontend currently connects these modules:

```text
TUM rgb.txt + depth.txt
        |
        v
One-to-one timestamp synchronization
        |
        v
RGB image + 16-bit depth image loading
        |
        v
Grayscale conversion and ORB extraction in both frames
        |
        +---------------------------+
        |                           |
        v                           v
Hamming descriptor matching   Pyramidal LK optical flow
        |                           |
        v                           v
Ratio + mutual filtering      Forward-backward filtering
        |                           |
        +-------------+-------------+
                      |
                      v
         Accepted 2D-to-2D correspondences
                      |
                      v
      First-frame 16-bit depth validation
                      |
                      v
       Raw depth / 5000 = metric depth
                      |
                      v
          Camera::pixelToCamera()
                      |
                      v
         Metric 3D-to-2D correspondences
                      |
                      v
               PnP + RANSAC
                      |
                      v
       Inlier-refined relative camera pose
                      |
                      v
     Inlier ratio + mean reprojection error
                      |
                      v
      Invert and accumulate relative pose
                      |
                      v
        World-from-camera trajectory
                      |
                      v
            TUM trajectory file
                      |
                      v
     Trusted-pose keyframe selection
                      |
                      v
 Valid-depth features transformed to world
                      |
                      v
       Sparse map points + observations
```

The camera projection model is now connected to LK tracks through validated
depth measurements. PnP/RANSAC now converts the metric correspondences into a
relative rotation and translation while rejecting geometric outliers.
The odometry loop keeps a trusted reference frame, uses LK as its normal path,
falls back to ORB matching after LK/PnP failure, and excludes lost frames from
the accumulated trajectory.
Trusted poses may become keyframes after minimum spacing and motion checks.
Only their valid-depth ORB features create metric world map points; invalid
frames cannot enter the sparse map.

## Project Structure

```text
AdaptiveFusion-SLAM/
├── app/
├── include/
├── src/
├── config/
├── evaluation/
├── experiments/
├── docs/
├── results/
├── CMakeLists.txt
└── README.md
```

The structure will grow incrementally as each module becomes runnable and testable.

## License

A license will be selected before the first public release.

## Python offline geometry backend (Stage 2A)

The new backend reuses C++ observations. Python implements camera geometry,
window selection, grouped robust RGB-D residuals, sparse Jacobian structure,
state validation, rejection/rollback and anchor-relative history correction.
SciPy supplies numerical least squares; Ceres/OpenCV remain third-party
solvers/front-end primitives. There is **no online writeback**, new map-based
tracker, risk-policy change, dynamic reconstruction or real-data accuracy claim.

### Build and reproduce without a real dataset

Keep builds and generated evidence outside the checkout. Existing C++ dependency
requirements still apply (Eigen 3, OpenCV 4.6, Ceres 2.2). Python requires 3.10+
with `python/requirements.txt`. Optional synthetic image integration tests also
need Python OpenCV. No script automatically installs dependencies or downloads data.

```bash
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j 4
ctest --test-dir "$BUILD" --output-on-failure

# OUT must not exist; its parent must exist. This generates mathematical data.
PYTHONPATH=python python tests/python/make_fixture.py --output "$OUT"
python python/run_geometry_replay.py \
  --snapshot "$OUT/input.snapshot" --frames "$OUT/frames.txt" \
  --ceres-executable "$BUILD/replay_local_ba" \
  --output "$ANALYSIS" --max-nfev 800

# With these executables supplied, the two integration tests run rather than skip.
PYTHONPATH=python AFS_CERES_REPLAY="$BUILD/replay_local_ba" \
  AFS_ODOMETRY="$BUILD/run_rgbd_odometry" \
  python -m unittest discover -s tests/python -v
```

Set `BUILD`, `OUT`, `ANALYSIS` explicitly to distinct absolute locations. New
entries refuse existing output locations. Exit status 0 means accepted/complete,
2 means a Python update was rejected (the output snapshot is the original), and
1 means an input/process/I/O failure. A failed Ceres process retains its log.
The original C++ entry retains its historical behavior when export is disabled.

### Export from an existing local sequence

```bash
"$BUILD/run_rgbd_odometry" "$DATASET" "$TRAJECTORY" "$MAX_FRAMES" - none "$CACHE"
python python/run_geometry_replay.py \
  --snapshot "$CACHE/$FRAME.before.snapshot" --frames "$CACHE/frames.txt" \
  --ceres-snapshot "$CACHE/$FRAME.ceres.snapshot" --output "$ANALYSIS"
```

Export mode currently requires risk disabled, no degradation, and the existing
Freiburg1 calibration. Do not use it for another camera calibration. It refuses
existing trajectory/sidecar outputs and reserves a new cache directory. The
completion marker is written only after successful output flushing; a partial
cache must not be treated as a complete sequence. Export I/O is included in the
runner's frame timer, so those timings are not the untouched system baseline.

Each BA event exports its before/after pair. `ba_events.csv` retains acceptance,
residual counts and separate before-export, Ceres-solve and after-export timing.
`frames.txt` records every frame,
validity, timestamp, the **actual tracking reference ID**, the assigned anchor ID
and raw `T_wc`. Failed frames stay in that log and are never filled by correction.
`final.snapshot` is the final window dependency snapshot, not a full map export.
The Python command replays **one independent frozen window**. It does not chain
windows, recompute associations, or represent a separate end-to-end SciPy run.

### Interchange contract and geometry

`AFS_GEOMETRY 1` is a tokenized text schema with 17-digit double precision:

- `META`: source revision (with dirty marker when configured), state version.
- `CAMERA`: fx, fy, cx, cy; pixel coordinates are u=column, v=row.
- `CONFIG`: window size, minimum observations, Ceres iteration limit, Huber delta,
  raw-depth divisor and depth residual weight.
- `KEYFRAMES`: contiguous keyframe IDs, source frame IDs, timestamps, depth image
  dimensions and row-major 4x4 `T_wc` matrices. Frame IDs and point IDs are separate.
- `POINTS`: original point IDs, first-observation host IDs, metric world XYZ.
  Only points touching the active window are exported, including singleton points.
- `OBSERVATIONS`: point/keyframe/feature IDs, the original float keypoint UV,
  and the **exact uint16 sample used by C++ BA** (0 means no depth residual).
  External observations of exported points remain available for auditing.

Stored poses are `T_wc`; optimized angle-axis plus translation describe `T_cw`.
Camera Z is metric axial depth (`raw / depth_scale`), not range. The cost is
`0.5 * sum rho(||pixel_error||²) + 0.5 * sum rho((100 * Z_error)²)` at default
weight. The 2D pixel block has one Huber loss; depth has its own scalar loss.
Python transforms complete residual blocks and uses SciPy linear loss to match
that Ceres **objective**, not its linearization or update sequence.

Parity mode uses the last five keyframes, fixes the oldest in that window,
selects points with at least two window observations, and excludes external
observations from fitting, matching existing Ceres. Depth supplies metric scale;
a fixed pose alone does not. Ceres iterations and SciPy function-evaluation
limits are different quantities and are reported separately. Reports include
input hashes, versions, solver/process timing and Ceres executable hash when run.
The dirty marker alone is not a complete source fingerprint; retain the diff
and executable with a reproduction package.

Python checks finite/valid poses and topology, connected anchor support,
non-collinear camera support, positive active depths and metric depth support.
It accepts only a converged finite solution with nonincreasing robust cost.
Input is never mutated; failure returns an independent original snapshot. An
exact numerical rank audit is used at <=200 variables; larger problems have
structural checks only, which do not certify complete numerical observability.
A wall-time check occurs between residual calls, not inside a single SciPy call.
This is not a hard process memory or deadline limiter.

Historical correction uses original frame-stream anchors, never already-BA
poses: `T_wf_corrected = T_wk_corrected * inverse(T_wk_raw) * T_wf_raw`.
Only anchors in the chosen window are corrected; keyframe source frames use their
optimized pose. Original, Python-corrected, and optional Ceres-corrected TUM files
are kept separately. No truth is read by optimization or history correction.
This does not mean ordinary frames were optimized, or that later tracking improved.

`--repair-singletons` is a separate map-consistency ablation: singleton world
coordinates follow the host change while preserving their old host-camera
coordinates. Multi-view points are never rigidly dragged by an arbitrary host.
Parity snapshots retain existing Ceres boundary behavior; neither mode claims
to enforce all outside observations or make the complete live map consistent.

### Verification evidence and limits

Development verification on 2026-10-20 used Python 3.12.3, NumPy 2.5.2,
SciPy 1.18.1, Ceres 2.2.0 and OpenCV C++ 4.6.0. C++ dependencies were extracted
into an external user-space prefix because system development packages were
absent; no system installation or data download was performed. The external
prefix required local CMake discovery adjustments; these are not project code.

- All 17 CTest checks passed (16 existing plus snapshot round-trip/native-vs-replay).
- All 16 Python checks passed with both integration executables supplied; 0 skipped.
  Without those executables, the two integration tests explicitly skip.
- An 18-frame synthetic textured RGB-D stream exercised actual C++ export.
  Trajectory and health output were byte-identical with export on/off; this is
  a correctness check on this fixture, not a claim of universally zero overhead.
- The shared mathematical fixture has 3 keyframes, 8 points, 24 observations and
  36 free variables. Both backends optimize the same observations and grouped loss.

| Shared quantized-depth fixture | Initial | SciPy | Ceres |
|---|---:|---:|---:|
| Robust objective | 328.0057053 | 0.0001459396003 | 0.0001459396071 |
| Pixel RMSE | 6.8895164 | 0.0010071453 | 0.0010069895 |
| Keyframe translation RMSE against synthetic truth (m), no alignment | 0.04122297 | 0.00007709 | 0.00007705 |

The maximum backend translation difference was approximately `1.03e-7 m`.
Final diagnostic verification measured about 30.8 ms for the Python backend including
validation/rank checking, 0.232 ms inside Ceres, and 3.45 ms for its child process
including exchange/startup. The entire Python CLI took about 0.35 s with 82,016 KiB
peak RSS. **Python was substantially slower on this tiny fixture**; scopes differ,
these are single samples, and none is a full-system performance measurement.
No real-window scalability or speed benefit has been established.

The first Python test run had two failed assertions: serialized observation
ordering differed, and a general-rotation uint16-depth fixture exceeded the
unquantized rotation tolerance. Round-trip validation now compares observation
identity/value sets; a representable-depth rotation case uses the strict
`1e-4 m / 1e-5 rad` recovery thresholds, while quantized-depth recovery is tested
separately with 0.3 mm translation tolerance. Thresholds were not silently relaxed.
Initial local C++ configuration/link attempts also failed while resolving the
external dependency prefix, before the successful build and full test run.

This stage demonstrates a working Python geometry model, matching snapshot
objectives, rollback and offline correction. Real-sequence ATE, fixed-interval
RPE, coverage, held-out geometry, full-window cost and any online tracking benefit
remain **unmeasured**. Stop here pending real snapshots; do not infer multi-scene
benefit, begin online writeback, or add risk scheduling from these results.

### Real development-sequence check: fr1_xyz (2026-10-20)

After the original machine became unavailable, the user authorized downloading
**only fr1_xyz** to continue Stage 2A on this machine. The official archive was
448,204,271 bytes; its locally computed SHA-256 is
`a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`.
This is a local provenance checksum, not a publisher-verified checksum.
The 798 RGB and 798 depth entries were all present, at 640x480 with 8-bit RGB
and 16-bit depth. The unchanged C++ association produced 792 synchronized frames.
All data, snapshots, raw logs, commands and resource records remain outside Git.

The full 792-frame C++ export had 792 valid poses, 0 lost frames, 114 keyframes,
79,790 map points and 113 accepted BA updates, reproducing the historical counts.
ATE was `0.0620809808 m` (790 truth matches, 88.3019% truth-span coverage), also
matching the historical accuracy. Risk was disabled and no degradation was used.
The independently implemented Python evaluation matches existing C++ ATE and
adjacent RPE within `1e-10`; the new fixed-interval RPE uses 1 s with 0.02 s
endpoint tolerance in both truth and estimated time spans (756 pairs).

Before replay, event selection was frozen to first/lower-middle/last by frame ID:
**10, 498, 791**. Windows, points, observations and image resolution were not reduced.
SciPy's existing `max_nfev=200` and 30 s budget were unchanged. This run explicitly
set OPENBLAS/OMP/MKL thread variables to 1; OpenCV retains its original default
and Ceres its existing one-thread setting. Earlier unset-thread or other-machine
runtime numbers are not equivalent profiles. The runner reported mean/P95 frame
cost 23.7461/71.5843 ms **including export**, and export-process peak RSS was
322,548 KiB. These do not establish an algorithmic speed improvement.
The Python parent recorded 18.91 s for the export subprocess while GNU time
recorded 21.84 s; both raw timing records are retained rather than reconciled
by selecting the more favorable clock. Per-frame costs use the C++ steady clock.

| Frozen event | Free variables | Pixel observations | Python outcome | Python backend ms | Ceres BA ms |
|---|---:|---:|---|---:|---:|
| 10 | 684 | 452 | Rejected: 200 evaluations exhausted | 755.3 | 6.92 |
| 498 | 2718 | 1803 | Rejected: 200 evaluations exhausted | 1583.5 | 35.29 |
| 791 | 3795 | 2521 | Rejected: 200 evaluations exhausted | 2206.0 | 53.82 |

All Python updates were rolled back. Each output snapshot retained its original
state; all three corrected trajectory files were byte-identical to their
Python-reserialized original trajectory. Unix timestamps are now written with
9 decimal places (pose values use 17 significant digits): the old 12-significant-
digit formatting could lose milliseconds at TUM Unix timestamps and confound
truth association. A new regression test covers this; older synthetic artifacts
were retained. Original C++ trajectory files were not rewritten.

These are failures of the **current model/solver configuration and acceptance
rule**, not proof that SciPy cannot solve these problems. Ceres accepts a usable
solution via `IsSolutionUsable()`, while Python currently requires SciPy success;
those acceptance policies are not equivalent. The unaccepted SciPy last iterate
was not exported and its final objective was not recorded, so no claim is made
about its residual improvement or distance from convergence. In these rejected
run reports, the legacy `backend_difference` field compares the Ceres output to
the **rolled-back original state**, not to an accepted Python solution.

For an independent historical-correction check, the matching Ceres snapshot was
used on only its selected window's anchored frames; the rest of the original
trajectory was retained. Each row below is a **separate single-window change**
to the whole original trajectory, not a chained run or an online feedback test.

| Trajectory | ATE m | ATE change | 1 s translation RPE m |
|---|---:|---:|---:|
| Original | 0.06208098 | reference | 0.04325817 |
| Python, any selected event, rejected/rollback | 0.06208098 | effectively unchanged | 0.04325817 |
| Ceres history correction, event 10 | 0.06225478 | +0.28% | 0.04328402 |
| Ceres history correction, event 498 | 0.06445355 | +3.82% | 0.05563475 |
| Ceres history correction, event 791 | 0.06778765 | +9.19% | 0.05399101 |

All rows retained 792 poses, 790 truth matches and the same 756 RPE pairs, with
no internal gaps over 0.1 s and no jumps under the predeclared 0.2 m/15 degree
threshold for adjacent poses <=0.1 s apart. No held-out-observation test was run.
Fitted Ceres residuals fell, but historical trajectory accuracy worsened; this
is direct evidence against claiming accuracy improvement from fitted cost alone.
Truth was read by separate evaluation processes after export and replay only.

The independent evaluator is `python/adaptivefusion/evaluation.py`; five new
checks cover SE(3) alignment without scale, metric-scale errors, fixed-time RPE,
invalid input/jumps, partial coverage and Unix timestamp precision. With both
C++ integration executables supplied, the expanded Python suite passes **21/21,
0 skipped**. No C++ algorithms changed in this real-data follow-up; the prior
17/17 CTest result remains the C++ check for the same binaries.

**Decision: stop backend expansion at this Stage 2A gate.** All three Python
windows were rejected and none of the tested Ceres history corrections improved
ATE; the middle and final cases exceed the proposed degradation allowances.
Do not increase budgets, loosen acceptance, select replacement windows, download
additional sequences, start online writeback or add risk scheduling automatically.
A separately approved next step could diagnose termination/linearization and
boundary consistency, while preserving this frozen failed comparison. No
multi-scene benefit, real-time Python backend, or online tracking gain is proven.

