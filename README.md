# AdaptiveFusion-SLAM

> Current engineering focus: **static-scene RGB-D geometry**, preserving the
> C++ frontend and adding a substantive Python offline optimization backend.
> The latest [offline backend evidence](#python-sparse-schur-backend-and-batch-evidence)
> includes controlled failures, same-problem Ceres comparisons, and a
> budget-dependent improvement on one development sequence. Online writeback,
> dynamic/4D reconstruction, and cross-scene benefit are not implemented or proven.

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

See the [Chinese resume-ready description](docs/resume_project_zh.md), the
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

## Development Roadmap

- [x] Initialize the repository and build system
- [x] Verify the minimal executable
- [x] Implement the pinhole camera model
- [x] Add TUM RGB-D dataset loading and synchronization
- [x] Implement ORB feature extraction
- [x] Implement ORB descriptor matching
- [x] Implement LK optical-flow tracking
- [x] Build metric RGB-D 3D-to-2D correspondences
- [x] Estimate camera pose using RGB-D correspondences and PnP
- [x] Add persistent frame state and trajectory accumulation
- [x] Add keyframe and map-point representations
- [x] Associate map points across adjacent keyframes
- [x] Implement local bundle adjustment
- [x] Record per-frame geometric health signals
- [x] Build the tracking-failure event dataset
- [x] Implement the first temporal risk prediction baseline
- [ ] Calibrate the temporal risk predictor on real sequences
- [x] Add risk-adaptive frontend and map-update policies
- [x] Add model persistence and causal online risk control
- [x] Add baseline/adaptive benchmarks and controlled degradation experiments
- [x] Release reproducible v0.1 results and documentation
- [ ] Calibrate and validate on multiple real-world datasets

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

## Stage 10: Cross-Keyframe Map-Point Association

The sparse map now reuses a world landmark when the latest keyframe and a new
keyframe observe the same scene point. ORB appearance proposes correspondence;
positive camera depth, a 3-pixel reprojection gate, and optional RGB-D depth
agreement verify that the match is physically plausible.

Accepted features reuse the old map-point id and append an observation.
Remaining features with valid depth create new world points. A controlled test
reobserves two points, adds one new point, and keeps the total at three rather
than creating duplicates. A descriptor-identical observation shifted by 50
pixels is rejected by the geometry gate. The full suite passes 9/9 tests.

The implemented chain now runs from synchronized RGB-D input through ORB/LK
tracking, PnP trajectory accumulation, keyframe selection, metric map creation,
and cross-keyframe landmark association. Association currently searches only
the latest keyframe; local bundle adjustment, covisibility search, landmark
culling, geometric-health history, and proactive failure prediction remain.

## Stage 11: Local Bundle Adjustment

Ceres Solver 2.2 now jointly refines the poses and shared map points in the
latest five-keyframe window. For every observation it minimizes the 2D
reprojection residual between the measured ORB keypoint and the projection of
the world point through the current camera pose. Only points with at least two
window observations participate.

The first pose in the window is fixed to preserve the world coordinate gauge.
The remaining poses use angle-axis rotation plus translation, map points use
three world coordinates, and valid RGB-D samples add depth residuals that lock
the reconstruction to metric scale. A 2-pixel Huber loss reduces outlier
influence. A usable solution is copied back transactionally. The RGB-D runner
invokes local BA after a keyframe reobserves existing landmarks.

In the deterministic test, two cameras observe six points through 12 image
measurements. Starting with perturbed pose and point estimates, true pixel RMSE
falls from `11.797` to `8.54062e-06`; the fixed anchor remains unchanged. The
complete Release test suite passes 10/10 tests.

Current BA uses visual reprojection and RGB-D depth residuals in a small latest-
keyframe window. It does not yet remove outlier observations, construct a
covisibility window, or synchronize optimized keyframe poses back
into the already-written odometry trajectory.

## Stage 12: Per-Frame Geometric Health Monitoring

Every processed frame now carries a `GeometricHealth` record. After the first
initialization frame, it records tracking success and fallback use together
with PnP inlier ratio, reprojection error, LK forward-backward error, feature
coverage over a 4-by-3 image grid, median feature parallax, and valid-depth
ratio. Counts for extracted features, candidate correspondences, and PnP
inliers preserve the raw evidence behind normalized metrics.

The odometry executable writes one row per input frame to
`<trajectory-path>.health.csv`, including lost frames. Initialization is marked
with `has_tracking_measurement=0`; later failed frames retain zero/partial
measurements and `tracking_success=0` instead of disappearing from the dataset.
This distinction is required to create future-failure labels without selection
bias.

A controlled four-correspondence test produces an inlier ratio of `0.75`, mean
forward-backward error of `0.3` pixels, spatial coverage of `0.333333`, median
parallax of `5` pixels, and valid-depth ratio of `0.5`. The continuous odometry
test verifies both strong tracked-frame health and retained lost-frame evidence.
The Release suite passes 11/11 tests.

These are raw per-frame signals. Temporal windows, conditioning indicators,
future-failure labels, normalization, prediction, and risk calibration remain
for subsequent stages.

## Stage 13: Temporal Windows and Future-Failure Labels

The dataset builder converts per-frame health records into fixed temporal
windows. With default history length `L=5` and prediction horizon `H=3`, a
sample anchored at frame `t` contains health from `[t-4, t]` and receives a
positive label when any frame in `[t+1, t+3]` is lost. Only anchors with a
complete measured and successful history are retained, ensuring that each
positive sample represents a warning made before failure.

Each time step contributes seven features: inlier ratio, reprojection error,
LK forward-backward error, spatial coverage, median parallax, valid-depth
ratio, and ORB-fallback use. `frames_until_failure` stores the warning lead
time; `H+1` denotes no failure inside the prediction horizon.

`run_rgbd_odometry` now writes `<trajectory>.failure_dataset.csv` after the
health log. A standalone `simulate_failure_dataset` executable creates a
24-frame controlled sequence with gradual degradation and failure at frame 15:

```bash
./build/simulate_failure_dataset results/my_simulation.csv
```

It produces 11 usable samples: eight negative samples and three positive
warnings anchored at frames 12, 13, and 14, with lead times of 3, 2, and 1
frames. The Release suite passes 12/12 tests.

The current dataset contains deterministic labels and unnormalized raw
features. It does not yet train a predictor, balance classes, split sequences
without leakage, or calibrate output probabilities.

## Stage 14: Interpretable Temporal Risk Baseline

The first predictor converts each five-frame health window into 21 explanatory
variables: the latest value, temporal mean, and least-squares slope of each of
the seven health signals. A class-balanced, L2-regularized logistic regression
then outputs the probability of failure within the three-frame horizon.

Feature means and standard deviations are fitted only on training sequences.
Training uses stable sigmoid evaluation and balanced class weights so the fewer
failure-warning samples are not ignored. Evaluation reports accuracy, AUROC,
AUPRC, Brier score, and mean lead time among correctly warned failures.

The held-out simulation uses separate episodes with unseen failure times and
degradation lengths:

```bash
./build/simulate_risk_prediction results/my_risk_predictions.csv
```

It trains on 202 windows and evaluates 119 held-out windows, including 15
positives. Results are `0.957983` accuracy, `0.997436` AUROC, `0.983824` AUPRC,
`0.0366757` Brier score, and 2-frame mean warning lead. All 15 positives are
detected at threshold 0.5, with five false warnings. The Release suite passes
13/13 tests.

These strong values are expected on controlled synthetic trends and are not
evidence of real-world performance. Model persistence, real-sequence fitting,
probability calibration, threshold selection, and cross-dataset validation
remain to be implemented.

## Stage 15: Risk-Adaptive Frontend and Map Protection

Predicted failure probability now maps to four operating levels with thresholds
`0.30`, `0.60`, and `0.85`. Low risk keeps efficient LK and normal mapping.
Medium risk adds ORB verification and halves the normal keyframe motion/gap
requirements. High risk forces ORB redetection, requests an early keyframe,
allows observations of trusted landmarks, but suppresses creation of new map
points. Critical risk blocks keyframes and all map writes while preserving the
last trusted tracking reference.

The odometry API executes ORB verification/redetection and trusted-reference
preservation. `KeyframePolicy` consumes the same decision for early or blocked
insertion, and `SparseMap` reports how many valid-depth candidates were
suppressed under restricted updates.

The controlled comparison presents 100 candidate landmarks per frame while
risk rises across 12 frames. An always-update baseline stores 1,200 points,
including 500 candidates from high/critical phases. The adaptive policy stores
700 points, suppresses all 500 risky candidates, issues four early-keyframe
requests, and freezes three critical frames. The Release suite passes 14/14
tests.

This is a policy-mechanism simulation: classifying every high/critical candidate
as unreliable is an experimental assumption, not a measured real-scene result.
The runner does not yet load a trained model for live online probabilities;
model persistence and end-to-end real-sequence comparison remain.

## Stage 16: Model Persistence and Causal Online Control

The temporal predictor now serializes its configuration, five-frame history
length, 21 normalization means/scales, logistic weights, and bias with a versioned
model header. Loading validates dimensions, finite values, and positive scales.
A save/load round trip reproduces probabilities within `1e-12`.

`OnlineRiskController` maintains only health records already observed. Once five
successful measurements exist, it predicts the action for the next frame. A
failed observation clears the window and raises critical risk. A 0.05 hysteresis
margin prevents risk levels from dropping immediately when probability merely
oscillates around a threshold.

The RGB-D runner optionally loads a model and records both the decision applied
to the current frame and the prediction produced for the next frame. On the tiny
fixture, frame 1 fails while using the prior low decision; that result raises
critical risk, which is first applied to frame 2. This demonstrates causal order.
The Release suite remains 14/14 passing.

The committed model is trained only on synthetic data and exists to verify the
online plumbing. It must not be treated as a calibrated real-scene model. The
next requirement is paired baseline/adaptive evaluation on natural sequences.

## Stage 17: TUM Trajectory Evaluation and First Real-Sequence Run

The evaluation module reads TUM trajectories, performs one-to-one timestamp
association, estimates a rigid SE(3) alignment without changing metric scale,
and reports translational ATE RMSE plus consecutive-pose RPE translation and
rotation RMSE. It also reports estimated-pose match ratio and ground-truth
duration coverage so a short or incomplete trajectory cannot be judged by
accuracy alone.

```bash
./build/evaluate_trajectory groundtruth.txt estimate.txt 0.02 report.csv
./build/compare_slam_trajectories \
    groundtruth.txt baseline.txt adaptive.txt comparison.csv
```

On the first 200 frames of the real TUM `rgbd_dataset_freiburg1_xyz` sequence,
the baseline produces all 200 poses; 198 associate within 0.02 seconds. Its ATE
is `0.0241339 m`, RPE translation is `0.00492338 m`, and RPE rotation is
`0.00610731 rad`. Duration coverage is `0.227318` because this run intentionally
uses only the first 200 frames.

The same frames with the synthetic Stage 16 model produce ATE `0.024611 m` and
655 fewer map points. This slight accuracy decrease is reported rather than
hidden: the model is synthetic and uncalibrated for TUM. The result verifies the
paired experiment path, not an adaptive-method improvement claim. The Release
suite passes 15/15 tests.

## Stage 18: Reproducible v0.1 Benchmark

The full 792-frame TUM Freiburg1 XYZ RGB-D sequence now runs in both baseline
and adaptive modes. Adaptive control keeps all poses, reduces keyframes from
114 to 92 and map points from 79,790 to 63,983, and increases measured
throughput from 16.65 to 20.87 FPS. Aligned ATE changes from 0.06208 m to
0.06244 m, so this experiment supports lower mapping cost rather than an
accuracy-improvement claim.

The runner can deterministically inject motion blur, low light, central
occlusion, Gaussian noise, or every-tenth-frame loss. Baseline and adaptive
200-frame results for all five conditions are committed in
`results/stage_18_degradation_comparison.csv`. The drop experiment causes 20
lost frames and 19 one-frame recoveries in both modes; adaptive mapping stores
10,710 points instead of 12,015 while preserving the same 90% tracking success
rate. See `results/stage_18_v0.1_release.md` for commands, tables, scope, and
interpretation. Release CTest passes 16/16 tests.

## Reproducibility Policy

Each runnable development stage will include:

- source code and configuration;
- a dedicated Git commit;
- updated documentation;
- experiment commands;
- quantitative results when available.

Large datasets and generated build files will not be stored in the repository. Download instructions and evaluation scripts will be provided instead.

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

## Python sparse Schur backend and batch evidence

The Stage 2A stop above is a retained historical decision. The user subsequently
explicitly authorized further autonomous diagnosis and branch publication.
All failed runs remain part of the evidence; no new scene, risk module, online
feedback path or learned model was added in this continuation.

### Implemented geometry and controls

Python now implements the additive angle-axis projection derivatives (including
SO(3)'s right Jacobian), correct grouped-Huber derivatives, and a standard robust
block Schur LM step. Each 3x3 landmark block is eliminated before solving camera
increments. Local camera systems use NumPy dense solves; larger camera systems
use SciPy sparse assembly and sparse linear solves. This is an implementation
of established BA mathematics, **not a claim of an original numerical method**.
Tests check derivatives against central differences and Schur steps against the
unreduced damped normal equations, for both dense and sparse camera systems.
The original SciPy TRF route remains available as a separate comparison.

The new C++ fields expose only termination type, iteration count and objective;
the C++ residuals, solving logic and acceptance behavior are unchanged. All three
original selected Ceres windows actually returned `NO_CONVERGENCE` after 20 steps
while still being usable. Python's old strict-success gate was therefore a
stricter policy. `--acceptance-policy usable` explicitly permits a finite,
positive-depth, cost-decreasing budget-limited result while reporting its real
termination status. It does not relabel it as converged. The default single-window
CLI retains the old strict policy. Numerical failure still rejects the entire update.

State replay reuses raw relative camera motion to initialize a new keyframe from
its corrected predecessor. Existing landmark states persist; a new landmark's
original raw-host-local XYZ is preserved when placed into the corrected host
pose. Singleton repair is controlled separately. The trajectory path normalizes
the window gauge to the raw history, interpolates pose corrections between
keyframes, and carries the last correction into later history. Failed frames
remain absent. These changes affect later **offline backend windows**, not the
original frontend's matching or PnP. Cached observation associations remain frozen.

The new batch entry unions actual cached observations by landmark/keyframe ID,
checks that duplicate records agree, and retains original landmark birth
geometry. It optimizes all 114 keyframes with the first fixed, 21,556 shared
landmarks, and 44,018 observations: 65,346 free variables. Singleton landmarks
are not part of that optimization or the exported PLY. This is an explicitly
larger **full-sequence offline resource profile**, not the five-keyframe local
backend and not an online SLAM result. No ground truth is passed to these entries.

### Same-sequence evidence and negative controls

All rows use the same development sequence and SE(3) trajectory evaluation.
Full numerical rows, cohort counts and configurations are in
[`results/python_offline_fr1_xyz.csv`](results/python_offline_fr1_xyz.csv).
The fixed-time RPE interval is 1 s with 0.02 s endpoint tolerance in both streams.

| Mode | ATE m | 1 s translation RPE m |
|---|---:|---:|
| Original C++ trajectory | 0.062081 | 0.043258 |
| Python consistent local windows, 20 iterations | 0.084692 | 0.036458 |
| Ceres consistent local windows, 20 iterations | 0.084694 | 0.036459 |
| Python local with fixed external observations | 0.083660 | 0.036541 |
| Python full batch, inherited 20-step budget | **0.048474** | **0.032214** |
| Ceres, identical batch input and 20-step budget | 0.059818 | 0.032767 |
| Python batch, extended budget up to 100 steps | 0.078999 | 0.035635 |
| Ceres batch, extended budget up to 100 steps | 0.083687 | 0.036540 |
| Python batch20, whole-landmark holdout | 0.048942 | 0.032498 |

The Python 20-step offline batch reduces ATE by **21.92%** and fixed-time
translation RPE by **25.53%** on this development sequence. All full-trajectory
rows retain 792 poses, 790 truth matches, the same 88.30% truth-span coverage,
756 RPE pairs and no thresholded jumps. With correction interpolation disabled,
the same optimized poses give ATE `0.048554 m`, showing that interpolation alone
is not responsible for the improvement. A separate 114-keyframe cohort improves
from ATE `0.057462 m` to `0.046171 m`; do not compare that cohort directly with
the 792-frame table.

The 20-step budget was inherited from the existing Ceres configuration before
these batch results. It is **not a converged-solution accuracy guarantee**:
more iterations lowered the fitting objective but substantially worsened ATE.
On the identical 20-step input, Python/Ceres objectives are about 32,027/32,042,
yet their ATE differs materially; numerical path and budget matter. The gain
cannot all be attributed to state feedback or language replacement. Local-window
state consistency alone also worsened ATE. These are development findings,
not independent-scene validation, and no configuration is selected inside the
optimizer by consulting truth.

For the holdout check, all observations of landmark IDs divisible by 10 were
excluded from optimization: 2,198 complete landmarks and 2,309 cross-camera
validation observations. Geometry is evaluated by transferring their original
host-local XYZ through the estimated relative poses, without fitting held-out
points. Depth RMSE fell from `0.027156 m` to `0.024341 m` (~10.4%), but pixel RMSE
increased from `1.44345 px` to `1.47146 px` (~1.9%); invalid predictions remained
zero. This is mixed evidence, not a claim that every geometry metric improves.
It is a holdout from optimization only: the original frontend and association
construction already saw the sequence. The holdout run is not used for automatic
stopping or parameter selection.

Three repeats from the exact same serialized input/configuration produced
byte-identical optimized snapshots. Python backend times were 2.714, 2.927 and
2.972 s, including validation; these are solver-repeat measurements, not three
independent scenes. One full-cache Python batch invocation recorded ~7.63 s
using its internal timer and ~11.07 s via GNU time, peak RSS 284,836 KiB; both
clock readings are retained. The paired Ceres solve took ~1.02 s (177,432 KiB
process peak). **Python is not faster than Ceres on this batch problem.** Neither
backend timing substitutes for frontend-plus-backend system FPS.

### Reproduce and inspect

After the original C++ export into a new CACHE (see earlier instructions):

```bash
# Explicit same-problem controls; strict defaults remain available.
python python/run_geometry_replay.py \
  --snapshot "$CACHE/$FRAME.before.snapshot" --frames "$CACHE/frames.txt" \
  --ceres-executable "$BUILD/replay_local_ba" --output "$SINGLE_WINDOW" \
  --method schur-lm --jacobian analytic \
  --termination-profile ceres-cost --acceptance-policy usable

# All original frozen local windows, with state consistency and visible failures.
python python/run_offline_backend.py --cache "$CACHE" --output "$LOCAL_RUN" \
  --backend python-schur --state-mode consistent

# Full-sequence offline batch; explicitly a larger resource profile.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python python/run_batch_reference.py --cache "$CACHE" --output "$BATCH_RUN"

# Independent optimization holdout; keep results separate.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python python/run_batch_reference.py --cache "$CACHE" --output "$HOLDOUT_RUN" \
  --holdout-modulus 10

# Ceres on the EXACT serialized batch problem (same selected points/observations).
"$BUILD/replay_local_ba" "$BATCH_RUN/input.snapshot" "$CERES_BATCH_RUN"
```

All output variables must refer to new locations, outside Git. Batch output
contains the input/output snapshots, `corrected.tum`, an ASCII `shared_map.ply`
in metres, source hashes and reports. The map is the optimized shared-landmark
subset, not a dense reconstruction; map accuracy has not been independently
measured. `evaluation.py` remains a separate truth-reading module. No optimizer
entry takes a ground-truth path.

Cumulative contribution: a reusable C++ observation interface plus Python camera
geometry, analytic derivatives, sparse robust BA, auditable state management,
offline trajectory/map output and independent evaluation. A finite-budget offline
accuracy gain is demonstrated on one development sequence, alongside local-window,
longer-optimization and held-out-pixel failures. Cross-scene stability, online
tracking benefit, real-time backend deployment and dynamic 4D remain unproven.

The final check for this continuation passes **17/17 CTest and 36/36 Python
checks, zero skips** with the two C++ integration executables supplied. The
Python suite includes an end-to-end synthetic batch CLI/PLY/trajectory test and
checks that rerunning into an existing output is refused. Both dense and sparse
Schur elimination are checked against a full normal-system solve. The main
20-step ATE result is also independently reproduced by the original C++ evaluator
within `1e-10`. Test success does not remove the sensitivity and negative results
reported above.
