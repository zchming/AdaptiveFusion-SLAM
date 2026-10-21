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
> [Native feature export](#native-orb-feature-export-and-python-matching) now passes
> interface/state checks; descriptor proposals add two cross-cut rigid pairs,
> with zero triangle-qualified bridges. Both cross-cut pairs subsequently
> [fail frozen independent evaluation](#fixed-proposal-independent-evaluation);
> neither is promoted into BA, and no new trajectory-accuracy gain is claimed.
> [Fixed-model diagnostics](#fixed-model-error-diagnostics) expose rotation and
> translation compensation near the matched support, without uniquely identifying
> the underlying measurement error.
> A frozen [3D-2D reused-scene comparison](#frozen-reused-scene-pair-model-validation)
> has local gains but fails overall acceptance through desk degradation and
> additional rollback; numerically converged wrong poses remain possible.
> A [development-only update gate](#development-only-pair-update-guard) now rejects
> some bad cached updates with exact pose rollback, but retains median error
> equal to the raw baseline. Its subsequent [frozen reused-scene validation](#frozen-reused-scene-update-guard-validation)
> fails through inaccurate accepted updates and regressions; online benefit remains
> unproven and the gate is not integrated.
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


## Frozen held-out sequence validation

**Overall gate: FAIL, with two evaluable positive results and one baseline
crash.** No algorithm code or parameters changed during this validation.
The immutable [protocol](results/frozen_cross_scene_protocol.json) froze source
commit `3bf94912ce1647ad7042df5933b01ed21395e15e` before downloading or processing
three complete test sequences: `fr1_desk`, `fr1_room`, `fr1_teddy`. `fr1_xyz`
remains development-only. Protocol SHA-256:
`108204ee8c9b82718da9167a888b9ecfd6aef20da6f7e20ed0db00bbb6216512`.

These are held-out sequences with different motions/content within Freiburg1;
physical capture environments can overlap. This is not a cross-sensor,
cross-dataset or statistically independent physical-site study. Official archive
URLs, expected lengths and binary/source hashes are in the protocol; downloaded
archive SHA-256, complete image counts, planned synchronized counts and all
outcomes are in [the numerical results](results/frozen_cross_scene_metrics.csv).
No archive, image, raw snapshot, log or model weight is tracked.

The fixed primary model is full-cache Python Schur LM, analytic derivatives,
20 iterations, at most 200 evaluations, a 180 s solver limit, Huber delta 2 px,
depth weight 100, native images, original associations and raw/birth initialization.
Risk scheduling, synthetic degradation and online writeback are disabled. All
synchronized pairs within 0.02 s are processed, without frame or resolution
reduction. The first keyframe fixes the batch gauge. The three thread variables
in the protocol are fixed to 1; existing C++ frontend defaults otherwise remain.
The same numerical configuration also runs with whole-landmark holdout modulo 10.
Ceres receives the exact primary `input.snapshot` as a separate solver control.

Truth is read only after **all** sequence inference attempts finish. Evaluation
uses SE(3) alignment without scale and 1 s RPE with 0.02 s tolerances on both
streams. Existing C++ evaluation independently reproduces ATE for all eight
available trajectory arms, maximum discrepancy below `2e-16 m`; its adjacent
RPE is a different metric from the fixed-time RPE reported here.

| Sequence | Planned RGB-D pairs | Original ATE m | Python ATE m | ATE improvement | Original → Python 1 s translation RPE m | Outcome |
|---|---:|---:|---:|---:|---:|---|
| fr1_desk | 573 | 0.083488 | 0.057982 | 30.55% | 0.053976 → 0.038772 | Per-sequence gate passes |
| fr1_room | 1352 | 0.431807 | 0.408782 | 5.33% | 0.074316 → 0.062309 | Per-sequence gate passes |
| fr1_teddy | 1401 | unavailable | unavailable | unavailable | unavailable | Baseline aborts; dependent arms not run |

Both evaluable primary runs retain all original timestamps, 573/1352 matched
poses and 524/1313 fixed-time RPE pairs. Truth-span coverage is unchanged at
84.79%/92.72%; original and corrected trajectories have zero thresholded jumps
or gaps over 0.1 s and the frontend reports zero lost frames. Rotation RPE also
improves. These facts establish offline accuracy gains on these two sequences,
not recovery from lost tracking: frontend decisions already happened.
`fr1_room` still has about **0.41 m ATE** after correction; the remaining drift
is material. Do not infer a tracking-success rate for the crashed teddy run.

The frozen gate requires all three primary runs to be accepted, at least two
ATE improvements of 5%, median improvement of 5%, and no per-sequence
ATE degradation beyond max(3%, 2 mm), translation RPE degradation beyond
max(5%, 0.5 mm), rotation RPE degradation beyond max(5%, 0.05 degrees), loss of
coverage or extra jumps. The two available cases pass their limits. **The missing
third outcome prevents acceptance; a three-scene median is not reported.**
The failed sequence is not replaced, truncated or omitted from the denominator.

### Failure and solver controls

On teddy the baseline aborts while attempting zero-based frame **1082** (last
successful frame log: 1081), GNU time exit **134 / SIGABRT**. Its final trajectory
and `complete.json` are absent; partial buffered `frames.txt` is incomplete and
is not evaluated as a shorter run. The final `1082.before.snapshot` contains
207 keyframes and 39 eligible shared landmarks. Its five-frame BA window starts
at keyframe 202, but residual observations only touch 204/205/206 (14/39/25
observations). The existing
[`SetParameterBlockConstant`](src/local_bundle_adjuster.cpp#L275) calls Ceres on
the unregistered oldest-pose block and triggers its fatal check. This statement
also exists in the pre-Python main source. The diagnosis only reads source and
cached topology; **no patched rerun or alternative sequence is included**.
The next engineering step should treat an unsupported gauge anchor as a
separate safety/rollback fix, then freeze a new version before a new validation.

| Same serialized batch problem | Python ATE m | Ceres ATE m | Python / Ceres final robust cost |
|---|---:|---:|---:|
| fr1_desk | 0.057982 | 0.057849 | 13168.47 / 13169.10 |
| fr1_room | 0.408782 | 0.381235 | 59700.64 / 59673.85 |

Python and Ceres both produce finite usable budget-limited states, **not
converged solutions**. Ceres is slightly better on desk and materially better on
room. Gains over the original output support offline geometry correction with
the existing observations; they do not demonstrate Python superiority, isolate
an online feedback effect or establish a new numerical solver. The Python
contribution is geometry modeling, analytic derivatives, robust sparse state
optimization, safe result handling, trajectory correction and evaluation;
NumPy/SciPy supply standard numerical operations and linear algebra.

Whole-landmark holdout removes all observations of the selected IDs from fitting.
Host-local XYZ transfer evaluates cross-camera geometry without truth or fitting
those landmarks. It remains an optimization holdout, not independent frontend
matching data. These secondary runs are not used to select the primary model.

| Sequence | Held-out landmarks / validation observations | Pixel RMSE before → after | Depth RMSE m before → after | Invalid before → after |
|---|---:|---:|---:|---:|
| fr1_desk | 812 / 830 | 1.71498 → 1.80622 (worse 5.32%) | 0.026512 → 0.022857 | 0 → 0 |
| fr1_room | 3005 / 3030 | 1.59757 → 1.62389 (worse 1.65%) | 0.030634 → 0.029289 | 0 → 0 |

Training objective and held-out depth improve, but held-out pixel RMSE degrades
on both sequences. Together with the earlier local-window and 100-step failures,
this rules out a blanket claim that further optimization improves all geometry.

### Costs, checks and reproduction

| Sequence | Exporting baseline mean / P95 frame ms | Python backend s | Extra Python full-process wall s | Python process peak RSS MiB |
|---|---:|---:|---:|---:|
| fr1_desk | 19.66 / 46.35 | 0.986 | 4.321 | 174.86 |
| fr1_room | 26.04 / 74.86 | 4.141 | 14.789 | 382.38 |

Baseline wall times are 11.37/35.30 s. Python's internal full-pipeline timers are
4.078/14.512 s; external GNU times above also include process startup. These are
single-run costs on this machine, not runtime variance estimates or replacements
for online system FPS. Backend-only Ceres times are 0.361/2.306 s, faster than
Python. Whole-process Python timing includes cache merge, parsing, serialization,
correction and PLY output, so it is not a pure data-exchange measurement. There
is no separately isolated exchange-only measurement. All completed processes
stay within the frozen 8 GiB address-space, 180 s solver, 900 s backend-process
and 1800 s exporter budgets; there are no OOMs or timeouts. Teddy aborts after
28.49 s, rather than exhausting resources.

Checks after freeze: **17/17 CTest, 36/36 Python, zero skips or test failures**.
Real-data status is separately **2/3 complete baseline exports, 1/3 abort**;
the third sequence's three dependent arms are not run because no complete input
exists. Test success does not override this failure. An initial orchestration
launch also failed before processing frames because an unqualified executable
name lost the virtual-environment prefix. Directly invoking the same declared
interpreter corrected the launcher; its traceback is retained. No algorithm,
configuration, selected sequence or budget changed.

Reproduction uses the existing export/batch/holdout/Ceres commands above on each
complete official dataset. Pass the full `inspect_tum_dataset` synchronized count
to `run_rgbd_odometry`; require the cache completion marker before using it. Use
fresh output paths and the protocol's thread environment, limits and exact code
revision. Run all inference attempts before evaluating against `groundtruth.txt`.
Binary hashes are frozen; their retained embedded `ad862...-dirty` revision label
comes from the development build before the source was committed as `3bf9491`.
Do not silently substitute a newly compiled executable when reproducing binary
identity, and do not interpret that old embedded label as a clean-main run.

Raw evidence is kept outside Git under the local validation root
`stage3-frozen-20261020`, including archives, extraction verification, complete and
partial caches, process commands/limits, logs, timing, source hashes, the original
failed launcher, per-arm evaluation and the failed aggregate gate. The evidence
manifest covers 1,207 files (dataset content represented by archive hashes), SHA-256
`83f3cbabf4a48305e618d7238b4d1e95d710b938f87cd8b196ec65f473657ef8`.
The frozen protocol and concise metrics remain auditable in Git; the raw evidence
requires this machine's external storage and is not included in a Git clone.

Cumulative contribution now includes a frozen held-out protocol, two additional
sequence-level offline gains, an independently identified baseline BA crash,
same-input solver controls, and explicit held-out reprojection degradation.
**No online state writeback, risk-policy experiment, new training, or dynamic/4D
module is added.** Reliable three-sequence validation remains incomplete.


## Anchor safety fix and frozen revalidation

**The C++ abort is fixed; the full Python-backend gate still fails.** Source
commit `c3876cb2efdfaaa188fad37bdecd120309ef23e4` adds one fail-closed check:
if the oldest local BA pose is absent from Ceres' registered residual blocks,
return `optimized=false` before solving or writing state. It does not replace
the anchor or change BA parameters. The only algorithm edit is the seven-line
check/comment in `src/local_bundle_adjuster.cpp`; Python source is unchanged.
Two regression cases cover initial and sliding windows, checking exact pose,
landmark and association preservation. The healthy RGB-D recovery test remains.

The actual old `1082.before.snapshot` now returns the CLI's expected exit 2
(safely skipped, **not** optimized), zero solver iterations and unchanged poses,
points, observations and metadata. The original crash evidence and failed
[first frozen protocol](results/frozen_cross_scene_protocol.json) remain intact.
The fix addresses this specific absent-anchor abort; it does not make arbitrary
disconnected Ceres problems observable or validate every failure mode.

A separate clean-commit build and [new frozen protocol](results/anchor_rollback_protocol.json)
reuse the three complete official archives after verifying their hashes. All
numerical settings, solver/resource budgets, arms and acceptance criteria exactly
match the first protocol. New protocol SHA-256:
`2cd9b3b653e8b9f53f6f935f3e57fd5ba0b4c46a0aa8f3076d9432fdff3488a1`.
All inference attempts finish before truth evaluation. This is **maintenance
revalidation on the same sequences after inspecting the failure**, not three
new unseen scenes or another independent dataset study. No online writeback,
new matching, extra anchor, relaxed preflight or test-time tuning is introduced.

| Sequence | Full synchronized frames | Baseline ATE m | Python output ATE m | Primary result |
|---|---:|---:|---:|---|
| fr1_desk | 573 | 0.083488 | 0.057982 | Accepted, 30.55% improvement |
| fr1_room | 1352 | 0.431807 | 0.408782 | Accepted, 5.33% improvement |
| fr1_teddy | 1401 | 0.427380 | 0.427380 | Rejected: disconnected graph; original-state rollback |

The desk and room baseline, Python, Ceres and holdout trajectory files are each
**byte-identical to their previous counterparts**. All their earlier accuracy,
coverage and held-out-pixel degradation findings therefore remain unchanged.
Teddy completes all 1401 frames: 266 BA attempts include two safe missing-anchor
skips (frames 1082 and 1087), with 264 optimized windows. Those skips preserve
state and allow completion; this is a process-reliability improvement, not proof
of better localization. Its baseline still has four thresholded jumps and ATE
about 0.43 m despite reporting every pose valid and zero lost frames.

### Disconnected geometry and the negative solver control

Read-only co-observation analysis of teddy's shared batch input finds **three
components**: keyframes 0–202 (203 cameras), isolated keyframe 203, and keyframes
204–268 (65 cameras). Only the first component contains fixed keyframe 0. The
other components have no shared-landmark link to this anchor; a single fixed
pose cannot establish the missing inter-component geometry. No truth or
invented association is used to bridge them.

Python rejects both the primary and modulo-10 holdout inputs with
`disconnected window`, exit 2, **zero objective evaluations**. The saved rollback
snapshot exactly preserves input poses, landmarks and observations. The tiny
(~`1.5e-11 m`) difference between baseline and fallback trajectory ATE is output
precision/round-trip arithmetic, not an optimization gain. The holdout geometry
is unchanged after rejection; that is not a successful validation result.

Ceres receives the exact same serialized batch input. It reports a usable,
non-converged 20-step state and lowers robust cost from 210289.20 to 149554.54,
but the disconnected component gauges remain unsupported. Its diagnostic output
has ATE **0.468401 m** (9.60% worse), 1 s translation RPE **0.249121 m** versus
baseline **0.179559 m** (38.74% worse), and **14** thresholded jumps versus **4**
(ten extra). The raw output is retained, marked `diagnostic_unanchored_ceres` in
[all numerical results](results/anchor_rollback_revalidation.csv). It is not
reclassified as a valid geometric model because `IsSolutionUsable()` is true.

The primary fallback retains 1401 poses, 1400 truth matches, 1353 fixed-time RPE
pairs, 92.99% truth-span coverage and the original four jumps. The frozen
aggregate still **fails** because only two of three primary optimizations are
accepted. The numerical median improvement is 5.33% including the zero-gain
fallback; this does not override the all-three-accepted requirement. ATE and
coverage from a rollback must not be counted as successful optimization.

### Checks, costs and next boundary

The clean frozen build passes **17/17 CTest and 36/36 Python checks, zero skips**.
Both missing-anchor regression cases run inside the existing C++ BA test.
The real-data outcome is **3/3 complete baseline exports, 2/3 accepted Python
primary runs, 1/3 safe rejection**; holdout has the same accepted/rejected counts.
All 12 output trajectories receive independent Python/C++ ATE checks within
`1e-10 m`. The original exporting binaries and first protocol hashes are intact.

| Sequence | Baseline process wall s | Additional Python process wall s | Python peak RSS MiB | Primary status |
|---|---:|---:|---:|---|
| fr1_desk | 11.23 | 4.32 | 173.80 | Accepted |
| fr1_room | 36.70 | 15.09 | 382.21 | Accepted |
| fr1_teddy | 37.95 | 11.73 | 295.56 | Rejected; includes merge/check/rollback output |

These are single-run offline process costs, not system FPS or isolated exchange
latency. No OOM or timeout occurs. Initial isolated-build configuration/link
attempts missed existing BLAS/LAPACK search paths; the corrected dependency
paths/environment reuse the same installed libraries without downloads or code
workarounds. Those failed attempts are retained alongside the successful build.

Raw evidence lives outside Git in the local `anchor-rollback-v2-20261021` root:
new snapshots, reports, all process logs, regression/rollback checks, read-only
graph analysis, timing, the failed aggregate gate and provenance. The manifest
covers 1,383 evidence files plus frozen binary/archive hashes, SHA-256
`8f2e8ae351adeac3d650c7a6fc36fd63313ff5faa0a7343c890c4b3a88af1c7b`.
The original `stage3-frozen-20261020` evidence is preserved separately. Raw
artifacts require local storage; only concise protocols and numerical rows are
published in Git.

Cumulative contribution: a minimal C++ crash guard with exact rollback tests,
unchanged substantive Python geometry modeling/optimization and independent
validation, plus evidence that graph preflight prevents a misleading solver
success from becoming an accepted correction. NumPy/SciPy and Ceres remain
third-party numerical tools; no original numerical solver is claimed. Two
sequence-level offline gains remain supported, while teddy has **no new accuracy
gain** and reliable three-sequence operation is not established.

Further work should evaluate conservative Python optimization of the anchored
connected subgraph while explicitly retaining unconstrained parts, under a
separate protocol and controls. This cannot recover missing cross-component
geometry by itself. No such model change, online feedback, risk scheduling or
new module is included in this frozen revalidation.


### Experimental anchor-component subset

**Result: usefulness gate FAIL; keep the strict default.** Python source commit
`1ff5361e1e8e218e976d060719939d743feb9dee` adds explicit
`--component-policy anchor` to the existing batch entry. Default `all` continues
to reject disconnected input. The optional model retains original IDs, selects
the shared-landmark component containing fixed keyframe 0, and optimizes only its
poses/points with the existing solver. Excluded geometry stays exactly unchanged.
Existing SO(3)/translation interpolation uses raw excluded keyframes as identity
knots; frames owned by excluded anchors are copied exactly. Output is labeled
`accepted_partial` when only a subset passes the existing numerical checks.
This label is **not an accuracy or full-observability certificate**. Output PLY
contains both optimized and unchanged shared points, not a wholly refined map.

The [frozen protocol](results/component_subset_protocol.json), SHA-256
`05e4b0287c952550bd165437b166bddebab96d131deacbd80b986b6cc082609a`, declares
strict/all, anchor, fixed modulo-10 holdout, and a no-interpolation output from
the same optimized poses. Schur LM, 20 iterations, 200 evaluations, 180 s solver
budget, Huber/depth settings and CPU limits are unchanged. All arms reuse exact
v2 caches/input snapshots; no C++ code, frontend, data download or online feedback
is added. All inference finishes before truth evaluation. These are the same
previously seen sequences, not additional independent test scenes. Numerical
rows and selected-cohort ATE are in [the results](results/component_subset_results.csv).

| teddy output | Full ATE m | 1 s translation RPE m | Thresholded jumps |
|---|---:|---:|---:|
| Original baseline | 0.427380 | 0.179559 | 4 |
| Default all: rejected, original-state rollback | 0.427380 | 0.179559 | 4 |
| Anchor subset, existing interpolation | 0.461562 | 0.254718 | 13 |
| Same optimized poses, no interpolation | 0.467645 | 0.277262 | 6 |
| Anchor subset with fixed holdout | 0.471314 | 0.225691 | 13 |

The primary subset covers 203/269 keyframes, 21,455 shared points, 43,035
observations and 1,067/1,401 valid frame records. The other 66 keyframes, 5,214
shared points and 334 frame records (including excluded keyframe instants)
retain their input state; serialized frame poses agree within `1e-12`. Coverage,
1,400 truth matches and 1,353 fixed-time RPE pairs are unchanged. Yet full ATE
worsens **8.00%**, translation RPE **41.86%**, and nine new jumps appear. Active
keyframe ATE worsens from `0.179472 m` to `0.245295 m` (**36.68%**) with either
history rule. Thus interpolation alone cannot explain the failure. New jumps
occur at frames 1018–1027, before the component-ownership boundary at 1067;
this is not solely a boundary/tail deformation problem.

The active training objective falls from 168381.20 to 120054.67, but all-held-out
pixel RMSE rises from `1.65149 px` to `2.58755 px`; depth RMSE falls from
`0.071168 m` to `0.068104 m`, invalid predictions remain zero. This pool includes
unchanged components. Lower fitting loss and depth error do not establish better
poses. Desk/room primary outputs stay byte-identical to both the strict control
and the previous version: their 30.55%/5.33% gains are retained, not new gains.
The predeclared partial-usefulness criteria (at least 5% improvement in full and
selected-keyframe ATE, bounded RPE, no extra jumps, preserved excluded state and
resource limits) fail. The earlier full connected-model gate also remains false.

A post-freeze **read-only observability audit**, without fitting or truth, finds
a concrete missing constraint inside the selected component. The temporal cut
before keyframe 159 (frame 847) is connected by only one shared landmark, ID
117225. Rotate all right-side cameras and right-only points about that landmark:
for center `X`, let `G = [R, X - R X]`, then
`T_wc' = G T_wc`, `P' = G P` and `T_cw' P' = T_cw P`. The shared point is fixed
because `G X = X`. Consequently pixel and depth residuals cannot constrain this
relative rotation, while keyframe 0 remains fixed. A 0.1 rad test rotation moves
camera positions by up to **0.21687 m** with maximum camera-coordinate difference
`1.27e-14 m` and robust-residual difference `2.28e-12`. This is an invariance
calculation on cloned state, not another performance arm or a parameter search.
It proves an unresolved rotational gauge exists; it does not identify every
cause of the observed accuracy loss. Topological connectivity and per-camera
spatial support are insufficient here. The existing large-problem preflight
only checks structure, unlike the exact numerical rank audit for small examples.

Checks pass **44/44 Python and 17/17 CTest, zero skips**. Eight new Python checks
cover subset geometry recovery, noncontiguous IDs, fixed/isolated anchors,
rollback, excluded ordinary frames, connected-policy equivalence, CLI labeling
and output preservation. They do not certify observability of this larger graph.
Teddy's optional run costs **14.14 s** wall time (backend 3.09 s), peak RSS
347.89 MiB; desk/room take 4.42/15.39 s. No OOM, timeout or configuration change
occurs. Timings are offline, single-run, and include parsing/merge/output costs.
Six independent C++ ATE checks reproduce baseline/candidate Python evaluation
within `1e-10 m`. Raw evidence remains outside Git in
`anchor-component-v3-20261021`; 141 evidence files are covered by manifest SHA-256
`d9783d916e8a9adb03c2ab63d8ad70fcd4524e996a6b2729d36b48d46d3e8b55`.
All previous failures, caches and public branches remain available.

Cumulative contribution: Python subgraph/state management and eight regression
checks, plus an auditable negative result and a geometric gauge-invariance
counterexample. NumPy/SciPy provide numerical primitives; no new solver is
claimed. The opt-in model stays experimental and is **not promoted to the
default**. Further work should first diagnose geometric observability and
single-/few-landmark bridges, rather than add optimization layers or relax
checks to force a positive result. No such additional model/gate is implemented
in this frozen study, and no online tracking benefit is established.

### Python geometric bridge guard

**Frozen safety gate PASS; full accuracy gate still NOT PASSED.** Python source
`972a01cc962d06c92e2b7c77297a45b29b79a9f8` adds an explicit batch-only
`--observability-check bridge-audit`, intended for the experimental anchor subset.
Default `structural` and default `--component-policy all` remain unchanged.
No C++ implementation, frontend observations, solver settings or online state
feedback changes in this step.

The guard checks the active camera/landmark graph with iterative Tarjan
single-landmark separator detection, then examines shared landmarks across each
ordered temporal cut. One/two-point support provides a rotational gauge witness;
three or more estimated points are checked for numerical collinearity using
centered-coordinate SVD (`sigma2 <= 1e-8 * sigma1`, absolute spread `1e-10 m`).
A noncollinear planar triangle is allowed. Detected problems return before
optimization (`nfev=0`) with complete state/trajectory rollback. These are
**sufficient rejection witnesses, not a full observability certificate**:
arbitrary non-temporal two-point/collinear separators can be missed, and
estimated-coordinate collinearity depends on the input state. A mathematical
counterexample for the non-temporal two-point blind spot is retained as a test.

The [protocol](results/observability_guard_protocol.json), SHA-256
`e18b0d98cb69673fbea0346ade683d4c5f471bbef1a05c775d97b5cf937911b2`, was frozen
before real-cache evaluation. Nine fresh arms reuse the exact v2 caches: unguarded
anchor control, guarded anchor, and guarded fixed modulo-10 holdout for each
sequence. All inference completes before separate truth evaluation. These are
three previously seen Freiburg1 sequences, not new independent test scenes.
The same 20-iteration/200-evaluation, 180 s backend, 900 s process and 8 GiB
address-space budgets apply. Thresholds, graph selection and defaults are not
adjusted after results. [All numerical rows](results/observability_guard_results.csv)
include rejected outputs and holdout regressions.

| Sequence | Baseline ATE m | Unguarded anchor ATE m | Guarded ATE m | Guard outcome | Jumps baseline/control/guard |
|---|---:|---:|---:|---|---:|
| desk | 0.083488 | 0.057982 | 0.057982 | accepted, not certified | 0 / 0 / 0 |
| room | 0.431807 | 0.408782 | 0.408782 | accepted, not certified | 0 / 0 / 0 |
| teddy | 0.427380 | 0.461562 | 0.427380 | rejected, full rollback | 4 / 13 / 4 |

Fresh controls match v3 outputs byte-for-byte on all three sequences. Accepted
primary and holdout desk/room outputs also match their corresponding unguarded
outputs byte-for-byte; no previous ATE benefit is forfeited on these caches.
The 30.55%/5.33% primary ATE improvements are retained earlier gains, not extra
gains produced by the guard. Primary 1 s translation RPE is respectively
`0.038772 / 0.062309 / 0.179559 m`, versus baseline
`0.053976 / 0.074316 / 0.179559 m`.

Teddy has a single-landmark separator (point 117225) isolating 44 cameras, plus
temporal cuts before keyframes 112 and 159 with two/one crossing landmarks.
The holdout arm has four failing temporal cuts. Both guarded teddy arms exit 2,
perform zero optimizer evaluations, and preserve every input pose/point and
all 1,401 timestamps/valid poses. Baseline ATE/RPE agree within `1e-8 m` after
serialization; the original four jumps remain. This avoids the failed control's
8.00% ATE degradation and nine extra jumps, but does not improve teddy or repair
its missing geometric constraints. A rejected solve is still a rejected solve.

Held-out pixel RMSE still worsens on desk (`1.71498 -> 1.80622 px`) and room
(`1.59757 -> 1.62389 px`), despite depth RMSE improving
(`0.026512 -> 0.022857 m`, `0.030634 -> 0.029289 m`). Teddy holdout metrics stay
exactly unchanged through rollback. No invalid holdout predictions occur.
This safety check does not establish generalization of the residual model.

Primary audit times are `22.62 / 88.15 / 73.40 ms`, below the frozen 5 s audit
budget. Guarded complete offline process times are `4.42 / 15.59 / 11.18 s`,
with peak RSS `174.08 / 383.86 / 295.32 MiB`. The teddy rejection still incurs
parsing/export/state costs; audit speed is not whole-system throughput.
Single-run timings do not establish P95 or a statistically reliable speedup.
No timeout/OOM occurs. Six independent C++ ATE evaluations agree within
`1.95e-16 m` with Python evaluation.

Checks: **58/58 Python and 17/17 CTest pass, zero skips**. Fourteen new tests
cover gauge invariance, planar support, non-temporal limitations, brute-force
separator crosschecks, a 1,500-camera chain, coordinate invariance, solver bypass,
complete rollback and CLI coverage. Raw runs, failed exits, state audits and test
logs remain outside Git in `observability-guard-v4-20261021`; 116 evidence files
have manifest SHA-256
`75e530fdb80215df610e04667795c35810b7a73a2e7f3d84a1755773e9b5103f`.
All previous negative results and reproduction paths remain intact.

Cumulative contribution: substantive Python residual/state/trajectory handling,
subgraph selection, geometric gauge analysis and an opt-in rejection guard with
regression evidence. Tarjan/SVD/Schur LM are established algorithms; NumPy/SciPy
provide numerical primitives, not an original solver claim. Further geometric
improvement requires stronger valid observations or a separately justified model;
passing this guard does not authorize online feedback or risk scheduling.

### Cached-feature bridge candidates

**Diagnostic outcome: 0/12 directed pairs accepted; no geometric improvement
established.** A Python-only read-only entry reuses existing cached feature
coordinates and the original RGB-D images. It proposes sparse cross-keyframe
tracks using OpenCV forward/backward LK (21 px window, 3 pyramid levels, 30
iterations, 0.01 termination), then checks native depth and metric rigid geometry.
It neither detects a new feature frontend nor changes existing observations,
map ownership, BA, trajectory output or online state. No C++ change or download
is required. OpenCV supplies LK; Python implements selection, provenance,
ownership/depth filtering, rigid model verification and auditable output.
Kabsch/RANSAC are standard methods, not an original solver claim.

The [frozen protocol](results/bridge_candidate_protocol.json), SHA-256
`0a114b5ad484b44778d698faa4c6346706867c6770cfc46ae8bc884fbbd8ddc7`, selects every
adjacent temporal cut with fewer than three crossing landmarks, or the first
minimum-support cut if there are none, in both directions. The same three
previously seen snapshots/images are used; this is not fresh independent scene
validation. Matching uses no cached pose or truth. Fixed gates: forward/backward
error <=1 px; exclude targets within 2 px of cached observations or earlier
proposals; reject missing/border depth or >0.05 m local 3x3 depth spread.
Kabsch/RANSAC uses metric 0.03 m inliers, 256 trials and seed 17, fitting only
even point IDs and validating odd IDs without refitting. At least 12 fit and
6 validation inliers, each >=50% of its split, are required. A split is only
within-pair consistency, not independent accuracy or verified landmark identity.
Target-ownership exclusion deliberately does not handle duplicate-point merging.

| Sequence / selected cut | Directed pairs | Candidates after image/depth gates | Accepted pairs |
|---|---:|---|---:|
| desk 22/23 | 2 | 0, 0 | 0 |
| room 142/143 | 2 | 7, 2 | 0 |
| teddy 111/112 | 2 | 0, 0 | 0 |
| teddy 158/159 | 2 | 10, 2 | 0 |
| teddy 202/203 | 2 | 7, 0 | 0 |
| teddy 203/204 | 2 | 0, 0 | 0 |

[All directed results](results/bridge_candidate_results.csv) retain counts,
split sizes, rejection reasons and resource costs. Every real pair fails support
before rigid fitting; no real-data rigid transform is accepted. Keyframe 203 has
no source observations in this merged snapshot. The original snapshot hashes and
all sampled cached raw-depth values match their native images. These facts do
not establish a lack of image overlap: only cached observations are considered,
and valid candidates near another cached point could require a separate,
carefully verified point-merge model. No threshold was relaxed after outcomes.

The [first frozen attempt](results/bridge_candidate_initial_failure.json) failed
all three sequences before matching: the importer over-rejected disjoint equal
RGB/depth association costs. A narrow reader repair allows disjoint edges whose
processing order commutes, while still rejecting conflicting eligible ties.
Candidate settings/selection stayed fixed; the corrected importer was re-frozen
before matching. Both failed and repaired sources/logs are retained. An initial
synthetic outlier-recovery expectation also failed because the fit split had two
competing coherent transforms; its safe rejection is now an explicit regression,
alongside a separate 20% outlier recovery case. No algorithm gate was loosened.

**70/70 Python tests pass, zero skips**; the unchanged C++ build passes **17/17
CTest**. Twelve new Python checks include transform direction, metric scale,
noise/outliers, independent split rejection, planar/collinear support, depth
provenance, real synthetic-image LK, association ties and overwrite refusal.
Full audit process times are `0.42 / 0.87 / 0.92 s` for desk/room/teddy, within
300 s and 8 GiB limits; no timeout/OOM occurs. These are single-run candidate-audit
costs, not whole-system frame timings. The previous ATE results are unchanged
and were not re-evaluated in this diagnostic.

Run with explicit locations (an existing output directory is refused):

```bash
PYTHONPATH=python python python/audit_bridge_candidates.py \
  --snapshot /path/to/input.snapshot --dataset /path/to/tum_sequence \
  --output /path/to/new_audit
```

Raw evidence remains outside Git in `bridge-candidates-v5-20261021` (19 files,
manifest `a1cbf01ff1dff802e3a51a7a51b8ed67c37fdbe7480907e7147163923203a49c`) and
`bridge-candidates-v5b-20261021` (32 files, manifest
`186c86223bb444ed27fc9031007e1f98ea20715d2d31afadb48f7f9692f01925`).
Cumulative contribution adds a reproducible Python observation-candidate audit
and preserves an unsuccessful attempt to supply stronger geometry. This route
is not connected to BA or promoted. A separate next step could inspect short
multi-frame tracks or verified duplicate-landmark identity; it must establish
valid constraints before any new optimization or online claim.

### Short-window landmark identity audit

**No supported identity candidates; no new geometric or accuracy benefit.**
The previous audit excluded targets near cached features. This Python-only
follow-up explicitly examines existing feature identities in a four-keyframe
window (two each side of the same weak/minimum-support cuts). It evaluates every
unordered pair, including within-side controls, deduplicating pairs shared by
windows. No feature detector, C++ interface, point/observation rewrite, BA,
trajectory output, online feedback or data download is added.

The [frozen protocol](results/landmark_identity_protocol.json), SHA-256
`a4fadbe238bd3b71c42cb77b3c7dd598c9142a3486f8fa4accfa81a51f3684d9`, retains the
prior LK (21 px, 3 levels, 30 iterations, 0.01 termination), forward/backward
<=1 px, native depth, 0.05 m local depth-spread and 0.03 m rigid inlier gates.
New identity matching requires a unique reciprocal cached feature within 2 px,
using both image directions. Even source point IDs fit Kabsch/RANSAC (256 trials,
seed 17); odd IDs validate without refitting. Minimum support stays 12 fit and
6 validation inliers, each >=50% of its split. Existing same-ID matches are
controls, not new identities. No cached pose, map coordinate or truth guides
matching or geometric verification.

Three accepted camera transforms must compose as `T_ca = T_cb T_ba`, within
0.03 m translation and 0.02 rad rotation. A pair needs at least one accepted
triangle and no failing accepted triangle. An identity then needs geometric
inliers from two distinct cross-cut pairs and at least three keyframes.
Original co-observation conflicts reject pairwise identities and entire
transitive identity clusters. Repeated pair records do not increase evidence.
These are necessary consistency checks, not independent identity truth:
views and ID-parity splits remain correlated. The tool never merges points.

| Sequence | Unique pairs | Depth-valid different-ID matches | Accepted rigid pairs | Identity candidates |
|---|---:|---:|---:|---:|
| desk | 6 | 2 | 0 | 0 |
| room | 6 | 2 | 0 | 0 |
| teddy | 21 | 5 | 0 | 0 |

All **33 unique pairs fail split support before rigid fitting**. Therefore no
real-data transform, triangle or identity is validated; only synthetic tests
exercise those later gates. The largest depth-valid pool is 18 matches in teddy
201/202, split 9/9, below the frozen 12-fit requirement. Nine different-ID matches
are raw proposals, not nine verified duplicates. Six windows contain 36 pair
references but only 33 unique pairs; neither repeated views nor windows are
counted as independent scenes. These are the same previously seen sequences.
[Every pair](results/landmark_identity_pairs.csv) and
[every window](results/landmark_identity_windows.csv) retains its rejection,
counts and resource costs. No post-result tuning or replacement window occurs.
This does not prove there are no duplicates or image overlap; it establishes
that this specific cached-feature/keyframe-LK route supplied no usable evidence.

**82/82 Python and 17/17 CTest pass, zero skips.** Twelve new tests cover
synthetic duplicate recovery, metric direction, independence from cached poses,
immutable state, same-ID controls, native depth provenance, reciprocal ambiguity,
repeated evidence, triangle order, and pairwise/transitive ownership conflicts.
All three processes exit 0, while every geometric pair is explicitly rejected.
Wall times are `0.42 / 0.92 / 1.02 s`, within 300 s/8 GiB budgets, with no timeout
or OOM; these are candidate-audit costs, not SLAM performance. Input snapshot
hashes and native cached-depth provenance pass. No truth or accuracy evaluator
is read, and previous ATE results are unchanged.

```bash
PYTHONPATH=python python python/audit_landmark_identity.py \
  --snapshot /path/to/input.snapshot --dataset /path/to/tum_sequence \
  --output /path/to/new_identity_audit
```

Raw pair records, failed gates, image hashes, frozen source and test logs remain
outside Git in `landmark-identity-v6-20261021`: 59 files, manifest SHA-256
`0a562e5a657d78f57eb46a4fdea0249d24c68bd69a80238307576768c5b7c78b`.
Cumulative contribution adds Python reciprocal identity evidence, transform-cycle
checks and conflict-safe candidate grouping with auditable negative validation.
OpenCV provides LK, NumPy/SciPy provide established numerical operations; no
original solver or reliable merge claim is made. This frozen route stops without
BA integration. A separately scoped ordinary-frame tracking audit could test
whether shorter image intervals recover support; no such benefit is established
here, and no threshold is relaxed to force acceptance.

### Ordinary-frame track comparison

**Usefulness gate FAIL; no BA integration or accuracy gain.** Python now supports
transporting the same cached feature seeds through every actual processed image
between the selected keyframes, in both directions. Lost tracks never reactivate;
there is no new detector, pose initialization, skipped processed frame or depth
screening at intermediate frames. Each adjacent LK step keeps the original
1 px forward/backward check. Endpoint native-depth, reciprocal identity, 12-fit /
6-validation rigid, triangle and ownership gates remain unchanged. Per-step
checks do not guarantee a global drift bound. This is offline correspondence
analysis, not feedback to the original tracker.

The [frozen protocol](results/ordinary_track_protocol.json), SHA-256
`aaa934353cdbd3c58f387ffeba3649ab58139fe9b1cc778ff661787d9590631b`, declares
paired direct/chain arms for the same 33 unique pairs and six windows. All 33
fresh direct pair reports reproduce v6 byte-for-byte. The only model change is
feature transport; both arms share endpoint/depth/geometry/identity verification.
Snapshots and complete processed-frame streams retain exact hashes. The same
previously seen three scenes are reused, without truth or new downloads.

| Count over the same 33 pairs | Direct keyframe LK | Ordinary-frame chain |
|---|---:|---:|
| Reciprocal endpoint matches | 68 | 90 |
| Native-depth-valid matches | 48 | 61 |
| Different-ID matches | 9 | 7 |
| Accepted rigid pairs | 0 | 0 |
| Supported identity candidates | 0 | 0 |

All real pairs still fail split support before fitting. More matches do not
establish accurate geometry. Some teddy pairs regress: depth-valid matches drop
14 -> 10 at 157/158 and 18 -> 16 at 201/202. Every rejection and transport loss
is retained in [paired results](results/ordinary_track_pairs.csv) and
[window results](results/ordinary_track_windows.csv); no threshold/window changes
follow results. No map, BA, trajectory or online state is changed, and ATE is
not re-evaluated. This transport route supplies no supported bridge constraints.

**92/92 Python and 17/17 CTest pass, zero skips.** Ten new checks cover exact
one-step equivalence, multistep direction, immutable seeds, irreversible loss,
missing frames, unchanged geometry through the endpoint interface, and complete
CLI execution on a synthetic RGB-D dataset. The initial 8/9 synthetic run failed
an incorrect assertion that arbitrary seeds must all survive LK; the corrected
test permits loss while checking surviving displacement and monotone survival.
Its initial log is preserved, with no algorithm/gate change.

Joint process times are `0.62 / 1.07 / 1.47 s` for desk/room/teddy, peak RSS
`112.64 / 148.87 / 158.03 MiB`, within 300 s/8 GiB limits, no OOM/timeout.
Across pairs, direct tracking plus endpoint checks take about 0.097 s versus
0.340 s for chain transport plus the same endpoint checks, excluding shared
loading/export. These single-run audit costs do not establish SLAM throughput or
P95 performance; the extra work did not yield validated geometry.

```bash
PYTHONPATH=python python python/audit_ordinary_tracks.py \
  --snapshot /path/to/input.snapshot --frames /path/to/cache/frames.txt \
  --dataset /path/to/tum_sequence --output /path/to/new_track_audit
```

Raw evidence remains outside Git in `ordinary-tracks-v7-20261021`: 95 files,
manifest SHA-256 `4a598270982b9329a2350fecfaab7fcb0d239a2c6a8f7484c9593d3dd5272067`.
Cumulative contribution adds Python loss-state management and a controlled
transport ablation; OpenCV provides LK and the existing numerical backend is
unchanged. This family of cached-seed LK bridge attempts has not supplied useful
geometry. Further work should first inspect what original frontend features and
matches the cache omits, then justify any minimal export interface, rather than
add more variants or relax validation to force acceptance.

### Candidate observation pool and minimal feature-export proposal

**Candidate data recovery succeeds; geometric usefulness still FAILS.** The prior
matching audits reused the BA input produced by `merge_cached_problem`, which
intentionally keeps only points with >=2 observations. That BA filter is valid
for its existing model, but using it as the only matching candidate pool omits
already cached single-view features. For example, teddy keyframe 203 has zero
observations in the shared BA input but 183 in the original before-snapshot
union. This corrects the scope of the earlier negative matching studies; those
results describe their restricted candidate pools, not the complete frontend.

A Python-only `--observation-cache` option now supplies a separate read-only
candidate view. It unions immutable original before-snapshot observations,
checks source/camera/depth/frame metadata and exact shared-subset consistency,
and preserves baseline poses/points/observations. Extra birth coordinates are
provenance only, unused by matching. The BA input builder and optimizer are
unchanged; recovered points are never inserted into BA or the live map.

| Sequence | Shared observations | Recovered candidate observations | Single-view points recovered |
|---|---:|---:|---:|
| desk | 15,844 | 71,230 | 55,386 |
| room | 61,274 | 212,829 | 151,555 |
| teddy | 53,699 | 219,506 | 165,807 |

The [frozen protocol](results/observation_pool_protocol.json), SHA-256
`5bfa58657c1a6b9c46b1c2a9770118169e4906ae1e2006f1b9f54dae43c03c1a`, changes
only the candidate pool. Both pools run direct and ordinary-frame transport with
identical existing gates on the same 33 baseline-selected pairs. All 66 shared
pair reports reproduce v7 byte-for-byte. All 132 pair/arm outcomes still fail
support before rigid fitting; neither pool produces an accepted identity.
For the chain arm, depth-valid totals change `61 -> 72` and different-ID totals
`7 -> 67`, but some scene/pair counts fall. More dense candidates can change
reciprocal-uniqueness outcomes; this is not an accuracy or universal matching gain.
No threshold, window, repeated-ID rule or BA model is changed after results.
Single-view point pairs can also lack repeated-ID support even if a local match
is valid; that limitation is retained rather than loosened. The same previously
seen scenes are reused. [All aggregate outcomes](results/observation_pool_results.csv)
and raw rejected records remain available.

**99/99 Python and 17/17 CTest pass, zero skips.** Seven new tests cover recovery,
immutable baseline geometry, deduplication, conflicting cached observations,
wrong shared subsets, metadata and incomplete/risk-enabled inputs. All six CLI
runs finish without timeout/OOM. Recovered-pool paired process times are
`3.97 / 10.43 / 13.03 s`, peak RSS `234.02 / 467.46 / 475.66 MiB`; these include
cache recovery and both tracking arms, not SLAM throughput. Every original cache
manifest, input snapshot and processed-frame stream remains unchanged. No truth,
map merge, BA, online feedback, C++ change or download occurs.

```bash
PYTHONPATH=python python python/audit_ordinary_tracks.py \
  --snapshot /path/to/shared/input.snapshot --frames /path/to/cache/frames.txt \
  --dataset /path/to/tum_sequence --observation-cache /path/to/original/cache \
  --output /path/to/new_pool_audit
```

Raw evidence remains outside Git in `observation-pool-v8-20261021`: 175 files,
manifest SHA-256 `15bf243791ce4471aa1ae3e5a2d797a2d18c4816661dc3966cd80bd7c26ee25a`.
The contribution is a corrected Python candidate-data boundary and an auditable
negative result, not improved localization or a new solver. OpenCV/NumPy/SciPy
continue to supply their existing numerical operations.

**Proposed next interface, NOT IMPLEMENTED; requires approval for C++ edits.**
`Keyframe` already exposes complete keypoints, ORB descriptors and map-point IDs.
Geometry snapshots export only map observations and omit descriptors; their C++
reader creates zero descriptor placeholders. Those placeholders must never be
used as recovered ORB evidence. Frontend chosen LK/ORB correspondence vectors
are local temporaries; health CSV preserves counts, not their identities.

The minimal proposal exports each existing keyframe once, after map insertion
and before BA, through an opt-in feature-export flag on the current geometry
export entry. No frontend, matcher, map or optimizer algorithm changes. An
`AFS_FEATURES 1` sidecar records source revision, capture phase, keyframe/source
frame IDs, actual RGB/depth timestamps, camera/depth/ORB settings, original feature
order, full keypoint fields, native uint16 depth, optional map-point ID and all
32 ORB descriptor bytes per feature (lossless hex). Relative image references and
separate byte/time accounting accompany the sidecars. No image duplication,
binding framework, online writeback or new dependency is needed. New output
files refuse overwrite; incomplete exports retain evidence without a completion
marker. Old snapshots and reproduction paths remain intact.

Expected C++ changes are limited to `include/geometry_snapshot.h`,
`src/geometry_snapshot.cpp`, `app/run_rgbd_odometry.cpp` and the existing
`tests/test_geometry_snapshot.cpp`; reuse the existing test target. Python then
owns strict loading, descriptor matching controls and the existing metric
validation, with separate loader/matcher regression tests. It will not claim to
replay original frontend match decisions: new descriptor matches are a distinct
offline observation proposal. Export-off/on runs must preserve numerical
trajectory/map state, and descriptor rows/feature IDs/depth must round-trip
exactly. Refuse attaching newly generated descriptors to an old run if IDs/state
differ; keep a separate paired baseline instead. Export does not itself prove
better geometry or tracking.

Alternative: recompute Python descriptors from images. Existing snapshots omit
size/angle/octave for most original features, so this would be a different feature
construction; it cannot recover the exact C++ rows and is not preferred for a
shared-feature comparison. Current cached-singleton recovery is already tested
above and does not supply enough supported geometry. Proposed budget: CPU only,
no downloads, <=8 GiB process address space, <=300 s per initial sequence replay
and <=1 GiB new raw outputs; start with existing development data and freeze
matching settings before any later held-out replay. Stop if export changes state,
identity/descriptor alignment fails, budgets fail, or geometry remains unsupported;
do not alter the original risk policy, BA model or online state to force gains.

### Native ORB feature export and Python matching

**Interface acceptance PASS; cross-cut cycle gate FAIL.** Source
`136ef571cd2552d5827f7e97c21c7c3e4face393` implements the approved four-file C++
interface proposal above. The default frontend/map/BA algorithms are unchanged.
Optional `--export-keyframe-features` on `run_rgbd_odometry` writes each existing
keyframe once, after insertion and before BA. Versioned `AFS_FEATURES 1` sidecars
preserve full native keypoint fields/order, optional map IDs, uint16 depth,
actual RGB/depth association timestamps, relative image references, calibration,
frontend settings and all 32 original descriptor bytes. Existing snapshots keep
their schema. Feature files refuse overwrite, use temporary-file finalization,
and receive a completion marker only at successful run finalization.

Python supplies strict sidecar/provenance checks and a bounded-block NumPy
Hamming matcher using the original C++ defaults (distance <=64, strict ratio
0.75, mutual best). It verifies feature rows against existing map observations
and native depth images, then compares descriptor and LK proposals on identical
full frontend features. New matches are offline proposals, not original frontend
match decisions, landmark merges or feedback. Rigid fitting uses source-feature
row-ID parity, rather than prior map-point-ID parity; both new arms share that
split. The existing metric RANSAC, depth and triangle gates remain fixed.
NumPy/OpenCV/SciPy supply established operations; no original numerical solver
or whole-frontend rewrite is claimed.

The [protocol](results/frontend_feature_protocol.json), SHA-256
`cbb7700feaa762ff96d8a0bf4b15788db4b9f3c4f5c3e11d654350eedffb4cdc`, was frozen
before all real runs, including development. On all 792 xyz frames, feature-off
and feature-on runs of the same clean binary produce byte-identical trajectories,
frame streams and all 227 geometry snapshots. Feature-on runs on desk/room/teddy
also reproduce retained v2 trajectories/frame streams exactly; all 161/491/533
snapshot states agree after normalizing only source-revision metadata. Old/new
binaries differ, so those checks establish exact state reproduction, not a
runtime comparison. No newly generated feature data is attached until these
state checks pass. The same previously seen test sequences are reused.

| Sequence | Keyframes | Native feature rows | Descriptor accepted pairs | Accepted cross-cut pairs | Triangle-qualified cross-cut pairs |
|---|---:|---:|---:|---:|---:|
| xyz development | 114 | 113,947 | 2 / 6 | 0 | 0 |
| desk | 81 | 79,844 | 2 / 6 | 1 | 0 |
| room | 246 | 238,580 | 1 / 6 | 0 | 0 |
| teddy | 269 | 259,676 | 2 / 21 | 1 | 0 |

The LK control accepts zero pairs on these same 39 selected pairs. Descriptor
proposals accept seven pairs in total, including desk 22/24 and teddy 201/204
across weak cuts. The latter spans two cuts but counts as **one unique pair**.
Neither forms the required accepted triangle. Teddy 201/204 validation median
is 0.02947 m (20/39 validation inliers, P95 0.14757 m), close to the frozen
0.03 m inlier threshold; retain its full error
and outlier evidence rather than treating acceptance as reliable identity.
Every positive/negative case is retained in [pair results](results/frontend_feature_pairs.csv)
and [summary results](results/frontend_feature_summary.csv). This supports
availability of real descriptor-driven geometric proposals, not improved global
trajectory, safe map merging, general observability or online tracking. No BA
integration, ATE evaluation or test-time tuning is performed.

All **692,047 feature rows** agree with frontend health feature counts, and all
native depth samples agree exactly with original images. The C++ regression
checks every descriptor byte/keypoint/depth/map ID against native fixture state,
source immutability, overwrite/partial-output/path refusal and unchanged Ceres
replay. **110/110 Python and 17/17 CTest pass, zero skips**. Eleven new Python
tests cover strict schema/alignment, descriptor policies and an end-to-end
synthetic feature/geometry CLI. NumPy matching agrees with OpenCV BF policy on
controlled descriptor data. Initial configure/link failures from existing local
BLAS/LAPACK/glog runtime paths are preserved; the corrected build reuses the
existing sysroot/preload, with no downloads or repository build-system changes.

[Interface costs](results/frontend_feature_interface.csv) record mean export
1.41–1.44 ms and P95 1.50–1.57 ms per keyframe, with 114.5–385.6 ms total per
sequence. Sidecars total about 114.6 MiB. Frontend-on whole-process times are
18.21/11.55/36.77/38.69 s; these include geometry/feature I/O and are not an
algorithmic speedup claim. Python matching processes take 0.77–2.62 s. BLAS/OpenMP environment threads were set to 1; OpenCV internal concurrency was
not explicitly constrained, so these are not guaranteed single-thread timings.
No OOM or timeout occurs. New raw output is 981,708,128 bytes (<1 GiB), peak process RSS
<679 MiB, within the frozen 8 GiB/300 s per-process budgets. No GPU is used.

```bash
# Existing full xyz development sequence; explicit fresh output locations.
./build/run_rgbd_odometry /path/to/xyz /path/to/new.tum 792 - none \
  /path/to/new_cache --export-keyframe-features
PYTHONPATH=python python python/audit_frontend_features.py \
  --geometry /path/to/state_matched_shared.snapshot \
  --features /path/to/new_cache/features --dataset /path/to/sequence \
  --output /path/to/new_descriptor_audit
```

Raw evidence remains outside Git in `frontend-features-v9-20261021`: 2,538 files,
manifest SHA-256 `6ecc104c81dfff66ece31e90396e16812963471e235590938a0142c75cab80c3`;
binary/source hashes and exact controls are in the frozen protocol. Cumulative
contribution adds the minimal native-data interface, substantive Python Hamming
policy/validation and a controlled geometric proposal comparison. Reliable
multi-view support and trajectory usefulness remain unproven; the accepted
pair proposals are not inserted into the map or fed to BA/online tracking.

## Fixed-proposal independent evaluation

**Both cross-cut proposals fail the frozen evaluation thresholds.** Preserve
native feature export and the original proposal evidence, but do not use these
pairs to repair weak map connections. This step adds Python evaluation only;
frontend tracking, map state, BA and trajectories are unchanged.

The [frozen protocol](results/candidate_validation_protocol.json), SHA-256
`e9bfe2dd96f75d42afa9998eb0a3ff0eb154fba87d439b82d10a99e603b9c1aa`, evaluates
all seven accepted descriptor models from the preceding audit: two cross-cut
primary cases and five ordinary-pair controls. No refitting, new matching,
threshold changes or rejected-model reconsideration occurs. All 2,538 prior
raw evidence files are hash-verified before and after evaluation. These reuse
previously seen scenes; they are not new held-out scenes.

Python projects a fixed 16-pixel depth grid in both directions, excluding a
4-pixel box around every original ORB depth location in both endpoint images.
Native depth validity and a fixed 3x3 spread check precede projection. Every
remaining valid signed depth residual is retained, including possible
occlusions; there is no ICP, residual-based visibility rejection or pose update.
Each direction requires >=100 comparisons, >=30% comparison/source coverage,
>=70% agreement within 3 cm and median absolute depth error <=2 cm. These are
predeclared engineering thresholds, not a universal accuracy standard.

`depth-only.json` is finalized before ground truth is opened. Evaluation then
compares each fixed camera-to-camera transform with TUM truth at RGB timestamps,
using both nearest-unused association within 20 ms and separately bracketed
translation/Slerp interpolation without extrapolation. Relative transforms need
no world-frame or scale alignment. Fidelity requires <=3 cm translation and
<=2 degrees rotation in both timestamp protocols, both depth checks passing,
and no sensitivity above 5 mm/0.5 degrees or changed fidelity classification.
These truth-based labels are **evaluation only**, never edge selection or BA
admission. Raw baseline transforms come from the actual exported frame poses,
not optimized keyframe snapshots.

| Primary pair | Raw baseline translation / rotation error | Candidate translation / rotation error | Forward / reverse depth agreement | Frozen evaluation |
|---|---:|---:|---:|---|
| desk 22 -> 24 | 0.03236 m / 1.411 deg | 0.10379 m / 6.345 deg | 66.78% / 64.23% | fail |
| teddy 201 -> 204 | 0.95741 m / 22.334 deg | 0.13077 m / 3.648 deg | 68.35% / 65.87% | fail |

The table uses nearest association; interpolated candidate errors are
0.10439 m / 6.499 degrees for desk and 0.12943 m / 3.454 degrees for teddy,
so neither failure changes under the second timestamp protocol. Teddy improves
substantially over its anomalous raw baseline but still misses absolute
accuracy and depth thresholds. Desk degrades both translation and rotation.
Neither supplies the previously missing third-view cycle support.

All seven cases are evaluable, with no declared timestamp sensitivity.
Three of five ordinary controls meet fidelity checks; only xyz 23/24 and room
143/144 also meet the predeclared pair-level improvement criterion. The other
three controls, including every degradation, remain in the
[complete results](results/candidate_validation_results.csv) alongside both
primary failures and both timestamp protocols. Teddy 201/202 passes both depth
checks but has 5.88 cm translation error, a real example of depth agreement
being insufficient to establish pose accuracy. Synthetic tests also demonstrate
a planar tangential-motion blind spot; this does not establish the specific
cause of every real failure.

**121/121 Python tests and 17/17 CTest pass, zero skips.** Eleven new tests check
transform direction/world-frame invariance, depth/scale, unused sample masks,
coverage failure, the planar blind spot, timestamp interpolation boundaries,
paired improvement criteria, overwrite refusal and preserved pre-truth evidence
when truth is unavailable. Python owns the projection, provenance, evaluation
protocol and failure accounting; NumPy/SciPy/OpenCV supply standard numerical,
rotation and image operations. No novel solver is claimed.

```bash
PYTHONPATH=python python python/evaluate_candidate_proposals.py \
  --matching /path/to/frozen_descriptor_audit \
  --features /path/to/feature_cache/features \
  --frames /path/to/feature_cache/frames.txt \
  --dataset /path/to/sequence --truth /path/to/sequence/groundtruth.txt \
  --output /path/to/fresh_evaluation
```

Four serial evaluation processes complete in 0.62-0.97 s each, peak RSS below
103 MiB, with no timeout/OOM or GPU use. These are fixed-model evaluation costs,
not frontend/optimization/system-speed results. Raw evidence remains outside
Git in `candidate-validation-v10-20261021`: 31 files, 142,433 bytes, manifest
SHA-256 `dc6aea42c1bc2146c841922c1dac51a9c13aeac7f5ddb012784e2c2b2d1a5305`.
The cumulative contribution now includes a reproducible way to falsify plausible
geometric proposals before integration. Stop this frozen candidate experiment;
retain its failures and interface work. No trajectory ATE gain, reliable bridge,
map merge, online feedback or fresh cross-scene generalization follows from
these pair-level results.

## Fixed-model error diagnostics

Python now describes all seven frozen proposals without fitting, matching,
changing thresholds or altering BA admission. Original depth-valid records and
3 cm inlier masks reproduce exactly. The
[diagnostic protocol](results/proposal_diagnostics_protocol.json) freezes source
hashes, all/even-fit/odd-validation/original-inlier groups and the two existing
RGB/depth endpoint timestamp alternatives before this descriptive run. This is
post-hoc analysis of previously seen results, not a new held-out experiment.

An exact identity explains why low point residuals can accompany large
camera-origin pose errors: at source centroid `c`, disagreement with truth is
`(R_candidate - R_truth) c + (t_candidate - t_truth)`. Rotation and translation
terms can nearly cancel over the observed support.

| Cross-cut pair | Translation disagreement | Rotation lever arm at original inlier centroid | Remaining centroid disagreement | Candidate 3D inlier median |
|---|---:|---:|---:|---:|
| desk 22 -> 24 | 0.10439 m | 0.10846 m | 0.01031 m | 0.01080 m |
| teddy 201 -> 204 | 0.12943 m | 0.15387 m | 0.02694 m | 0.01648 m |

Desk's 29 original inliers span only 174x66 pixels in four of sixteen image
cells; their centered RMS axes are 0.0880/0.0506/0.0073 m. Teddy's 50 inliers
cover nine cells with axes 0.5827/0.5392/0.0795 m. These spectra describe support,
not full-system rank or proof of a unique cause. The algebraic compensation is
observed relative to interpolated RGB truth; it does not prove which measurement
or fitting error produced it.

The endpoint RGB-minus-depth offsets are 15.71/15.93 ms for desk and
7.42/7.75 ms for teddy. Evaluating the unchanged candidate against interpolated
truth at the existing depth timestamps gives 0.09986 m / 6.465 degrees and
0.12650 m / 3.045 degrees respectively: both still fail the original absolute
thresholds. Depth-timestamp substitution is diagnostic only because the depth
images are registered to RGB; it is not a corrected measurement model. No
clock-offset search or calibration adjustment is performed. The inverse-direction
negative control has rotation errors 73.76/27.63 degrees, inconsistent with
explaining these failures by simply inverting the proposals.

[All seven case summaries](results/proposal_diagnostics_results.csv) preserve
3D residuals, pixel reprojection, depth-independent symmetric epipolar distance,
support geometry, timestamp alternatives and inverse controls. Raw reports also
retain every per-record error and all four fixed groups. All use the existing
zero-distortion pinhole model; epipolar consistency alone cannot certify match
identity. Calibration/distortion, depth registration/error, timing and mismatches
remain possible confounded contributors. No unique root cause, repaired pose,
new trajectory gain or online benefit is established.

```bash
PYTHONPATH=python python python/diagnose_candidate_proposals.py \
  --evaluation /path/to/fixed_evaluation/report.json \
  --matching /path/to/frozen_descriptor_audit \
  --features /path/to/feature_cache/features --dataset /path/to/sequence \
  --truth /path/to/sequence/groundtruth.txt --output /path/to/fresh_diagnostics
```

**126/126 Python tests pass, zero skips**; five new mathematical tests cover
transform direction, projection, depth-independent epipolar geometry, pure
rotation/behind-camera non-evaluability, centroid cancellation and support
accounting. C++ is unchanged and its prior 17/17 result is retained, not rerun
in this step. NumPy/SciPy/OpenCV provide standard operations; the Python
contribution is the diagnostic model, frozen-record checks and evaluation
accounting. Four serial processes take 0.51-0.77 s each with peak RSS below
100 MiB, no timeout/OOM and no GPU. These are diagnostic costs, not SLAM speed.
Raw evidence stays outside Git in `proposal-diagnostics-v11-20261021`: 25 files,
1,028,591 bytes, manifest SHA-256
`0d70e2d7ca90f971c1cdf9b58cd5be96919b0113dee36b4b44fa6883b2bf6944`.
The original proposals, earlier evidence and trajectories remain unchanged.
A subsequent development-only comparison of source-depth 3D-2D modeling against
symmetric 3D-3D fitting could isolate sensitivity to target depth, but has not
been implemented or tested here; it must use a new declared configuration and
cannot revise the completed frozen test's conclusions.

## Development-only pair objective comparison

The preceding proposed 3D-2D comparison is now implemented in Python, preserving
C++ and using only the xyz development sequence. The
[frozen protocol](results/pair_model_dev_protocol.json), SHA-256
`a0e5f3213574f901b381c3110b66061c93fdb381296470716135386deb5a06aa`, includes all
six previously selected descriptor pairs, including four original rejections.
Even source-feature IDs fit; odd IDs validate. Neither arm uses truth, validation
rows or the original RANSAC inlier mask to choose fitting observations.

Both arms use the same six-DoF parameterization, SciPy TRF solver, numerical
Jacobian, block Huber policy, 100-evaluation limit and failure rollback. The
primary comparison starts both from identical raw frame poses; a secondary
control starts both from each available frozen 3D-3D proposal. The objectives
are metric point differences /0.01 m versus pixel reprojection /2 pixels, with
Huber delta 2. Different noise assumptions remain a confound; these are matched
robust refinement arms, **not the original RANSAC algorithm**. 3D-2D does not
consume target depths during fitting, but its shared input pool still uses the
original target-depth eligibility filter. No novel solver is claimed: Python
owns residual modeling, common data/initialization, checks, rollback and analysis;
NumPy/SciPy provide standard numerical operations.

| xyz pair, raw initialization | Refined 3D-3D translation / rotation | Refined 3D-2D translation / rotation | Outcome |
|---|---:|---:|---|
| 23 -> 24 | 0.07147 m / 4.931 deg | 0.02814 m / 1.163 deg | qualified development improvement |
| 23 -> 25 | unchanged raw pose | unchanged raw pose | both reject: only 4 fit rows |
| 23 -> 26 | unchanged raw pose | unchanged raw pose | both reject: only 3 fit rows |
| 24 -> 25 | 0.02961 m / 1.373 deg | 0.03593 m / 1.443 deg | degradation, misses 3 cm threshold |
| 24 -> 26 | 0.18682 m / 8.555 deg | 0.15460 m / 8.103 deg | both remain inaccurate |
| 25 -> 26 | 0.02559 m / 1.718 deg | 0.00814 m / 0.978 deg | qualified development improvement |

The table uses interpolated truth; both qualified pairs also meet the declared
3 cm / 2 degree bounds and improvement criteria under nearest association.
Across the four shared converged primary pairs, translation-error medians are
0.05054 -> 0.03203 m and rotation-error medians 3.325 -> 1.303 degrees. Neither
model adds failures relative to the other; both retain two insufficient-support
cases. The predeclared **development continuation criterion passes**, but these
small, previously seen pairs do not establish robust performance. Secondary
initialization produces similar results, without selecting the better run using
truth. Full [32 evaluation rows](results/pair_model_dev_results.csv) retain every
arm, rejection and both timestamp protocols.

Against the original frozen RANSAC proposals, 23/24 is worse: translation
0.02557 -> 0.02814 m and rotation 0.225 -> 1.163 degrees. Pair 25/26 improves:
0.01578 -> 0.00814 m and 1.497 -> 0.978 degrees. Thus the matched-refinement gains
cannot be presented as uniformly improving the existing full proposal pipeline.
A controlled synthetic target-depth perturbation changes only the 3D-3D fit;
it establishes objective dependence, not that target-depth error uniquely caused
the real failures. No new trajectory, map, BA, online or test-sequence run occurs.

```bash
PYTHONPATH=python python python/compare_pair_models.py \
  --matching /path/to/xyz/frozen_descriptor_audit \
  --features /path/to/xyz/feature_cache/features \
  --frames /path/to/xyz/feature_cache/frames.txt --dataset /path/to/xyz \
  --truth /path/to/xyz/groundtruth.txt --output /path/to/fresh_comparison
```

Fit-only evidence is finalized before reading truth; failures return the full
original pose. **131/131 Python tests pass, zero skips**. Five new tests cover
noiseless recovery, target-depth perturbation isolation, noisy/outlier pixels,
invalid/collinear/behind-camera support, input immutability and solver-budget
rollback. C++ is unchanged; its previous 17/17 result is retained, not rerun.
All 16 arms complete in one 1.02 s process with peak RSS below 111 MiB, no
OOM/timeout and no GPU; this is pair-comparison cost, not SLAM runtime. Raw
evidence remains outside Git in `pair-model-dev-v12-20261021`: 14 files,
188,761 bytes, manifest SHA-256
`ab114a9c15a2010b49482ea50b11fab0376ef02ea8455d8da69f1fa496fb0a7f`.
All earlier evidence and configurations remain unchanged. The next justified
step is to propose a separately frozen validation of this exact model; reusing
already inspected desk/room/teddy would be a reused-scene comparison, not fresh
independent validation. No BA or tracking integration is justified yet.

## Frozen reused-scene pair-model validation

**Overall acceptance FAIL.** The exact development implementation/configuration
is frozen and replayed on all 6 desk, 6 room and 21 teddy descriptor pairs,
including original rejected cases. These scenes were previously inspected;
this is a reused-scene comparison, not fresh independent validation. No fitting
code, thresholds, noise scales, split, initialization policy or solver budget
changes. Python supplies the already described objectives, checks and experiment
accounting; SciPy remains the numerical solver. The
[protocol](results/pair_model_reused_protocol.json), SHA-256
`bd73fd4adac569d8eeafd847c63891240e5810fada63aab163398c11d329edc8`, is frozen
before all three runs. Truth is opened only after each fit-only file is finalized.

| Reused scene | All pairs | Shared converged primary pairs | Median translation 3D-3D -> 3D-2D | Median rotation 3D-3D -> 3D-2D | Primary failures/rollbacks 3D-3D -> 3D-2D |
|---|---:|---:|---:|---:|---:|
| desk | 6 | 3 | 0.03718 -> 0.05295 m | 1.740 -> 3.302 deg | 1 -> 3 |
| room | 6 | 3 | 0.14995 -> 0.05212 m | 8.099 -> 1.530 deg | 3 -> 3 |
| teddy | 21 | 13 | 0.16208 -> 0.09069 m | 2.945 -> 2.436 deg | 8 -> 8 |

Medians use interpolated truth and only pairs where both primary refinements
converge; failed outputs retain their initial poses and remain in the full
results rather than being counted as successful improvements. Nearest association
has the same scene-level outcome. These are pair-refinement failures, not online
tracking failures. Desk violates the declared 5 mm/0.5 degree non-degradation
allowance and has two additional 3D-2D failures: 21/23 encounters a nonpositive
trial prediction, and 22/24 exhausts 100 evaluations. Both return the original
pose completely. The alternative frozen-proposal initialization also exhausts
its budget on desk 22/24; no best initialization is selected using truth.

Room and teddy meet the declared median-benefit criterion, and teddy 157/158
and 157/159 meet both timestamp protocols' absolute accuracy/improvement labels.
The latter spans cut 159 and has interpolated error 0.02437 m / 1.347 degrees,
versus 0.04628 m / 1.771 degrees for its raw initial pose. This positive pair
must remain visible alongside the failures. Its label is evaluation only: no
truth-based measurement selection, independent depth certificate, triangle
certificate, identity merge or BA admission follows from it.

Important retained failures include teddy 201/203: 3D-2D numerically converges
to **3.686 m / 109.72 degrees** error, compared with a raw initial error of
0.699 m / 17.23 degrees; held-out pixel median is 56.21 pixels. Solver convergence,
lower fitting cost and a full local Jacobian therefore do not guarantee a useful
pose. Teddy 201/204 improves over the matched 3D-3D refinement but remains
0.11917 m / 3.126 degrees, outside the accuracy bounds. Desk's original cross-cut
candidate is not repaired. Original-pipeline differences also remain: room
143/144's original RANSAC proposal is 0.00450 m / 0.827 degrees, whereas this
3D-2D refinement is 0.01714 m / 1.424 degrees. A matched-objective gain cannot
be presented as uniformly improving the original full RANSAC pipeline.

The [complete results](results/pair_model_reused_results.csv) contain all 82
arms and both truth protocols (164 rows), including skipped/rolled-back outputs,
raw initial errors and available original accepted-RANSAC errors. The four
predeclared criteria yield: two scenes with median benefit PASS, at least one
cross-cut pair improvement PASS, all scenes non-degrading FAIL, and no additional
3D-2D failures FAIL. Overall acceptance remains FAIL; do not integrate this
configuration into BA or tracking or tune it on these reused scenes.

All three processes finish successfully in 1.07-3.22 s each, peak RSS below
112 MiB, no OOM/timeout or GPU. Process success is distinct from failed geometry
acceptance, and these costs are not whole-system speed measurements. Source and
all V9/V12 raw-evidence hashes remain unchanged. No implementation changed, so
the existing 131/131 Python tests (zero skips) and earlier 17/17 CTest results
are retained, not rerun in this step. No fresh test-pass count is claimed.
Raw evidence is outside Git in `pair-model-reused-v13-20261021`: 23 files,
714,986 bytes, manifest SHA-256
`d40c2696834f12ef7ad49c035bb43f614f51ad296e633f3685333848275ea6af`.
Use the preceding comparison entry point with each scene's explicit input/output
paths; never overwrite an earlier output directory. No data download, trajectory
correction, ATE experiment, risk scheduling or online writeback occurs. The
cumulative contribution is a reproducible model comparison that exposes both
local gains and unsafe accepted solutions; stable geometric/trajectory benefit
is still unproven. Any next rejection/rollback study belongs on development
inputs under a new protocol, not a revision of these frozen results.

## Development-only pair update guard

**Development prototype checks PASS; stable accuracy gain remains unproven.**
Python adds an opt-in gate for the six cached xyz primary 3D-2D updates, leaving
all V12/V13 fitting code, thresholds and evidence unchanged. No new fitting or
scene replay occurs. The [frozen protocol](results/pair_update_guard_dev_protocol.json),
SHA-256 `86dc4b536437e5807779d3c4c349b965974f3751fc4f85a32bcd3ed1cb0838f7`,
requires an already successful solve, at least six check correspondences, positive
projections, median pixel error <=4 pixels and no increase over the raw initial
median beyond 1e-9 pixels. The 4-pixel bound is the existing V12 2-pixel sigma
times Huber delta 2; it is not calibrated on desk/room/teddy. No alternate
threshold is tried after this result.

Odd source-feature rows, previously held out from fitting, now supply the gate.
They become **decision inputs, not independent geometric validation**. Source
native depths and target pixels are checked against hashed frozen records;
target depth remains an inherited eligibility dependency. Every rejected update
returns the full raw initial transform exactly. This is pair-transform rollback,
not online state repair. Original proposals, maps and trajectories are never
mutated. `guard-only.json` is finalized before the recorded V12 truth evaluation
is opened; truth is used only to assess the completed decisions.

| xyz pair | Decision | Interpolated translation error, unguarded -> selected |
|---|---|---:|
| 23 -> 24 | accept: 2.45 px check median | 0.02814 -> 0.02814 m |
| 23 -> 25 | retain solver rollback | 0.08154 -> 0.08154 m |
| 23 -> 26 | retain solver rollback | 0.09133 -> 0.09133 m |
| 24 -> 25 | reject: 4.75 px exceeds bound | 0.03593 -> 0.04372 m |
| 24 -> 26 | reject: 25.33 px exceeds bound | 0.15460 -> 0.05328 m |
| 25 -> 26 | reject: 1.39 px worsens initial median | 0.00814 -> 0.01080 m |

All six cases, all rejected outputs and both timestamp protocols are retained in
[complete results](results/pair_update_guard_dev_results.csv). The one accepted
update improves raw 23/24 from 0.03638 m / 2.082 degrees to 0.02814 m /
1.163 degrees, meeting the predeclared absolute and meaningful-gain criteria
in both timestamp protocols. Two inaccurate converged candidates are rejected,
three rejected converged transforms roll back exactly, and two existing solver
failures stay unchanged. The declared development criteria pass without
rejecting every candidate.

The all-six returned-pose translation-error median falls from unguarded
0.05874 m to 0.04850 m, **equal to the raw baseline median**. It does not improve
the baseline aggregate. Rejection discards a real modest improvement on 25/26,
and makes 24/25 truth error worse than its unguarded candidate. A low median can
also hide a substantial minority of incorrect checks, as an explicit synthetic
test demonstrates. Neither accepted nor rejected labels certify match identity,
independent accuracy, global observability or a full-system benefit. Earlier
3D-2D scene failures, including the extreme teddy error, have not been replayed
through this gate and cannot be claimed fixed.

```bash
PYTHONPATH=python python python/check_pair_updates.py \
  --comparison /path/to/xyz/comparison/fit-only.json \
  --matching /path/to/xyz/frozen_descriptor_audit \
  --features /path/to/xyz/feature_cache/features \
  --evaluation /path/to/xyz/comparison/report.json --output /path/to/fresh_guard
```

**138/138 Python tests pass, zero skips.** Seven new tests cover useful-update
acceptance, absolute and relative pixel rejection, insufficient/nonfinite/behind-
camera data, exact rollback and input immutability, the median's outlier blind
spot, unchanged decisions under altered truth labels, overwrite refusal and
preserved guard-only evidence when evaluation is missing. C++ is unchanged;
its prior 17/17 result is retained, not rerun. NumPy supplies established geometry
operations; Python owns the guard policy, provenance checks and rollback. The
six-case cached replay takes 0.27 s and peak RSS below 68 MiB, without GPU,
OOM or timeout; this is replay/provenance cost, not an online SLAM speed claim.
Raw evidence stays outside Git in `pair-update-guard-dev-v14-20261021`: 14 files,
78,269 bytes, manifest SHA-256
`e199c5ae5aabefaa7f3befc06d9c7302c261e90061da4202c4172e2be4d7ac6c`.
The cumulative contribution adds explicit truth-free update decisions and exact
pair rollback with measured tradeoffs. Any subsequent scene replay must freeze
this exact gate first, retain every rejection/degradation and be labeled reused-
scene evaluation; no BA/online integration follows from this development result.

## Frozen reused-scene update-guard validation

**Overall acceptance FAIL.** The exact development gate is frozen and applied
to every cached primary 3D-2D attempt on desk/room/teddy (6/6/21 pairs). No solver,
source, split, noise scale or gate threshold changes. This is a reused-scene
check of known data, not fresh independent validation. The
[protocol](results/pair_update_guard_reused_protocol.json), SHA-256
`5c1a9769e01bf1c1559765feb4bdaf3a61641d44938c82422a01fd0a133d4c4d`, is finalized
before the three runs. Each guard-only file is finalized before reading the
recorded truth evaluation. Odd check pixels are decision inputs, not independent
holdout. Truth never chooses which updates are returned.

| Scene | All attempts | Accepted | Accepted meeting 3 cm / 2 deg in both time protocols | Accepted with material regression against raw |
|---|---:|---:|---:|---:|
| desk | 6 | 2 | 0 | 2 |
| room | 6 | 2 | 1 | 1 |
| teddy | 21 | 9 | 2 | 1 |

Thirteen updates pass the truth-free gate, but ten miss the absolute accuracy
bounds and four worsen raw error by more than 5 mm or 0.5 degrees under at
least one timestamp protocol. For example, desk 22/23 passes the check yet
changes interpolated truth error from 0.02094 m / 0.970 degrees to 0.05295 m /
3.302 degrees. Room 141/142 changes 0.01196 m / 0.722 degrees to 0.05212 m /
1.530 degrees. Positive results remain: teddy 157/158 and 157/159 meet the
meaningful-improvement and accuracy labels against raw in both time protocols.
Only teddy supplies such gains, failing the predeclared two-scene benefit rule.
These labels are evaluation results, not a truth-based edge-selection policy.

| Scene | All-returned-pose translation median raw -> unguarded -> guarded | Rotation median raw -> unguarded -> guarded |
|---|---:|---:|
| desk | 0.03982 -> 0.05083 -> 0.05083 m | 2.238 -> 3.187 -> 3.187 deg |
| room | 0.01085 -> 0.01344 -> 0.01344 m | 1.073 -> 1.585 -> 1.477 deg |
| teddy | 0.05277 -> 0.08619 -> 0.04173 m | 1.899 -> 2.643 -> 1.662 deg |

Medians above use interpolated truth and **all** attempts, including exact raw
fallbacks, avoiding success-only aggregation. Requiring both timestamp protocols
makes desk and room fail the scene-level non-degradation criteria. Room's interpolated
median is within the allowances, but nearest association worsens rotation by
0.503 degrees, just over the frozen 0.5-degree limit; retain this boundary case
rather than rounding it into a pass. Teddy's all-attempt medians improve, but
its accepted 201/204 update remains 0.11917 m / 3.126 degrees despite passing
the pixel gate. Partial benefit does not establish safe or stable operation.

All 20 rejected/failed transforms return the raw initial pose exactly. The known
teddy 201/203 extreme error is rejected for insufficient check observations,
not because the pixel bound was measured successfully: the 3.686 m /
109.72-degree fitted result is replaced by a still-bad raw 0.699 m /
17.23-degree transform. This demonstrates rollback, not correction of lost
tracking. The [complete 66 evaluation rows](results/pair_update_guard_reused_results.csv)
retain every decision, failure, check count, measured pixel median and both
truth protocols. All four geometry/benefit acceptance conditions fail: accepted
accuracy, accepted non-degradation, multi-scene benefit and all-case median
non-degradation. Accounting, exact rollback, nontrivial coverage and the known
extreme-error rejection pass. Overall acceptance stays FAIL.

No implementation changes or new dependency are introduced in this step.
Python supplies the frozen gate and provenance checks described above; NumPy
supplies standard geometry operations. **138/138 Python regression tests pass,
zero skips**, rerun on the publication candidate. C++ is unchanged; its earlier
17/17 result is retained, not rerun. Three cached processes take 0.22-0.47 s
each, peak RSS below 68 MiB, no GPU/OOM/timeout; these are cache-check costs,
not whole-system runtime. V13/V14 source and raw-evidence hashes remain unchanged.
Raw evidence stays outside Git in `pair-update-guard-reused-v15-20261021`:
21 files, 174,417 bytes, manifest SHA-256
`80e0d147035b713748e74614912519e81366fc9fc1523f41a03a6c0667f14273`.
Use the preceding gate entry point with each scene's cached comparison and fresh
output directory. Stop promoting or tuning this gate configuration on these
scenes. Preserve its failure evidence and Python interface work; no threshold
relaxation, BA/online integration, map merge, trajectory correction or new ATE
claim follows. The cumulative contribution is reproducible offline geometry,
model diagnosis and falsifiable update validation, with stable cross-scene
trajectory improvement still an open project requirement.
The publication whitespace check initially rejected CRLF endings in the six new
CSV tables. Only publication copies are normalized to LF; parsed records and
original raw evidence remain identical. The failed check is retained outside Git.
