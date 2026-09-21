# AdaptiveFusion-SLAM

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
