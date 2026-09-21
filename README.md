# AdaptiveFusion-SLAM

**Proactive Tracking-Failure Forecasting and Risk-Adaptive Map Updating for Robust RGB-D SLAM**

AdaptiveFusion-SLAM is a research-oriented RGB-D visual SLAM system designed to predict tracking degradation before complete failure and proactively protect the map from unreliable observations.

The project is implemented incrementally from a minimal SLAM pipeline instead of directly modifying an existing complete system. Its long-term goal is to provide a reproducible platform for studying tracking-failure forecasting, risk-aware frontend control, and selective map updating.

> Current status: early development. The research hypotheses and modules described below have not yet been experimentally validated.

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
- [ ] Implement local bundle adjustment
- [ ] Record per-frame geometric health signals
- [ ] Build the tracking-failure event dataset
- [ ] Implement and calibrate the temporal risk predictor
- [ ] Add risk-adaptive frontend and map-update policies
- [ ] Perform benchmark and ablation experiments
- [ ] Release reproducible results and documentation

## Build

The project currently requires:

- Ubuntu 24.04
- C++17
- CMake
- Eigen3
- OpenCV 4.6

OpenCV is used for RGB-D image loading, ORB features, descriptor matching, and
LK optical flow. g2o, Ceres, and additional dependencies will be enabled as
their corresponding modules are implemented.

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
