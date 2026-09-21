# Stage 18: Reproducible v0.1 Benchmark

## Goal and Physical Meaning

This stage closes the first runnable research prototype. It measures whether
the complete system can process a real RGB-D sequence, whether proactive
control changes map growth and compute cost, and how the pipeline behaves when
the RGB sensor is deliberately degraded while depth remains unchanged.

The degradation layer is placed immediately after image loading. Keeping the
depth image fixed isolates loss of visual information from loss of geometric
range data. Each transform is deterministic for a frame index, so baseline and
adaptive runs receive identical input.

## Key Variables

- `degradation`: `blur`, `dark`, `occlusion`, `noise`, or `drop`.
- `lost_frames`: frames for which neither LK nor ORB/PnP yields a valid pose.
- `recovery_events`: transitions from one or more lost frames back to tracking.
- `mean_recovery_frames`: mean length of completed lost-frame streaks.
- `tracking_success_rate`: successful or initialized frames divided by inputs.
- `p95_frame_ms`: frame-processing time below which 95% of frames fall.
- `map_points`: persistent landmarks after all risk-dependent map decisions.

## Reproduction

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

./build/run_rgbd_odometry DATASET baseline.txt
./build/run_rgbd_odometry \
    DATASET adaptive.txt 10000 results/stage_16_simulated_risk_model.txt
./build/compare_slam_trajectories \
    DATASET/groundtruth.txt baseline.txt adaptive.txt comparison.csv

./build/run_rgbd_odometry DATASET blur.txt 200 - blur
./build/run_rgbd_odometry \
    DATASET blur_adaptive.txt 200 \
    results/stage_16_simulated_risk_model.txt blur
```

Repeat the last pair with `dark`, `occlusion`, `noise`, and `drop`.

## Full TUM Freiburg1 XYZ Result

| Metric | Baseline | Adaptive |
|---|---:|---:|
| Processed / valid poses | 792 / 792 | 792 / 792 |
| Tracking success | 100% | 100% |
| Matched ground truth | 790 | 790 |
| Ground-truth duration coverage | 88.30% | 88.30% |
| ATE RMSE | 0.062081 m | 0.062442 m |
| RPE translation RMSE | 0.005518 m | 0.005537 m |
| RPE rotation RMSE | 0.006743 rad | 0.006745 rad |
| Keyframes | 114 | 92 |
| Map points | 79,790 | 63,983 |
| Local BA runs | 113 | 90 |
| Mean frame time | 60.07 ms | 47.91 ms |
| P95 frame time | 214.25 ms | 164.14 ms |
| Throughput | 16.65 FPS | 20.87 FPS |

Adaptive mode reduces keyframes by 19.3%, map points by 19.8%, and measured
runtime by 20.2%. ATE is 0.58% higher. The result demonstrates selective map
updating and lower compute cost on this sequence; it does not demonstrate a
trajectory-accuracy improvement.

## Controlled Degradation Result

Each row below uses the first 200 synchronized frames and severity 0.7. The
complete machine-readable table includes recovery and runtime fields.

| Input | Mode | Success | ATE (m) | Keyframes | Map points | FPS |
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

The every-tenth-frame drop creates 20 lost frames and 19 completed one-frame
recoveries in each mode. The current controller protects the map after failure
but cannot recover the deliberately removed image itself, so tracking success
does not rise. That distinction is central: proactive map protection and
failure prevention are separate claims and require separately calibrated real
data.

## Verification and Scope

Release compilation passes all 16 tests. The degradation test verifies that
every mode changes RGB dimensions safely while preserving the exact depth
image. Full numerical data are in:

- `stage_18_fr1_xyz_full_comparison.csv`
- `stage_18_fr1_xyz_full_runtime.csv`
- `stage_18_degradation_comparison.csv`

Version 0.1 is a complete research prototype and a credible portfolio project:
it owns the RGB-D frontend, pose estimation, sparse map, local optimization,
health logging, temporal predictor, causal controller, adaptive intervention,
evaluation, and simulation path. It is not yet a production replacement for
ORB-SLAM3 or RTAB-Map. Real training/calibration, relocalization, loop closure,
multi-dataset evaluation, and hardware integration remain follow-up research.
