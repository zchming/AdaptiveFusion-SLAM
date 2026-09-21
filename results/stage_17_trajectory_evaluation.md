# Stage 17 Experiment: TUM Trajectory Evaluation

## Evaluator Verification

A ten-pose synthetic trajectory differs from ground truth only by a global
rigid coordinate transform. SE(3) alignment yields ATE `2.89389e-16 m` and
negligible RPE. Added nonlinear drift yields ATE `0.00661266 m`, translation
RPE `0.0248341 m`, and rotation RPE `0.010401 rad`.

## Real Sequence

- Dataset: TUM RGB-D `rgbd_dataset_freiburg1_xyz`
- Evaluated prefix: first 200 RGB-D frames
- Timestamp tolerance: 0.02 seconds
- Camera calibration: Freiburg1 defaults already used by the runner
- Adaptive model: Stage 16 synthetic logistic model

## Paired Result

```text
metric                      baseline       adaptive
processed frames                 200            200
valid trajectory poses           200            200
matched ground-truth poses       198            198
estimated match ratio           0.99           0.99
duration coverage           0.227318       0.227318
ATE RMSE (m)               0.0241339       0.024611
RPE translation RMSE (m) 0.00492338     0.00498389
RPE rotation RMSE (rad)  0.00610731     0.00611786
keyframes                         17             18
map points                     11950          11295
local BA runs                     16             17
```

Adaptive decisions were 178 low, 2 medium, 3 high, and 17 critical; 17 frames
froze map updates. The map contains 655 fewer points, but ATE and RPE are
slightly worse. This is expected to be inconclusive because the risk model was
trained on synthetic trends and the experiment covers only 22.7% of the full
ground-truth duration.

The machine-readable comparison is in
`results/stage_17_fr1_xyz_200_comparison.csv`. Raw trajectories are generated
locally and excluded from the repository.
