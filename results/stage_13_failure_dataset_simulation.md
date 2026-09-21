# Stage 13 Experiment: Simulated Future-Failure Dataset

## Configuration

- Synthetic health frames: 24
- History length `L`: 5 frames
- Prediction horizon `H`: 3 frames
- Degradation interval: frames 9–15
- Forced tracking failure: frame 15
- Successful recovery: frames 16–23

The simulated degradation lowers inlier ratio, spatial coverage, parallax, and
valid-depth ratio while increasing reprojection and LK forward-backward errors.
ORB fallback activates at frames 13–15.

## Result

```text
Usable samples:                   11
Positive future-failure samples:  3
Negative samples:                 8

Anchor 12: failure in 3 frames, inlier ratio 0.578571,
           reprojection error 1.84286 px, fallback 0
Anchor 13: failure in 2 frames, inlier ratio 0.485714,
           reprojection error 2.22857 px, fallback 1
Anchor 14: failure in 1 frame,  inlier ratio 0.392857,
           reprojection error 2.61429 px, fallback 1

CTest: 12/12 passed
```

The generated dataset is stored in
`results/stage_13_simulated_failure_dataset.csv`. Each row contains the anchor,
future label, lead time, and 35 flattened inputs (`5 frames × 7 features`).

This simulation validates causal window construction and label alignment. It
does not measure prediction performance because the labels are generated but no
model has yet been trained.
