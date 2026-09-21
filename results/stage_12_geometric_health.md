# Stage 12 Experiment: Per-Frame Geometric Health

## Environment

- Ubuntu 24.04
- GCC 13.3.0
- CMake 3.28.3
- Eigen3
- OpenCV 4.6.0
- Ceres Solver 2.2.0
- Release build with `-Wall -Wextra -Wpedantic`

## Controlled Input

Four correspondences each move by `(3, 4)` pixels, giving 5-pixel parallax.
They occupy four cells of a 4-by-3 grid. Three are PnP inliers, two have valid
depth, and two LK tracks have forward-backward errors of 0.2 and 0.4 pixels.

## Result

```text
Correspondences:              4
PnP inliers:                  3
PnP inlier ratio:             0.75
Mean reprojection error:      0.8 pixels
Mean forward-backward error:  0.3 pixels
Occupied-grid coverage:       4/12 = 0.333333
Median parallax:              5 pixels
Valid-depth ratio:            2/4 = 0.5
CSV serialization:            passed
Odometry integration:         passed
CTest:                        11/11 passed
```

The tiny three-frame repository fixture writes one initialization row and two
lost-frame rows to the health CSV. This experiment verifies signal computation
and retention, not their predictive power. Predictive evaluation requires
natural sequences, temporal windows, future labels, and calibrated models.
