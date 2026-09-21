# Stage 11 Experiment: Local Bundle Adjustment

## Environment

- Ubuntu 24.04
- GCC 13.3.0
- CMake 3.28.3
- Eigen3
- OpenCV 4.6.0
- Ceres Solver 2.2.0 with SuiteSparse
- Release build with `-Wall -Wextra -Wpedantic`

## Controlled Geometry

Two calibrated cameras observe six world points at depths from 4.0 to 6.5
meters. Their exact projections generate 12 measurements. The initial second
camera pose has translation and rotation errors, and every map point is
perturbed. The first camera is fixed as the world-coordinate anchor.

The optimizer uses angle-axis camera rotations, world-coordinate point blocks,
automatic differentiation, metric RGB-D depth residuals, a 2-pixel Huber loss,
and the dense Schur solver.

## Result

```text
Initial reprojection RMSE: 11.797 pixels
Final reprojection RMSE:   8.54062e-06 pixels
Reduction factor:          > 1.3 million
Keyframes optimized:       2 (one fixed anchor)
Map points optimized:      6
Observations used:         12
Depth observations used:   12
Metric translation error:  1.07056e-07 m
CTest:                     10/10 passed
```

The reported RMSE is computed directly from unscaled pixel residuals rather
than Ceres' robustified objective. This test establishes numerical correctness
on synthetic geometry; it does not establish accuracy or runtime on a natural
RGB-D benchmark sequence.
