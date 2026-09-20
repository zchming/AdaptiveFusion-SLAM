# Stage 7 Experiment: PnP/RANSAC Pose Estimation

## Environment

- Ubuntu 24.04
- GCC 13.3.0
- CMake 3.28.3
- OpenCV 4.6.0
- Eigen3
- Release build with `-Wall -Wextra -Wpedantic`

## Controlled Pose and Outliers

Sixty non-coplanar 3D points were transformed by a known rotation composed from
`0.08` radians about Y and `-0.04` radians about X, with translation:

```text
t_current_from_previous = (0.10, -0.03, 0.05) meters
```

All points were projected with the TUM Freiburg 1 intrinsics. Ten of the 60
image observations were displaced by `(120, -80)` pixels to form known outliers.
The estimator used a 3-pixel RANSAC threshold, 100 iterations, and 99%
confidence, followed by iterative inlier-only PnP refinement.

## Result

```text
Input correspondences:       60
True inliers:                50
Estimated inliers:           50
Estimated inlier ratio:      0.833333
Rotation error:              1.47473e-12 radians
Translation error:           4.14648e-12 meters
Mean reprojection error:     5.25902e-11 pixels
CTest:                       7/7 passed
```

The experiment verifies the transform convention, OpenCV-to-Eigen conversion,
RANSAC outlier rejection, inlier-only refinement, and reprojection metrics for
exact synthetic geometry. It does not measure pose accuracy with real depth
noise, rolling shutter, calibration error, dynamic objects, or natural tracking
errors.
