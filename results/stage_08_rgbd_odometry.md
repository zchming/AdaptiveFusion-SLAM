# Stage 8 Experiment: Continuous RGB-D Odometry

## Environment

- Ubuntu 24.04
- GCC 13.3.0
- CMake 3.28.3
- OpenCV 4.6.0
- Eigen3
- Release build with `-Wall -Wextra -Wpedantic`

## Controlled Sequence

The sequence contains a deterministic textured 640 x 480 frame with constant
2-meter depth, a second frame shifted by `(5, 3)` pixels, and a uniform third
frame that cannot be tracked. TUM Freiburg 1 intrinsics are used.

An additional run raises the LK minimum-eigenvalue threshold so that LK returns
no usable tracks, forcing the ORB fallback path.

## Result

```text
Initialization pose:          identity
LK/PnP inliers:              996
Accumulated translation:     (-0.0193323, -0.0116166, -9.55013e-08) m
ORB fallback inliers:        481
Uniform-frame status:        lost
Trusted reference preserved: yes
Valid TUM trajectory poses:  2
CTest:                       8/8 passed
```

The expected translation for a 2-meter fronto-parallel plane and `(5, 3)` pixel
shift is approximately `(-0.0193311, -0.0116167, 0)` meters. The estimated
translation differs by less than 1 millimeter. The lost frame was omitted from
the trajectory and did not replace the trusted reference.

This controlled test validates state transitions, LK-to-ORB fallback, relative
pose inversion, world-pose accumulation, failure isolation, and TUM serialization.
It does not establish drift or robustness on a natural RGB-D sequence.
