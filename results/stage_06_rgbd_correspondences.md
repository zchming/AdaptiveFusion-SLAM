# Stage 6 Experiment: Metric RGB-D Correspondences

## Environment

- Ubuntu 24.04
- GCC 13.3.0
- CMake 3.28.3
- OpenCV 4.6.0
- Eigen3
- Release build with `-Wall -Wextra -Wpedantic`

## Controlled Input

A 640 x 480 unsigned 16-bit depth image and seven pixel correspondences were
constructed. The camera intrinsics were TUM Freiburg 1 values:

```text
fx = 517.3, fy = 516.5, cx = 318.6, cy = 255.3
depth scale = 5000 raw units per meter
valid range = 0.1 to 8.0 meters
```

The candidates included two valid depths plus zero depth, an out-of-bounds
previous pixel, an out-of-bounds current pixel, a 9-meter depth, and a NaN pixel.

## Result

```text
Input candidates:          7
Valid correspondences:     2
First raw depth:           10000
First metric depth:        2 meters
First metric 3D point:     (0.00618601, -0.0604066, 2)
CTest:                     6/6 passed
```

The result verifies conservative validity filtering, raw-to-metric conversion,
subpixel pinhole back-projection, source-index retention, and preservation of
the second-frame observation. The test does not yet validate calibration files,
real depth noise, RGB-depth registration errors, or full-sequence valid-depth
coverage.
