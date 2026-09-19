# Development Progress

## Stage 1: Pinhole Camera Model

### Goal

Implement and verify the coordinate transformations between image pixels and
3D points in the camera coordinate system.

The pixel-to-camera operation lifts a depth pixel into metric 3D camera
coordinates. The camera-to-pixel operation projects that 3D point back onto
the image plane. A near-zero round-trip error verifies that the two operations
are numerically consistent for the test input.

### Camera Parameters

- fx = 517.3 pixels
- fy = 516.5 pixels
- cx = 318.6 pixels
- cy = 255.3 pixels

### Test Input

- Pixel: (320, 240)
- Depth: 2.0 meters

### Reproduction Commands

```bash
cmake -S . -B build
cmake --build build
./build/run_slam
ctest --test-dir build --output-on-failure
```

### Test Result

```text
Original pixel: 320 240
3D point in camera coordinates: 0.00541272 -0.0592449 2
Projected pixel: 320 240
Round-trip error: 0
Camera model test passed.
```

CTest result: 1/1 test passed.
