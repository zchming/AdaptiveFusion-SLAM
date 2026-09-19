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

## Stage 2: TUM RGB-D Loading and Timestamp Synchronization

### Goal

Read the `rgb.txt` and `depth.txt` indexes supplied with a TUM RGB-D sequence,
associate color and depth images captured at nearly the same time, and load a
synchronized frame without losing the original depth precision.

### Physical Meaning

The RGB camera and depth sensor publish measurements with independent
timestamps. Combining measurements taken too far apart can assign current
color observations to geometry from another camera pose. The loader therefore
accepts a pair only when its absolute timestamp difference is no greater than
the configured synchronization threshold.

For all candidate pairs inside the threshold, the smallest timestamp
differences are selected first. Each RGB image and each depth image can appear
in at most one synchronized pair.

### Key Variables

- `rgb_timestamp`: RGB image capture time in seconds.
- `depth_timestamp`: depth image capture time in seconds.
- `time_difference`: absolute difference between the two timestamps.
- `max_time_difference`: maximum accepted difference; the default is 0.02 s.
- `depth_image`: depth image loaded unchanged so that 16-bit measurements are
  preserved.

### Reproduction Commands

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/inspect_tum_dataset tests/data/tum_sample
```

### Test Result

The repository test fixture contains four RGB timestamps and four depth
timestamps. Three pairs lie inside the 0.02 s threshold; the deliberately
unmatched records are rejected.

```text
Synchronized RGB-D pairs: 3
First RGB timestamp: 1.000000
First depth timestamp: 0.999000
Time difference: 0.001000 s
Image size: 2 x 2
RGB channels: 3
Depth bit depth: 16 bits
```

CTest result: 2/2 tests passed, including the existing camera-model test.

This stage verifies parsing, synchronization, one-to-one association, RGB image
loading, and preservation of 16-bit depth data using a controlled fixture. A
full public TUM sequence has not yet been evaluated.
