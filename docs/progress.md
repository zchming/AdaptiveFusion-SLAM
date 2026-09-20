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

## Stage 3: ORB Feature Extraction

### Goal

Detect repeatable image locations and compute a compact binary description of
the local appearance around every detected location. These features will form
the visual observations used by descriptor matching, optical flow, and camera
pose estimation.

### Physical Meaning

A keypoint is a pixel location whose surrounding intensity pattern is locally
distinctive, such as a corner. An ORB descriptor samples brightness comparisons
around that point and stores their outcomes as 256 bits. Descriptors from two
frames can later be compared with Hamming distance to decide whether the two
keypoints may observe the same physical scene point.

### Key Variables

- `max_features`: requested upper limit on retained keypoints; default 1000.
- `scale_factor`: image-pyramid scale change between adjacent levels; default
  1.2.
- `pyramid_levels`: number of scales searched for features; default 8.
- `fast_threshold`: minimum FAST corner contrast; default 20.
- `keypoints`: detected 2D locations with scale, orientation, and response.
- `descriptors`: one 32-byte binary descriptor row per keypoint.

The extractor accepts 8-bit grayscale, BGR, or BGRA images. Color images are
converted to grayscale because ORB describes local brightness structure rather
than color. Empty images, unsupported channel counts, and non-8-bit input are
rejected explicitly.

### Reproduction Commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/test_orb_feature_extractor
./build/run_feature_frontend tests/data/tum_sample 0
```

### Test Result

The deterministic 640 x 480 checkerboard produced the following result:

```text
ORB feature test passed with 304 checkerboard keypoints and 32-byte descriptors.
```

A uniform gray image produced zero keypoints and an empty descriptor matrix, as
expected because it contains no brightness corners. CTest reported 3/3 passing
tests, including the camera and RGB-D dataset tests.

The complete data path was also exercised with the repository's 2 x 2 RGB-D
fixture:

```text
Frame index: 0
RGB timestamp: 1.000000
Depth timestamp: 0.999000
Keypoints: 0
Descriptor rows: 0
Descriptor columns: 0
```

The tiny fixture is intentionally sufficient for image-loading tests but too
small for ORB's default 31-pixel patch, so zero keypoints are expected. Feature
quality and spatial distribution on a real TUM sequence remain to be measured.

### Implemented Pipeline After Stage 3

`run_feature_frontend` now connects the previously independent modules in this
order:

1. `TumRgbdDataset::loadAssociations()` pairs RGB and depth timestamps.
2. `TumRgbdDataset::loadFrame()` loads one color image and its synchronized
   16-bit depth image.
3. `OrbFeatureExtractor::extract()` converts the color image to grayscale.
4. OpenCV ORB detects keypoints and computes one binary descriptor per point.
5. The application reports timestamps, feature count, and descriptor shape.

The `Camera` module remains independently verified. It will be connected after
feature matching supplies 2D correspondences and the depth image supplies the
metric depth required to lift selected pixels into 3D camera coordinates.

## Stage 4: ORB Descriptor Matching

### Goal

Establish candidate 2D-to-2D correspondences between consecutive RGB frames by
comparing their ORB descriptors, while rejecting ambiguous or one-sided
associations before geometric pose estimation.

### Physical Meaning

Each accepted match hypothesizes that one keypoint in the earlier frame and one
keypoint in the later frame observe the same physical scene location. ORB
descriptors contain 256 binary tests, so their difference is measured with
Hamming distance: the number of bit positions that disagree.

A small distance alone is insufficient in repetitive scenes. The matcher also
requires the best candidate to be clearly better than the second-best candidate
and, by default, requires the reverse search to select the original feature.

### Key Variables and Filters

- `query_index`: keypoint and descriptor row in the first frame.
- `train_index`: matched keypoint and descriptor row in the second frame.
- `distance`: Hamming distance between the two 256-bit descriptors.
- `ratio_threshold`: best-to-second-best distance ratio; default 0.75.
- `max_hamming_distance`: absolute distance limit; default 64 bits.
- `require_mutual_consistency`: enables the forward/backward best-match check.

For best distance `d1` and second-best distance `d2`, a match must satisfy:

```text
d1 < 0.75 * d2
d1 <= 64
```

With mutual consistency enabled, first-frame feature `i` may match second-frame
feature `j` only when the reverse nearest-neighbor search maps `j` back to `i`.

### Reproduction Commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/test_orb_feature_matcher
./build/run_feature_frontend tests/data/tum_sample 0
```

### Controlled Test Result

The test creates a deterministic textured 640 x 480 image and translates it by
6 pixels horizontally and 4 pixels vertically. Because the motion is known,
accepted matches can be checked against the expected displacement.

```text
ORB matcher test passed with 357 matches, 356 translation-consistent matches,
and mean Hamming distance 18.1961.
```

The translation-consistent fraction was 99.72%, above the required 90% test
threshold. Empty descriptor input correctly returned no matches. CTest reported
4/4 passing tests with warning-enabled Release compilation.

The repository's 2 x 2 RGB-D fixture exercised the full two-frame application
path and produced zero matches because neither tiny image can contain an ORB
patch. Evaluation on natural consecutive TUM images remains required.

### Implemented Pipeline After Stage 4

1. Synchronize RGB and depth timestamps.
2. Load two consecutive RGB-D frames.
3. Extract ORB keypoints and 256-bit descriptors from both RGB images.
4. Find the two nearest descriptor candidates with Hamming distance.
5. Apply the 0.75 ratio test and 64-bit absolute-distance limit.
6. Apply forward/backward mutual consistency.
7. Return accepted `FeatureMatch` records connecting keypoint indices across
   the two frames.

The accepted indices now provide 2D-to-2D visual correspondences. The next
bridge is to validate and scale the first-frame depth at each matched keypoint,
then use `Camera::pixelToCamera()` to create metric 3D points for PnP.
