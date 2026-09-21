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

## Stage 5: Pyramidal LK Optical-Flow Tracking

### Goal

Track first-frame ORB keypoints into the next RGB frame using local image
intensity changes, then reject tracks that fail status, image-boundary, or
forward-backward consistency checks.

### Physical Meaning

For a small time interval, the brightness pattern around a physical scene point
is assumed to remain similar while its image position moves. Lucas-Kanade optical
flow estimates the displacement that best aligns a local window in the two
frames. An image pyramid permits larger motion to be estimated first at coarse
resolution and refined at finer resolutions.

Every forward track is run backward from the current frame into the previous
frame. If the returned point does not land close to the original point, the
track is considered unstable and rejected.

### Key Variables

- `window_size`: local square tracking window; default 21 pixels.
- `max_pyramid_level`: highest additional pyramid level; default 3.
- `termination_count`: maximum refinement iterations; default 30.
- `termination_epsilon`: minimum update size before convergence; default 0.01.
- `min_eigenvalue_threshold`: rejects weak local image structure; default
  `1e-4`.
- `max_forward_backward_error`: maximum return-to-origin error; default 1 pixel.
- `source_index`: index of the original first-frame ORB keypoint.
- `previous_point` and `current_point`: the tracked 2D pixel positions.
- `forward_backward_error`: Euclidean return-to-origin error in pixels.

For original point `p`, forward result `q`, and backward result `p_back`, the
quality signal is:

```text
e_fb = ||p_back - p||_2
```

The default tracker accepts the result only when `e_fb <= 1.0` pixel and both
forward and backward positions remain inside their images.

### Reproduction Commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/test_lk_optical_flow_tracker
./build/run_feature_frontend tests/data/tum_sample 0
```

### Controlled Test Result

A fixed-seed textured 640 x 480 image was translated by 5 pixels horizontally
and 3 pixels vertically. ORB supplied 800 starting keypoints.

```text
LK optical-flow test passed with 800 tracks, 800 translation-consistent tracks,
and mean forward-backward error 0.00047618 pixels.
```

All accepted tracks agreed with the known translation to within 1 pixel. Empty
keypoint input returned no tracks. CTest reported 5/5 passing tests with
warning-enabled Release compilation.

The repository's 2 x 2 fixture again produced no tracks because it contains no
valid ORB starting features. Natural-sequence tests must still measure retention
under rotation, depth variation, occlusion, blur, illumination change, and
nonrigid motion.

### Implemented Pipeline After Stage 5

1. Synchronize and load two consecutive RGB-D frames.
2. Extract first-frame and second-frame ORB features.
3. Use ORB descriptors to produce globally checked feature matches.
4. Use first-frame keypoints as LK starting positions.
5. Track forward through a four-scale pyramid and backward to the first frame.
6. Reject failed, out-of-bounds, and high forward-backward-error tracks.
7. Report both ORB matches and LK tracks as complementary 2D-to-2D evidence.

LK will serve as the efficient normal tracking path. ORB matching will support
verification, larger motion, and redetection when predicted failure risk rises.
The next geometric bridge is depth validation and metric 3D correspondence
construction before PnP pose estimation.

## Stage 6: Metric RGB-D 3D-to-2D Correspondences

### Goal

Connect tracked first-frame pixels to their depth measurements, convert raw TUM
depth values into meters, back-project them through the camera model, and pair
the resulting 3D points with their tracked second-frame pixels.

### Physical Meaning

LK optical flow provides only image motion from pixel `p1` to pixel `p2`. The
first frame's aligned depth image supplies the distance along the camera Z axis
at `p1`. Combining this distance with the camera intrinsics turns `p1` into a
metric 3D point. The pair `(3D point in frame 1, 2D pixel in frame 2)` is the
observation required by PnP to estimate camera motion.

### Key Variables and Validation

- `depth_scale`: raw TUM depth units per meter; default 5000.
- `min_depth_meters`: nearest accepted depth; default 0.1 m.
- `max_depth_meters`: farthest accepted depth; default 8.0 m.
- `previous_pixel`: first-frame feature location used for depth lookup.
- `current_pixel`: second-frame tracked observation used by PnP.
- `point_previous_camera`: metric 3D point in the first camera frame.
- `depth_meters`: converted metric Z value.

Depth is sampled at the nearest integer pixel to the subpixel feature location:

```text
column = round(u)
row    = round(v)
Z      = raw_depth / depth_scale
```

The builder conservatively rejects non-finite pixels, image-boundary failures,
raw depth zero, current-frame pixels outside the image, and metric depths outside
`[0.1, 8.0]` meters. It does not fill missing depth from neighboring pixels,
avoiding accidental depth transfer across object boundaries.

For valid depth, the existing camera model computes:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

The subpixel feature coordinate is used in this back-projection even though the
depth sample comes from the nearest integer pixel.

### Reproduction Commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/test_rgbd_correspondence_builder
./build/run_feature_frontend tests/data/tum_sample 0
```

### Controlled Test Result

Seven candidates exercise valid depth, zero depth, a previous pixel outside the
image, a current pixel outside the image, a 9-meter out-of-range depth, and a
NaN coordinate. Two candidates are valid.

```text
RGB-D correspondence test passed with 2 valid correspondences from 7 candidates.
First depth: 2 meters.
First 3D point: 0.00618601 -0.0604066 2
```

The first raw value is 10000, so division by 5000 produces exactly 2 meters.
The computed 3D point agrees with the expected pinhole equations. CTest reported
6/6 passing tests with warning-enabled Release compilation.

### Implemented Pipeline After Stage 6

1. Synchronize and load two consecutive RGB-D frames.
2. Extract ORB features and obtain ORB matches plus LK tracks.
3. Convert accepted LK tracks into generic pixel correspondences.
4. Sample the first frame's 16-bit depth at each previous pixel.
5. Validate and convert raw depth into meters.
6. Back-project the first-frame pixel with `Camera::pixelToCamera()`.
7. Preserve the tracked current-frame pixel as the 2D observation.
8. Return metric `RgbdCorrespondence` records ready for PnP.

The demo currently uses TUM Freiburg 1 RGB intrinsics and the standard TUM depth
scale. The builder itself is configurable; calibration-file loading will be
needed before running sequences with different camera intrinsics.

## Stage 7: PnP/RANSAC Relative Pose Estimation

### Goal

Estimate the rigid transformation from the previous camera coordinate frame to
the current camera coordinate frame from metric 3D-to-2D correspondences, reject
outliers with RANSAC, refine the pose on the inliers, and expose geometric
quality metrics.

### Transform Convention

The returned rotation and translation satisfy:

```text
P_current = R_current_from_previous * P_previous
            + t_current_from_previous
```

This transform changes the coordinates of a static scene point from the
previous camera frame into the current camera frame. It is not yet a global
camera trajectory pose; trajectory accumulation will require composing or
inverting relative transforms with a clearly defined world-frame convention.

### RANSAC and Refinement

The estimator uses EPnP inside RANSAC. Each hypothesis projects transformed 3D
points into the current image and treats observations within 3 pixels as
inliers. Defaults are 100 iterations, 99% confidence, at least six input
correspondences, and at least six final inliers.

After RANSAC, iterative PnP refines rotation and translation using only the
accepted inliers. If refinement fails, the original RANSAC pose is retained.
Invalid, non-finite, or behind-camera 3D inputs fail cleanly.

### Health Metrics

The estimator reports:

```text
inlier_ratio = number_of_inliers / number_of_correspondences
```

and the mean Euclidean reprojection residual:

```text
e_reproj = mean(||project(R * P + t) - observed_pixel||_2)
```

These provide the planned geometric health variables `r_inlier` and
`e_reproj` for future tracking-failure forecasting.

### Reproduction Commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/test_pnp_pose_estimator
./build/run_feature_frontend tests/data/tum_sample 0
```

### Controlled Test Result

The test generates 60 non-coplanar metric 3D points, transforms them with a
known rotation and translation, projects them into the current image, and then
corrupts 10 observations with large pixel offsets.

```text
PnP pose test passed with 50/60 inliers,
rotation error 1.47473e-12 radians,
translation error 4.14648e-12 meters,
and mean reprojection error 5.25902e-11 pixels.
```

RANSAC rejected all 10 artificial outliers and retained all 50 true inliers.
Insufficient input returns an unsuccessful estimate without throwing. CTest
reported 7/7 passing tests with warning-enabled Release compilation.

### Implemented Pipeline After Stage 7

1. Synchronize and load two RGB-D frames.
2. Extract ORB features and compute ORB matches plus LK tracks.
3. Validate first-frame depth and build metric 3D-to-2D correspondences.
4. Estimate a relative pose with EPnP inside RANSAC.
5. Refine the pose using only RANSAC inliers.
6. Return rotation, translation, inlier indices, inlier ratio, and mean
   reprojection error.

The minimal two-frame RGB-D visual odometry chain is now present. It still needs
validation on a natural sequence, calibration-file loading, persistent frame
state, trajectory accumulation, and map representations before it constitutes
a basic SLAM system.

## Stage 8: Continuous RGB-D Odometry and Trajectory State

### Goal

Turn the two-frame frontend into a stateful sequence processor that stores
frame data and poses, tracks from the latest trusted frame, falls back from LK
to ORB matching when necessary, accumulates world-frame poses, protects the
trusted state after failure, and writes a TUM-format trajectory.

### Frame State

Each `Frame` now stores its sequential id, RGB/depth association, loaded images,
ORB features, `T_world_from_camera`, and a pose-valid flag. The first accepted
frame defines the world coordinate system and receives the identity pose.

The PnP result maps previous-camera coordinates into current-camera coordinates:

```text
P_current = T_current_from_previous * P_previous
```

The stored trajectory pose maps current-camera coordinates into the world. It
is accumulated using:

```text
T_world_from_current = T_world_from_previous
                       * inverse(T_current_from_previous)
```

The inverse is essential because PnP and the trajectory use opposite transform
directions.

### Tracking State Machine

- First valid input: status `Initialized`, identity world pose.
- Normal path: track trusted-frame keypoints with LK and estimate PnP.
- Fallback path: if LK/PnP fails, match trusted and current ORB descriptors and
  retry RGB-D correspondence construction plus PnP.
- Success: status `Tracked`; accumulate pose and replace the trusted reference.
- Failure: status `Lost`; keep the current pose invalid, omit it from the
  trajectory, and retain the previous trusted reference frame.

This is the first implemented map-protection behavior: failed observations are
prevented from changing the persistent tracking reference or trajectory.

### TUM Trajectory Format

Each valid pose is written as:

```text
timestamp tx ty tz qx qy qz qw
```

The translation and quaternion represent `T_world_from_camera`. Timestamps must
be strictly increasing, invalid poses are rejected, and quaternions are
normalized before output.

### Reproduction Commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/test_rgbd_odometry
./build/run_rgbd_odometry tests/data/tum_sample /tmp/trajectory.txt 3
```

### Controlled Test Result

A deterministic 640 x 480 textured RGB-D frame with constant 2-meter depth was
followed by a frame translated 5 pixels horizontally and 3 pixels vertically.
A third uniform frame deliberately removed all usable visual structure.

```text
Frame 0: initialized at identity
Frame 1: tracked with 996 LK/PnP inliers
Accumulated translation: (-0.0193323, -0.0116166, -9.55013e-08) meters
Forced LK failure: ORB fallback recovered pose with 481 inliers
Frame 2: lost and excluded from trajectory
Valid TUM poses: 2
CTest: 8/8 passed
```

For a fronto-parallel plane at 2 meters, the expected world-frame translation
from `(5, 3)` pixels is approximately `(-5*2/fx, -3*2/fy, 0)`, matching the
estimated pose within 1 millimeter. The test also confirms that a lost frame
does not replace the trusted reference and that both timestamps appear in TUM
output.

The repository's tiny 2 x 2 fixture processed three frames, wrote only the
initial identity pose, and marked the remaining frames lost as expected because
ORB cannot form a 31-pixel patch on that image.

### Implemented Pipeline After Stage 8

1. Load synchronized RGB-D frames sequentially.
2. Create a persistent `Frame` with features and pose state.
3. Track from the latest trusted reference with LK.
4. Build metric correspondences and estimate relative pose with PnP/RANSAC.
5. If that fails, retry with globally matched ORB features.
6. On success, invert and compose the relative transform into the world pose.
7. On failure, preserve the trusted reference and omit the frame from output.
8. Write every valid world pose in TUM trajectory format.

The system is now a minimal continuous RGB-D visual odometry implementation.
It still lacks native calibration loading, real-sequence validation, keyframes,
map points, local optimization, and the temporal risk predictor required for the
final AdaptiveFusion-SLAM system.

## Stage 9: Keyframes and Sparse Map Points

### Goal

Introduce persistent keyframes, world-coordinate map points, feature-to-point
associations, observation records, and a conservative keyframe selection policy
without allowing failed tracking frames to modify the map.

### Keyframe State

A keyframe copies the trusted source frame's id, RGB timestamp, world pose,
RGB-D images, ORB keypoints, and descriptors. It also owns one optional map-point
id for every feature index. Construction rejects invalid or non-finite poses and
inconsistent keypoint/descriptor counts.

### Map-Point State

Each map point stores:

- a stable numeric id;
- a finite 3D position in the world coordinate system;
- one representative 32-byte ORB descriptor;
- observations identified by `(keyframe_id, feature_index)`.

For a valid-depth keyframe feature, the camera-frame point is transformed by:

```text
P_world = T_world_from_camera * P_camera
```

The originating keyframe feature stores the new map-point id, while the map
point stores the reverse observation. This bidirectional relation is required
for later reprojection, local optimization, and observation counting.

### Keyframe Selection

The default policy requires at least five frames since the previous keyframe.
After that spacing, a frame is selected when translation reaches 0.15 meters or
rotation reaches 0.15 radians. A keyframe is forced after 20 frames even when
motion is smaller. The first valid frame is always selected; a frame with an
invalid pose is never selected.

Selection thresholds limit redundant map growth. They are baseline engineering
values rather than experimentally optimized parameters and will later be
modified by predicted tracking risk.

### Transactional Map Insertion

Keyframe construction, depth validation, map-point creation, and associations
are completed in temporary objects first. The new objects are committed to the
sparse map only after the entire insertion succeeds, preventing partially
written map state after an exception.

### Reproduction Commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/test_sparse_map
./build/run_rgbd_odometry tests/data/tum_sample /tmp/trajectory.txt 3
```

### Controlled Test Result

A trusted frame contains three ORB features. Two have valid depths of 2 and 1
meters, while the third has zero depth. The frame pose translates camera points
by `(1, 2, 0)` meters into the world.

```text
Sparse-map test passed with 1 keyframe and 2 metric map points;
invalid and redundant frame insertion was rejected.
CTest: 9/9 passed
```

The test verifies camera-to-world conversion, descriptor retention, the
keyframe-feature to map-point link, the reverse observation, motion-based
selection, and rejection of invalid poses. The repository's 2 x 2 fixture
creates one initial keyframe but zero points because it contains no ORB features.

### Implemented Pipeline After Stage 9

1. Process the RGB-D sequence and estimate trusted world poses.
2. Evaluate frame gap, translation, and rotation against the latest keyframe.
3. Reject lost frames and nearby redundant frames.
4. Copy an accepted frame into persistent keyframe state.
5. Validate feature depths and back-project camera-frame points.
6. Transform valid points into world coordinates.
7. Store one descriptor and initial observation per new map point.
8. Link every originating keyframe feature to its map-point id.

This is a sparse-map foundation, not yet a complete map-management system.
Points observed in different keyframes are not yet fused into shared landmarks,
and keyframe poses or point positions are not yet jointly optimized. Those are
the next requirements for local bundle adjustment and risk-adaptive map updates.

## Stage 10: Cross-Keyframe Map-Point Association

### Goal and Physical Meaning

A physical scene point should keep one stable map identity when several
keyframes observe it. Appearance proposes a correspondence, then geometry
checks whether the existing 3D point predicts the new 2D observation and depth.

### Implementation

- ORB matching connects the latest and incoming keyframe descriptors.
- `predicted_camera_point = T_camera_from_world * P_world` expresses an old
  landmark in the incoming camera coordinate system.
- Projection provides `predicted_pixel`; its distance from the keypoint must be
  at most 3 pixels.
- Valid incoming depth must agree within `max(0.15 m, 0.10 * predicted_depth)`.
- Accepted observations reuse the existing id; unmatched valid-depth features
  create new points.
- A map point may have at most one observation from each keyframe.
- Copied state is swapped into the live map only after the insertion succeeds.

### Verification

Two landmarks are observed again and one new valid-depth feature is introduced.
The result is two keyframes and three points instead of five duplicate points.
A separate descriptor-identical keypoint shifted by 50 pixels is rejected by
the reprojection gate and becomes a distinct landmark.

```text
Existing points reobserved:          2
New points created:                  1
Total points after two keyframes:    3
Descriptor-only false reuse blocked: yes
Release CTest with warnings:         9/9 passed
```

The next stage is local bundle adjustment over shared observations. Association
currently searches only the latest keyframe; it does not update landmark
descriptors or positions, cull outliers, or search a covisibility neighborhood.

## Stage 11: Local Bundle Adjustment

### Goal and Physical Meaning

Tracking estimates each camera motion locally, so small errors accumulate.
Bundle adjustment uses the fact that one physical point is seen by several
keyframes: camera poses and point positions are changed together until their
predicted image locations agree with all measured keypoints.

For observation `i,j`, the residual is:

```text
e_ij = measured_pixel_ij
       - project(T_camera_i_from_world * point_world_j)
```

`pose_camera_from_world` contains a three-value angle-axis rotation followed by
three translation values. `point_world` contains the map point's `x, y, z`.
Each residual contributes horizontal and vertical pixel error.

### Implementation

- Ceres Solver 2.2 supplies automatic differentiation and nonlinear solving.
- The default local window contains the latest five keyframes.
- Only landmarks observed by at least two window keyframes are optimized.
- The oldest window pose is fixed, removing the unconstrained global gauge.
- `DENSE_SCHUR` exploits the camera/landmark block structure.
- Valid 16-bit RGB-D values add weighted metric-depth residuals, preventing the
  camera translation and scene depth from changing by a common scale.
- A Huber loss with 2-pixel transition limits outlier influence.
- The solver runs for at most 20 iterations by default.
- Results are written into copied keyframes and points, then swapped into the
  sparse map only when Ceres reports a usable solution.
- The sequence runner starts local BA after a new keyframe reobserves existing
  landmarks and reports optimization count and final pixel RMSE.

### Controlled Result

Two cameras observe six 3D points, producing 12 observations. The second camera
pose and all points begin with deterministic perturbations.

```text
Initial true reprojection RMSE: 11.797 pixels
Final true reprojection RMSE:   8.54062e-06 pixels
Keyframes in local window:      2
Shared points optimized:        6
Observations used:              12
Depth observations used:        12
Metric translation error:       1.07056e-07 m
Anchor pose unchanged:          yes
Release CTest:                  10/10 passed
```

This validates the local optimizer on controlled geometry. Natural-sequence
accuracy, runtime, outlier removal, covisibility selection,
and trajectory/map state synchronization still require later stages.

## Stage 12: Per-Frame Geometric Health Monitoring

### Goal and Physical Meaning

Tracking normally fails after several geometric signals have already degraded.
This stage records those signals for every input frame so a later temporal model
can learn deterioration trends before the final lost state.

- `inlier_ratio`: fraction of metric correspondences accepted by PnP/RANSAC.
- `mean_reprojection_error_pixels`: agreement between the estimated pose and
  accepted image measurements.
- `mean_forward_backward_error_pixels`: whether LK tracks return consistently
  to their starting pixels.
- `spatial_coverage`: occupied cells in a 4-by-3 image grid; clustered features
  give weaker pose geometry than broadly distributed features.
- `median_parallax_pixels`: robust typical feature displacement and therefore
  the amount of inter-frame viewpoint information.
- `valid_depth_ratio`: fraction of 2D correspondences with usable metric depth.
- `tracking_success` and `used_orb_fallback`: final outcome and whether the
  normal LK path had already failed.

Raw feature, correspondence, and inlier counts are retained alongside ratios.
The initialization frame is successful but explicitly has no tracking
measurement. Failed frames remain in the health stream.

### Runtime Integration

`RgbdOdometry::process` computes health from the exact correspondences used for
the final PnP attempt, plus the LK forward-backward evidence collected before a
possible ORB fallback. `run_rgbd_odometry` writes one CSV row for every input to
`<trajectory>.health.csv`.

### Controlled Result

```text
PnP inlier ratio:             0.75
Mean reprojection error:      0.8 pixels
Mean forward-backward error:  0.3 pixels
Spatial coverage:             0.333333
Median parallax:              5 pixels
Valid-depth ratio:            0.5
Tracked/lost integration:     passed
CSV round trip:               passed
Release CTest:                11/11 passed
```

The repository's tiny fixture also writes all three rows: one initialization
and two lost frames. Its images contain no features, so the geometric fields are
zero while the failure flags remain available for label generation.

The next stage will maintain a fixed-length temporal health window and generate
short-horizon future-failure labels. A geometric conditioning measure and
benchmark-derived normalization statistics are not implemented yet.

## Stage 13: Temporal Health Windows and Failure Labels

### Goal and Label Definition

A predictor must use only information available at the current time while its
training target comes from the future. For history length `L` and horizon `H`,
the sample anchored at frame `t` is:

```text
input  = [health_(t-L+1), ..., health_t]
label  = 1 if any tracking failure occurs in [t+1, ..., t+H]
```

The default is `L=5`, `H=3`. Every history frame must have a real tracking
measurement and successful pose, so a sample cannot use an already-lost state
to predict that tracking is failing. Initialization and windows containing a
past failure are skipped.

### Feature Layout

Each time step contains seven ordered values:

```text
[inlier ratio, reprojection error, forward-backward error,
 spatial coverage, median parallax, valid-depth ratio, ORB fallback]
```

`frames_until_failure` records the first lost-frame offset from 1 through `H`.
For negative samples, `H+1` is the explicit no-failure sentinel. CSV columns
are flattened as `h0_*` through `h(L-1)_*` for later model training.

### Real and Simulated Integration

After sequence processing, `run_rgbd_odometry` now generates:

```text
trajectory.txt
trajectory.txt.health.csv
trajectory.txt.failure_dataset.csv
```

The standalone simulator creates 24 health frames. Geometry is stable through
frame 8, degrades progressively from frames 9 to 14, fails at frame 15, and
then recovers. The current frame remains successful at every positive anchor.

```text
Usable temporal samples:        11
Negative samples:                8
Positive samples:                3
Frame 12 warning lead time:      3
Frame 13 warning lead time:      2
Frame 14 warning lead time:      1
Release CTest:                   12/12 passed
```

At frame 12, inlier ratio has fallen to `0.578571` and reprojection error has
risen to `1.84286` pixels, but ORB fallback has not yet occurred. At frames 13
and 14, fallback is active while the remaining lead time falls to two and one
frames. This demonstrates the intended distinction between a current failure
detector and a future-failure label.

The next stage will normalize these temporal features and implement a first
probabilistic risk baseline. Real predictive claims require multiple natural
sequences, sequence-level train/test separation, class balancing, and AUROC,
AUPRC, calibration, and warning-lead-time evaluation.

## Stage 14: Interpretable Temporal Risk Prediction Baseline

### Model Input

For each of the seven health signals, the encoder calculates:

- `latest`: the value available at the current frame;
- `mean`: average condition across the complete history window;
- `slope`: least-squares change per frame across the window.

This creates 21 variables from a five-frame window. The slope distinguishes a
stable mediocre observation from a rapidly deteriorating sequence.

### Training

The baseline is binary logistic regression:

```text
risk = sigmoid(bias + weights * normalized_temporal_features)
```

Normalization mean and standard deviation are computed from training samples
only. Batch gradient descent uses balanced positive/negative class weights and
L2 regularization. Both classes are required, feature dimensions are checked,
and numerically stable sigmoid branches prevent overflow.

### Metrics

- Accuracy measures decisions at risk threshold 0.5.
- AUROC measures ranking across all possible thresholds.
- AUPRC emphasizes the less frequent future-failure class.
- Brier score measures probability error and is lower when probabilities are
  both discriminative and well calibrated.
- Mean warning lead counts frames from a true positive warning to first failure.

### Held-Out Simulation

Training and test windows come from separate synthetic episodes. Failure time,
degradation duration, and small baseline variations change between episodes.
No window from a test episode participates in normalization or fitting.

```text
Training samples:          202
Held-out test samples:     119
Positive test samples:      15
True positives:             15
False negatives:             0
False positives:             5
True negatives:             99
Accuracy:              0.957983
AUROC:                 0.997436
AUPRC:                 0.983824
Brier score:           0.0366757
Mean warning lead:             2 frames
Release CTest:             13/13 passed
```

The five false positives occur in late but still negative degradation windows,
showing the threshold tradeoff that later calibration must address. Synthetic
metrics validate implementation and experimental plumbing only. The next stage
will connect live risk values to explicit low/medium/high/critical policies and
measure whether interventions reduce failure or map contamination.

## Stage 15: Risk-Adaptive Frontend and Map Protection

### Probability-to-Action Policy

```text
[0.00, 0.30): low
[0.30, 0.60): medium
[0.60, 0.85): high
[0.85, 1.00]: critical
```

- **Low:** normal LK path, normal keyframes, observations and new landmarks.
- **Medium:** run ORB verification, request earlier keyframes, continue mapping.
- **High:** force ORB redetection, request early keyframes, allow observations
  of existing trusted landmarks, suppress all new landmark creation.
- **Critical:** prevent keyframe insertion and every map write, retain the last
  trusted odometry reference, and prepare for recovery.

Probabilities outside `[0,1]`, non-finite values, and unordered thresholds are
rejected. Policy output is a plain decision structure so later experiments can
change thresholds without coupling the predictor to SLAM state classes.

### Integration

`RgbdOdometry::process` accepts an optional risk decision. Medium risk evaluates
ORB even after LK succeeds and retains the geometrically stronger PnP result.
High risk bypasses LK pose estimation and forces ORB redetection. Critical risk
may return a temporary pose but cannot replace the trusted reference.

`KeyframePolicy` halves normal minimum gap and motion thresholds at medium risk;
high risk forces an early keyframe after the shortened gap. Critical risk blocks
insertion. `SparseMap` independently controls existing observations and new
landmark creation, and reports suppressed valid-depth candidates.

### Controlled Map-Protection Comparison

Twelve frames each present 100 candidate points. Risk moves from low through
medium, high, and critical. The experiment defines candidates during high and
critical periods as unreliable to isolate policy behavior.

```text
Baseline total points:             1200
Baseline unreliable points:         500
Adaptive total points:               700
Adaptive unreliable points:            0
Suppressed risky candidates:         500
Early-keyframe requests:                4
Critical frozen frames:                 3
Release CTest:                       14/14 passed
```

The test suite additionally proves that a high-risk real `SparseMap` insertion
can add observations to two trusted points while suppressing one new landmark,
and that critical policy prevents the trusted reference from being replaced.

The next stage must serialize the learned risk model, load it in the RGB-D
runner, maintain the live health window, and compare baseline versus adaptive
trajectories and maps on complete sequences. Threshold hysteresis and recovery
logic are also still required to prevent rapid policy oscillation.
