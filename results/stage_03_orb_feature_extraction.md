# Stage 3 Experiment: ORB Feature Extraction

## Environment

- Ubuntu 24.04
- GCC 13.3.0
- CMake 3.28.3
- OpenCV 4.6.0
- Release build

## Controlled Inputs

1. A deterministic 640 x 480 black-and-white checkerboard tests whether the
   extractor finds corners and creates valid descriptors.
2. A uniform 640 x 480 gray image tests behavior when no visual structure is
   present.
3. The 2 x 2 RGB-D fixture from Stage 2 exercises the connected dataset-to-ORB
   application path.

The extractor was configured for at most 500 features in the checkerboard
test. Other settings used their defaults: scale factor 1.2, eight pyramid
levels, and FAST threshold 20.

## Results

```text
ORB feature test passed with 304 checkerboard keypoints and 32-byte descriptors.
```

Every checkerboard keypoint had exactly one `CV_8UC1` descriptor row containing
32 bytes. The feature count remained below the configured maximum. The uniform
image produced no keypoints or descriptors.

The connected frontend produced:

```text
Frame index: 0
RGB timestamp: 1.000000
Depth timestamp: 0.999000
Keypoints: 0
Descriptor rows: 0
Descriptor columns: 0
```

Zero features are expected for the 2 x 2 fixture because ORB uses a 31-pixel
local patch. CTest reported 3/3 passing tests. Evaluation on natural TUM RGB-D
images is still required before drawing conclusions about feature count,
coverage, repeatability, or runtime.
