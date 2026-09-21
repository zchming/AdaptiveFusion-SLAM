# Stage 9 Experiment: Keyframes and Sparse Map Points

## Environment

- Ubuntu 24.04
- GCC 13.3.0
- CMake 3.28.3
- OpenCV 4.6.0
- Eigen3
- Release build with `-Wall -Wextra -Wpedantic`

## Controlled Map Input

One trusted frame contains three ORB keypoints and descriptors. Its depth image
provides 2-meter and 1-meter measurements for two features and zero depth for
the third. The frame's `T_world_from_camera` translation is `(1, 2, 0)` meters.

The keyframe policy is also evaluated with a nearby frame, a frame translated
by 0.20 meters after five frame ids, and an invalid-pose frame.

## Result

```text
Inserted keyframes:             1
Valid map points created:       2
Zero-depth features rejected:   1
World-coordinate check:         passed
Descriptor retention:           passed
Bidirectional observation link: passed
Nearby frame rejected:          yes
Invalid-pose frame rejected:    yes
CTest:                          9/9 passed
```

The experiment verifies protected, transactional creation of metric world map
state from a trusted keyframe. It does not yet test cross-keyframe landmark
association, map-point fusion, culling, local bundle adjustment, or natural
sequence map quality.
