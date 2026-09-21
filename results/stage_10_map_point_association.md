# Stage 10 Experiment: Cross-Keyframe Map-Point Association

## Environment

- Ubuntu 24.04
- GCC 13.3.0
- CMake 3.28.3
- OpenCV 4.6.0
- Eigen3
- Release build with `-Wall -Wextra -Wpedantic`

## Controlled Input

The first keyframe contains two valid metric landmarks and one zero-depth
feature. The second observes the same two landmarks with matching ORB
descriptors, consistent reprojections and depths, then adds one unseen feature
at 1.5 meters. A second experiment shifts a descriptor-identical observation
by 50 pixels while retaining valid depth.

## Result

```text
First-keyframe map points:           2
Existing points observed again:      2
New second-keyframe points:          1
Final map-point count:               3
Observation counts of reused points: 2, 2
50-pixel inconsistent match:         rejected
Transactional failure isolation:     passed
CTest:                               9/9 passed
```

Repeated observations now share landmark identity, while descriptor similarity
alone cannot merge geometrically inconsistent features. This is a deterministic
module test; natural-sequence precision, recall, map growth, and trajectory
effects remain to be measured.
