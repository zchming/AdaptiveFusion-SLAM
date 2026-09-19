# AdaptiveFusion-SLAM

**Confidence-Aware Hybrid Visual SLAM**

AdaptiveFusion-SLAM is a visual SLAM system that combines ORB feature matching
and LK optical-flow tracking through an adaptive confidence-aware frontend.

## Project Goals

- Build a complete visual SLAM pipeline from scratch.
- Estimate camera trajectories and construct a sparse 3D map.
- Dynamically select ORB matching or LK optical flow according to tracking quality.
- Evaluate accuracy and efficiency using ATE, RPE, tracking success rate, and FPS.

## Planned Pipeline

1. Image sequence loading
2. ORB feature extraction and matching
3. LK optical-flow tracking
4. Confidence-aware frontend selection
5. Camera pose estimation
6. Triangulation and map-point management
7. Keyframe selection
8. Local bundle adjustment
9. Trajectory and runtime evaluation

## Current Status

Project initialization in progress.
