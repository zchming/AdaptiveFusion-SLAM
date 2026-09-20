# Stage 5 Experiment: Pyramidal LK Optical Flow

## Environment

- Ubuntu 24.04
- GCC 13.3.0
- CMake 3.28.3
- OpenCV 4.6.0
- Release build with `-Wall -Wextra -Wpedantic`

## Controlled Motion

A fixed-seed random texture was blurred and translated with the known motion:

```text
horizontal displacement: 5 pixels
vertical displacement:   3 pixels
```

ORB supplied 800 first-frame keypoints. LK used a 21 x 21 window, pyramid levels
0 through 3, at most 30 iterations per refinement, convergence epsilon 0.01,
minimum eigenvalue threshold `1e-4`, and maximum forward-backward error 1 pixel.

## Result

```text
Starting keypoints:              800
Accepted LK tracks:             800
Translation-consistent tracks:  800
Consistent fraction:            100.00%
Mean forward-backward error:    0.00047618 pixels
CTest:                          5/5 passed
```

A track counted as translation-consistent when its displacement error relative
to `(5, 3)` pixels was below 1 pixel. The experiment validates index retention,
forward tracking, backward verification, and filtering under a controlled
translation. It does not yet establish performance on natural RGB-D sequences
or under deliberate visual degradation.
