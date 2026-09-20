# Stage 4 Experiment: ORB Descriptor Matching

## Environment

- Ubuntu 24.04
- GCC 13.3.0
- CMake 3.28.3
- OpenCV 4.6.0
- Release build with `-Wall -Wextra -Wpedantic`

## Controlled Motion

A fixed-seed random texture was blurred to create stable local intensity
structure. The second 640 x 480 image was generated from the first with a known
translation:

```text
horizontal displacement: 6 pixels
vertical displacement:   4 pixels
```

Both images used the same ORB extractor with at most 800 features. Matching used
Hamming distance, a 0.75 nearest-neighbor ratio threshold, a maximum distance
of 64 bits, and forward/backward mutual consistency.

## Result

```text
Accepted matches:               357
Translation-consistent matches: 356
Consistent fraction:            99.72%
Mean Hamming distance:          18.1961
CTest:                          4/4 passed
```

A match counted as translation-consistent when its displacement error relative
to `(6, 4)` pixels was below 2.5 pixels. Empty descriptor input returned no
matches. The experiment verifies descriptor indexing and the implemented
filters under a controlled translation; it does not yet measure matching
quality under natural camera rotation, scale change, blur, illumination change,
dynamic objects, or repetitive real-world texture.
