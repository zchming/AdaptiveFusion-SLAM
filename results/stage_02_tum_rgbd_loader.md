# Stage 2 Experiment: TUM RGB-D Loader

## Environment

- Ubuntu 24.04
- GCC 13.3.0
- CMake 3.28.3
- OpenCV 4.6.0

## Controlled Input

The test fixture contains four RGB records and four depth records. Three
timestamp pairs are within the 0.02-second synchronization threshold. RGB and
depth images are 2 x 2 pixels; the depth image stores 16-bit values.

## Result

```text
Synchronized RGB-D pairs: 3
First RGB timestamp: 1.000000
First depth timestamp: 0.999000
Time difference: 0.001000 s
Image size: 2 x 2
RGB channels: 3
Depth bit depth: 16 bits
```

CTest reported 2/2 passing tests. No full TUM benchmark sequence was present in
the development environment, so throughput and full-sequence pairing coverage
remain to be measured when benchmark data is added.
