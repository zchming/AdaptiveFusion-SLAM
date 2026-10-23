# V18: ORB observation utilization controls

Using all available ORB correspondences increases native PnP acceptance but does
not improve pose accuracy on the frozen 39 pairs. Keep the solver fixed and move
the next comparison toward observation quality and geometric support.

## Results

An accurate pair must meet translation <= 0.03 m and rotation <= 2 degrees under
both nearest and interpolated GT alignment. Counts include exact raw fallbacks.
New damage means an accurate raw pair becomes inaccurate under either alignment.

| Arm | Input rows across 39 pairs | Applied | Accurate pairs | Applied and accurate | Newly damaged raw pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw | — | — | 14 | — | 0 |
| V17 native, even source IDs | 631 | 21 | 13 | 6 | 4 |
| All IDs, both endpoint depths valid | 1261 | 26 | 7 | 4 | 9 |
| All IDs, source depth valid | 1519 | 26 | 6 | 3 | 9 |

V17 is reused from its sealed study, not refitted. Each new arm has 8
insufficient-support rejections and 5 native estimator rejections. All 26 rejected
fits return the exact raw transform. All 78 new arm processes completed without
timeouts or process errors. Both GT alignments independently give 7 and 6 accurate
pairs respectively for the new arms. Full errors, rejection reasons, timings and
all 156 evaluation rows are retained in the linked artifacts.

- [Per-pair results](../results/orb_observation_results.csv)
- [Scene and aggregate summary](../results/orb_observation_summary.csv)
- [Acceptance and provenance hashes](../results/orb_observation_acceptance.json)

## What the controls resolve

Removing the even/odd split recovers enough support to apply five more poses.
It also adds five newly damaged raw pairs and loses one previously recovered
inaccurate raw pair. More accepted poses therefore do not establish a useful
performance improvement.

The room 142-143 pair illustrates weak geometric support: 11 input observations
produce 9 RANSAC inliers, occupying 2 of 16 image cells, with an inlier bounding
box of about 26 x 106 px and source depths 1.126–1.284 m. The resulting maximum
error across the two GT alignments is 0.423 m / 20.69 degrees; raw error is
0.0172 m / 0.411 degrees. These observations support investigating spatial/depth
support; they do not independently prove incorrect descriptor matches.

Removing target-depth eligibility adds 258 observations (20.5%) but does not
change acceptance or rejection counts. Desk 22-24 changes from accurate to
inaccurate: maximum error moves from 0.0146 m / 1.437 degrees to 0.0366 m /
2.631 degrees. Other poses also change continuously; the CSV retains those
changes even when the pass/fail category stays the same.

Neither control is suitable for adoption. Source-only depth eligibility is
still the fair ORB baseline for a future matcher comparison using that same
eligibility rule. That comparison must also retain raw and V17 references, so a
gain over the weaker ORB control alone cannot qualify a matcher for integration.

The next candidate remains shared RootSIFT features with nearest-neighbor and
SIFT-LightGlue matching. Freeze the source-depth rule, native PnP, camera,
resolution and 39 pairs. Declare the proposed 2048-feature budget separately
from the old ORB budget. Measure accurate applied poses, damage to accurate raw
poses, spatial/depth support and full extraction/matching/pose cost. This stage
does not execute that candidate. Depth noise, calibration and timing remain
possible contributors alongside matching and observation geometry.

## Frozen implementation and protocol

Entry point: `python/run_orb_observation_controls.py`. All data, source manifest,
parent study, native binary and output locations are explicit. Existing output
directories are rejected. No previous source files or results are modified.

The runner first verifies the V17 fit seal and binary hash, reconstructs all
39 frozen cases, recomputes descriptor matches, and checks exact double-depth
row identities, distances and order against the old proposal cache. The
original policy stays Hamming <= 64, strict ratio 0.75, mutual best matches.
Original feature rows, including odd source IDs, are retained.

Both arms use the same original `sample_depth`: rounded pixel center, interior
3x3 neighborhood, all depths nonzero, range <= 0.05 m, depth scale 5000. No new
depth range or quality threshold is added. The second arm removes only the
target-depth gate. Every first-arm row is therefore present in the second arm.
All eligible rows are fitted; there is no independent held-out residual metric.

Historical intrinsics remain [517.3, 516.5, 318.6, 255.3], with zero additional
distortion and unchanged 640 x 480 input coordinates. The existing native
`PnpPoseEstimator` and adapter binary are reused without rebuilding or modifying
solver settings. Requested seed is 17; OpenCV 4.6's legacy sampler uses a fixed
local RNG, so no independent multi-seed robustness is claimed.

Each arm has a 4 GiB address-space limit and 60 s timeout. Fitting and input
hashes are sealed before the separate evaluator opens dataset GT. The new fit
loop takes 17.86 s, excluding input reconstruction and subsequent GT evaluation.
Peak measured worker RSS is 80.90 MiB; median worker time is 0.222 / 0.226 s for
the two arms. These are pose-worker costs, not extraction/matching end-to-end
benchmarks. No GPU inference is used.

## Reproduction and verification

Use the existing Python environment with NumPy, SciPy and Pillow, plus the native
OpenCV runtime paths required by the V17 binary. Set `PYTHONPATH` to the project's
`python` directory and `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`.
The source manifest maps scene names to `matching`, `features`, `frames`,
`dataset` and `truth` paths as in V17.

```sh
python python/run_orb_observation_controls.py \
  --protocol results/camera_convention_preflight.json \
  --sources "$SOURCES_JSON" --parent-study "$V17_STUDY" \
  --native-binary "$NATIVE_PNP" --output "$NEW_STUDY"
python python/run_camera_pnp_comparison.py evaluate \
  --study "$NEW_STUDY" --sources "$SOURCES_JSON" --output "$NEW_EVALUATION"
```

The full Python suite passes 160 tests with zero skips, including native adapter
and exchange tests using the existing binaries. Five new tests cover source
holes/discontinuities, missing target depth, odd/even row preservation, duplicate
identities and overwrite refusal. All 211 pre-existing project files match
their pre-run SHA-256 values and have recoverable backups. Raw inputs, native
reports, failures, process logs and the complete fit seal remain outside tracked
files in the V18 evidence directory.

Cumulatively, V17 supplies the reusable bridge to the existing native PnP;
V18 supplies reproducible observation controls and shows that recovering more
of the same ORB observations does not resolve the accuracy problem. These are
selected-pair findings, with no BA, trajectory or online tracking claim.
