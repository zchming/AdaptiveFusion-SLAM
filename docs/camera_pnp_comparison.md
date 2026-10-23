# Frozen camera x existing-estimator comparison (V17)

**Implementation/contract checks PASS; engineering performance screen FAIL.**
All 39 pairs and 468 arms completed, yielding 936 evaluation rows with no
unavailable truth. Existing native PnP helps some pairs relative to the local
refinement but does not meet the frozen non-degradation rules against raw.
The ROS-default camera control does not improve the aggregate success count.
No default or integration change follows.

| Camera / estimator | Applied / 39 | Returned accurate in both time protocols / 39 | Applied and accurate / 39 | Raw-accurate pairs made inaccurate |
|---|---:|---:|---:|---:|
| Raw historical poses, reference | not an update | 14 | not an update | 0 |
| Historical K, local Huber | 23 | 10 | 5 | 7 |
| Historical K, native PnP | 21 | 13 | 6 | 4 |
| ROS-default K, local Huber | 23 | 9 | 4 | 7 |
| ROS-default K, native PnP | 21 | 11 | 4 | 4 |

Accuracy means translation <=3 cm and rotation <=2 degrees under both existing
timestamp protocols. Returned accuracy includes raw fallback; applied accuracy
does not. The last column counts raw-accurate returns crossing the bound under
either protocol. Every native requested-seed label gives the same numerical
outcome; the sampler limitation below explains why this is not five effective
random trials. The [full results](../results/camera_pnp_results.csv),
[per-scene/per-seed summaries](../results/camera_pnp_summary.csv) and
[acceptance/provenance](../results/camera_pnp_acceptance.json) preserve every
failure and degradation.

| Scene | Raw accurate | Historical local / native | ROS-default local / native |
|---|---:|---:|---:|
| xyz development, 6 pairs | 1 | 2 / 2 | 2 / 1 |
| desk, 6 pairs | 2 | 0 / 2 | 0 / 1 |
| room, 6 pairs | 5 | 3 / 4 | 3 / 4 |
| teddy, 21 pairs | 6 | 5 / 5 | 4 / 5 |

Preserved gains and failures, using interpolated truth:

- Desk 22->24: historical native PnP improves raw translation 3.09->2.06 cm
  (rotation 1.14->1.29 degrees); local refinement fails to converge and returns
  raw. ROS-default native PnP instead returns 3.38 cm / 1.59 degrees, outside
  the absolute bound.
- Desk 23->24: native PnP still damages an accurate raw pose, 1.94 cm /
  1.40 degrees -> 3.70 cm / 2.40 degrees. Room 141->142 likewise changes raw
  1.20 cm / 0.72 degrees to 5.21 cm / 1.53 degrees.
- Teddy 201->203: both native camera profiles reject and preserve raw
  69.92 cm / 17.23 degrees exactly, avoiding the local historical candidate's
  3.686 m / 109.72 degrees. This is successful rejection, not accurate tracking.
- xyz 23->24: historical native PnP returns 2.46 cm / 0.40 degrees, whereas
  ROS-default native PnP returns 7.89 cm / 2.58 degrees. Documentation preference
  alone therefore cannot justify replacing this project's historical default.

Historical native PnP has meaningful applied gains in desk and teddy, but still
introduces four inaccurate returns and fails scene-median non-degradation.
Every configuration fails the declared engineering screen; no useful result is
discarded and no threshold or seed is selected after evaluation. This rules out
promoting these tested combinations on current evidence, not all possible
PnP, camera or correspondence improvements.

## Recorded checks and resource use

The new build passes 17/17 CTest and 155/155 Python tests, zero skips, before
real-cache fitting. The eleven new tests cover metric/direction recovery under
both K profiles, controlled outliers, identity mapping, repeatability, invalid
input and overwrite refusal, exact nonidentity fallback/timeout evidence,
frozen input eligibility/splits and sealed/missing-truth evaluation. All 39
historical-camera local transforms reproduce V12/V13 exactly (maximum matrix
difference 0), including all original applied flags and failure reasons.

All arm processes exit successfully without timeout; there is no observed OOM.
Local refinement applies 46/78 attempts: the 32 failures are 28 insufficient
support, two nonpositive predictions and two nonconvergences. Native PnP applies
210/390 attempts: the 180 rejections are 140 insufficient support and 40 native
estimator rejections. These counts include both camera profiles and all native
requested-seed labels; they are not independent scene counts.

The complete fitting command takes 118.41 s on CPU. Maximum arm process RSS is
83.20 MiB for local refinement and 81.04 MiB for the native bridge. Median
in-estimator timings are 5.61 ms local and 0.436 ms native, but those scopes
differ and include quick rejections. Median full arm-process times are about
0.243/0.242 s because startup/I/O dominates. Do not infer SLAM FPS, a matched
solver speedup or a latency guarantee from these single-run measurements.

Raw evidence, original file backups, build/test logs, all failed arms and sealed
outputs remain outside Git in `camera-pnp-v17-20261022-2fg1754z` under the existing
adaptivefusion cache. Existing uncommitted work is preserved; only the declared
CMake target addition edits a pre-existing file in this step. The original
solver, camera, matching and V15/V16 source files remain unchanged.

## Requested-seed limitation

The five global RNG seed values were supplied as frozen, and all 78 pair/camera
groups produced identical returned transforms and native inlier lists across
the five calls. Inspection of the
[OpenCV 4.6 PnP call path](https://github.com/opencv/opencv/blob/4.6.0/modules/calib3d/src/solvepnp.cpp#L285)
and its [RANSAC implementation](https://github.com/opencv/opencv/blob/4.6.0/modules/calib3d/src/ptsetreg.cpp#L158)
shows that the legacy sampler creates its own fixed local RNG. The adapter's
`cv::setRNGSeed` does not randomize that sampler. These runs establish repeatability
of this unchanged solver, not robustness to independently varied RANSAC draws.
This is an upstream control limitation of the frozen plan. No source patch,
input permutation or different solver was introduced to manufacture diversity.
The complete requested matrix remains available and the limitation is explicit
in the acceptance JSON. No independent multi-seed robustness claim is made.

This engineering comparison reuses the original C++ `PnpPoseEstimator` without
editing its implementation. `native_pnp_pair` is an input/output adapter, not
a new solver or a replacement tracking pipeline. Its only numerical controls
outside the existing estimator are the frozen OpenCV RNG seed and one OpenCV
thread. It is built against the retained OpenCV 4.6 runtime.

The unchanged [preflight](../results/camera_convention_preflight.json) specifies
39 cached descriptor pairs: xyz 6, desk 6, room 6, teddy 21. For each pair the
two camera profiles are historical `(517.3,516.5,318.6,255.3)` and TUM-recommended
ROS default `(525,525,319.5,239.5)`, both zero additional distortion. Source 3D
points are recomputed from the same native pixel/depth samples for each K;
target projection uses the same K. Matching, common target-depth eligibility,
source-ID even fit/odd check split, raw frame initialization and pixel resolution
are unchanged. The new profile is a convention control, not certified true K.

Each camera/pair runs one unchanged Huber 3D-2D local refinement and five native
PnP seeds (17,23,41,67,101): 468 attempted estimator arms in total, including
insufficient-support inputs. The local budget remains 100 function evaluations,
pixel sigma2 and block Huber delta2. Native PnP retains EPNP RANSAC, 100 trials,
3-pixel threshold, 0.99 confidence, minimum six correspondences/inliers and its
existing iterative inlier refinement/fallback. The budgets are not equivalent.
Each arm runs in its own process with a 60-second wall limit and 4 GiB address
space limit; BLAS/OMP/MKL/OpenCV threads are fixed to one. No GPU or download is
needed. Invalid input stops preparation; individual process failures remain
failed arms with an exact historical raw-pose fallback, never an identity pose.

## Input and result contract

```text
AFS_PNP_PAIR 1
CAMERA fx fy cx cy
SEED unsigned_integer
CORRESPONDENCES N
C source_feature_id X_source_m Y_source_m Z_source_m target_u target_v
...
END
```

Finite, positive-depth inputs, unique nonnegative IDs, bounded sizes and no
trailing tokens are required. This file has neither target depth nor ground
truth. Output is a fresh directory containing `report.json`; existing outputs
are refused. `applied=false` is a recorded native rejection with exit0, whereas
exit2 is an argument/input/process error. Successful `t_target_source` maps
source-camera metric points into target-camera coordinates. Inlier identities
are original source-feature IDs rather than indices silently reassigned by the
adapter. The Python bridge records native exit codes, logs and exact fallback.

The fit runner verifies frozen proposal hashes, inventories, feature/frame
identities, common depth eligibility and parity counts before any solve. Its
worker receives fit rows only. Odd-row pixel errors are descriptive checks;
they do not accept or reject the returned pose. All fit inputs, outputs, logs,
source hashes and native binary hash are sealed before the separate evaluation
command can open GT. A changed fit seal stops evaluation. Missing truth stays
unknown in the result table rather than becoming a failed or successful pose.

## Reproduction

Build the new `native_pnp_pair` target using the project's existing CMake setup
and native dependencies. A source registry JSON explicitly supplies `matching`,
`features`, `frames`, `dataset` and `truth` paths for each frozen scene. It belongs
outside Git. Python requires NumPy, SciPy and Pillow; it does not use Python
OpenCV for this adapter or reread/extract descriptors.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -B python/run_camera_pnp_comparison.py fit \
  --protocol results/camera_convention_preflight.json \
  --sources /path/to/source_registry.json \
  --native-binary /path/to/build/native_pnp_pair --output /path/to/new_study
python -B python/run_camera_pnp_comparison.py evaluate \
  --study /path/to/new_study --sources /path/to/source_registry.json \
  --output /path/to/new_evaluation
```

Supply the original registered inputs and runtime library paths; do not reuse
old output directories or overwrite historical caches. A pure Git clone lacks
the real input caches. In the recorded run, historical-camera local transforms
are also compared with all 39 V12/V13 outputs before GT evaluation. Adapter
contract tests use synthetic 3D/pixels; the full regression additionally uses
the project's synthetic C++ exchange fixtures, not new real tracking runs.

## Interpretation boundaries

Report both returned-pose accuracy (including exact raw fallbacks) and accurate
applied updates. Include native seed ranges and every per-seed result, not the
best seed. Count 39 distinct pairs, not 468 independent scenes or 936 independent
truth observations. xyz is development; desk/room/teddy are previously seen,
selected failure windows. Neither improved inlier counts nor pixel residuals
alone establish an accuracy improvement. Historical raw poses came from the
old K, so this is conditional on frozen inputs, not two complete frontend runs.

The preflight requires gain in at least two reused scenes, no new inaccurate
returns from raw-accurate pairs, and no scene median degradation beyond
5 mm/0.5 degrees in either timestamp protocol. Before GT evaluation, the gain
interpretation was recorded using the unchanged V12 `pair_value` criterion:
an applied accurate update improving translation by >=5 mm and >=10% without
>0.5-degree rotation loss, or rotation by >=0.5 degrees and >=10% without >5 mm
translation loss, under both protocols. Native seeds are assessed separately;
all declared seeds must satisfy the engineering screen. These are screening
rules, not guarantees about rare failure probability.

No defaults, V15 gate, BA observations, map identity or online trajectory state
are changed. Cumulative contribution is a reusable adapter and controlled
comparison of existing estimators with preserved failures, not a new algorithm.
