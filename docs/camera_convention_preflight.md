# Camera/data convention preflight

**Input integrity checks pass; a camera-convention difference remains to be
tested. No accuracy gain has been measured and no default is changed.** This
step prepares a controlled comparison of existing components for engineering
performance, with no requirement to invent an algorithm. It does not run pose
estimation, replay a sequence or evaluate a new trajectory.

## Verified current inputs

| Cached sequence | Indexed RGB / depth PNGs | Processed frames | Feature files | Native feature depth samples checked | Candidate pairs |
|---|---:|---:|---:|---:|---:|
| xyz | 798 / 798 | 792 | 114 | 113,947 | 6 |
| desk | 613 / 595 | 573 | 81 | 79,844 | 6 |
| room | 1,362 / 1,360 | 1,352 | 246 | 238,580 | 6 |
| teddy | 1,419 / 1,418 | 1,401 | 269 | 259,676 | 21 |

All 8,363 indexed PNGs pass PNG header and CRC verification: RGB is 640x480,
8-bit truecolor; depth is 640x480, 16-bit single-channel. All 710 feature files
pass their strict existing reader. All 692,047 native nearest-depth samples
match the source PNG values exactly. Feature RGB/depth timestamps agree with
their image indices, RGB timestamps agree with cached source-frame records,
and completion counts agree with the frozen V9 protocol. Every feature camera
is `(517.3, 516.5, 318.6, 255.3)` and depth configuration is `(5000, 0.1, 8)`.
All 39 proposal identities, record counts and parity split counts are retained.

These are integrity and consistency checks, not an independent calibration or
depth-accuracy measurement. PNG verification does not establish pose accuracy.

## Camera and depth findings

The [official TUM format and calibration documentation](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
specifies depth in units of 1/5000 metre, zero as missing, and depth already
registered to RGB. Its Freiburg depth scaling corrections are already applied;
do not multiply the supplied PNG depths by 1.035 again.

The current `app/run_rgbd_odometry.cpp` uses calibrated Freiburg1 focal lengths
and principal point, while `src/pnp_pose_estimator.cpp` supplies zero distortion.
The Python pair model also uses this pinhole camera without distortion. TUM
lists nonzero Freiburg1 calibration distortion and recommends ROS default
intrinsics without additional undistortion for its registered inputs:

| Profile | fx | fy | cx | cy | Additional distortion |
|---|---:|---:|---:|---:|---|
| Historical pinhole | 517.3 | 516.5 | 318.6 | 255.3 | zero |
| Official recommended default control | 525 | 525 | 319.5 | 239.5 | zero |

The recommended default is an approximate input convention, not ground-truth
calibration. Neither adding the tabulated distortion to the current projection
nor undistorting RGB alone is justified for the existing registered depth grid.
Current code divides raw depth by 5000 once; no extra 1.035 correction was found
in the inspected frontend and pair-model paths. The source hashes are recorded
in the [preflight and planned comparison](../results/camera_convention_preflight.json).

**Conclusion:** treat these as two explicit camera profiles in a future
comparison. Do not call the historical profile a proven bug or replace the
default based on documentation alone. Historical outputs retain their original
configuration and remain valid records of what was run.

## Frozen next comparison, not executed

The planned 2x2 comparison uses both camera profiles with (1) the current Huber
3D-2D refinement and (2) the existing native `PnpPoseEstimator`, which already
performs EPNP RANSAC and iterative inlier refinement. A thin input adapter is
still needed; no new estimator is proposed. Reuse the retained OpenCV 4.6 native
runtime rather than introducing Python OpenCV 5 as an extra comparison factor.

All 39 existing pairs remain, including rejected/insufficient-support cases.
Recompute source 3D points from the original pixels and raw depth for each K;
use that same K for target projection. Keep original correspondence identity,
target-depth-valid eligibility and even-fit/odd-check partition. The input pool
therefore retains a target-depth selection dependency. No rematching, resizing,
additional undistortion, depth rescaling or V15 gate tuning is permitted.

PnP retains the existing 100 RANSAC trials, 3-pixel threshold, 0.99 confidence
and minimum six correspondences/inliers. Five declared seeds are all reported.
The local arm retains its original initialization and frozen numerical budget.
Trial counts and nonlinear evaluations are not equal compute budgets; measure
and disclose both. Native-adapter contract and synthetic checks must precede
any real pair run. No adapter is implemented or built by this preflight.

Keep the historical raw transforms as a fixed reference/fallback. They were
estimated with the historical profile, so the camera-profile comparison is
conditional on those frozen observations and raw transforms; it is not a
comparison of two independently rerun whole frontends. Fit-only results must
be sealed before reading GT. Report all returns and failures, accuracy at
3 cm/2 degrees, per-scene tails, newly damaged raw-accurate pairs and costs.
Nearest/interpolated truth and repeated seeds are sensitivity checks, not
additional independent scenes. Gains on this failure-window-selected, reused
sample must not be presented as whole-sequence or unseen-scene improvement.

The JSON records engineering continuation/stop rules, resource caps and the
39 exact proposal hashes. Its status is `planned_not_executed`; no pose result
or pass/fail performance outcome is present. Default changes and online/BA
integration require later evidence and authorization.

## Preservation and checks

Work is on `codex/camera-convention-audit`. All pre-existing uncommitted files
are backed up outside Git and must remain byte-identical. The detailed audit
script, dataset locations, timestamps and integrity evidence stay outside Git
in the `camera-convention-audit-20261022-*` cache directory. The public JSON
contains concise counts, source/proposal hashes and the future protocol.

No implementation changed, so Python/C++ regression suites were not rerun for
these two documentation/configuration files. Actual checks are the complete
PNG/feature/native-depth scan, source/old-evidence hashes, local document links,
JSON/schema-summary checks, whitespace and private-path inspection. No new
experiment, model download, training, default change, commit or push occurred.
V15's failed acceptance and V16's consistency-only conclusions remain unchanged.
