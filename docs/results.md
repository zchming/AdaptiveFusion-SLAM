# Results and operating tradeoffs

The extension adds independent auxiliary pose correction while retaining native
LK tracking. Accuracy, runtime and service-delivery measurements use separate
protocols. The original risk-adaptive mapping benchmark remains a distinct
system result.

## Trajectory accuracy

[trajectory_summary.csv](../results/trajectory_summary.csv) contains all five
recordings, input/valid counts, GT associations, ATE and translation/rotation
RPE. Evaluation uses the common valid timestamps, rigid SE(3) alignment for
ATE and the existing trajectory evaluation protocol. Failed frames and unmatched
GT timestamps are counted rather than silently removed from input coverage.
The risk controller is disabled in these auxiliary comparisons.

| Sequence | Frames | Native ATE (cm) | Auxiliary ATE (cm) | Change |
|---|---:|---:|---:|---:|
| RPY | 694 | 8.273 | 9.468 | +14.4% |
| 360 | 744 | 15.103 | 11.417 | −24.4% |
| floor | 1227 | 27.323 | 8.012 | −70.7% |
| plant | 1126 | 14.105 | 8.468 | −40.0% |
| desk2 | 620 | 13.730 | 7.863 | −42.7% |

RPY and 360 supported service development; floor, plant and desk2 were excluded
from that selection. The latter recordings were subsequently reused in resource
validation, so they are not a globally unseen test set for the entire project.
All five have complete native and corrected pose output in this evaluation.

Low-frequency assistance trades some accuracy for fewer expensive matches. In
the independent three-recording comparison, it used 423 calls versus 2,970
historical every-frame pair evaluations. Relative to that every-frame output,
floor/plant ATE increased 33.8%/28.4%, and desk2 rotation RPE increased 17.5%.
RPY ATE also increased relative to native LK. These results remain candidates
for applications that benefit from their measured accuracy/cost balance.

## CPU budget and source-prefetch latency

[auxiliary_latency.csv](../results/auxiliary_latency.csv) preserves all nine
full-recording runs, including busy drops, unfinished requests, output tails,
correction age, CPU time and GPU allocation. Timing is from recorded replay
with the configured source schedule, not physical camera/network latency.
No GT was read and no new absolute-accuracy claim is made for these profiles.

| Recording | Sparse median → prefetch median (ms) | P95 change | Sparse → prefetch CPU (s) |
|---|---:|---:|---:|
| floor | 150.35 → 142.81 | +13.6% | 16.06 → 16.09 |
| plant | 179.73 → 143.29 | −8.4% | 19.52 → 19.39 |
| desk2 | 188.16 → 153.65 | +0.7% | 17.04 → 17.65 |

Each recording/profile has one run. Plant and desk2 median latency decreased
20.3% and 18.3%; floor decreased 5.0% while its P95 increased. The RPY
development comparison had two runs per profile: median handoff latency
193.5 → 154.3 ms (−20.3%), CPU approximately unchanged, and mixed P95 direction
across repetitions. [Development measurements](../results/prefetch_development.json)

The 64-call cap leaves approximately 15.94 s of floor and 14.64 s of plant
without a new auxiliary target. Correction-age P95 reaches 13.95/12.75 s.
Desk2 has 55 eligible calls and a much shorter tail. Lower CPU totals therefore
include lower correction coverage; they do not represent equal-coverage
algorithm acceleration. Default remains the available full-call profile.

Across the nine resource-validation runs: 8,919 valid raw/corrected frame
outputs, raw-pose/map consistency, zero future-feature accesses, 787 matching
calls and 785 delivered corrections. Unfinished/default end-of-sequence work
is retained in the CSV. Prefetch completed 76 opportunities; busy opportunities
were logged. This validates isolation and causality for these recordings.

## Original system and retained negative evidence

The [v0.1 report](../results/stage_18_v0.1_release.md) records the original XYZ
mapping result and deterministic blur, dark, occlusion, noise and drop controls.
Its speed/map-size gains predate the auxiliary extension.

MAGSAC removes 2D geometrically inconsistent matches before native PnP, while
confidence fusion accounts for observation support. More inliers alone are not
an accuracy result. Earlier PnP-only and source-depth-time compensation screens
did not provide stable overall benefit; their machine-readable measurements
remain under `results/`, with full local failures and replay outputs retained.
[Provenance and reproduction](reproducibility.md) explains the recovery route.
