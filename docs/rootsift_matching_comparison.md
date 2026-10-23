# V19: shared RootSIFT matching comparison

On the frozen 39 pairs, SIFT-LightGlue produces 15 accurate returned poses versus
12 for nearest-neighbor matching on exactly the same RootSIFT features. Applied
accurate poses increase from 9 to 14, and interpolated-GT translation P95 drops
from 0.4325 m to 0.1540 m. This supports retaining LightGlue as the stronger
matcher candidate while keeping the native PnP backend fixed.

## Pose results

Accuracy requires <= 0.03 m translation and <= 2 degrees rotation under both
nearest and interpolated GT alignment. Returned-pose counts include exact raw
fallbacks. Newly damaged pairs were accurate under raw poses and fail after the
candidate is applied.

| Arm | Applied / 39 | Accurate / 39 | Applied and accurate | Newly damaged | Rescued raw-inaccurate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw reference | — | 14 | — | 0 | 0 |
| V17 native, half-split ORB | 21 | 13 | 6 | 4 | 3 |
| V18 ORB, all source-depth-valid | 26 | 6 | 3 | 9 | 1 |
| Shared RootSIFT + mutual ratio NN | 29 | 12 | 9 | 7 | 5 |
| Shared RootSIFT + SIFT-LightGlue | 37 | 15 | 14 | 7 | 8 |

The NN-to-LightGlue comparison isolates matching: six pairs become accurate and
three lose accuracy. LightGlue recovers eight raw-inaccurate pairs but damages
seven raw-accurate pairs, leaving a net gain of one over raw. It therefore does
not qualify for an online replacement. The ORB-to-RootSIFT comparison also changes
the detector, descriptor and feature budget, so its whole gain cannot be
attributed to LightGlue.

| Interpolated GT, all returned poses | RootSIFT NN | RootSIFT LightGlue |
| --- | ---: | ---: |
| Median translation error, m | 0.05539 | 0.03337 |
| P95 translation error, m | 0.43249 | 0.15397 |
| Median rotation error, degrees | 1.80161 | 1.51263 |
| P95 rotation error, degrees | 9.48142 | 4.10976 |

Nearest-GT accuracy counts independently equal 12 and 15 as well. Continuous
errors for both alignments remain in the complete result table.

## Observation support and remaining failures

NN supplies 4,058 matches / 3,057 source-depth-valid observations across the 39
pairs. LightGlue supplies 15,254 / 11,455. Median occupied 4x4 image cells among
accepted PnP inliers increases from 8 to 11. There are no support-count failures:
all inputs have at least six eligible observations. The native estimator rejects
10 NN pairs and two LightGlue pairs (room 141-144 and 142-144); every rejection
retains the exact raw transform. No matching/process failures, timeouts or OOMs
occurred.

LightGlue's seven newly damaged pairs are desk 22-23, room 141-143, room 142-143,
teddy 110-112, teddy 157-160, teddy 158-159 and teddy 158-160. They include both
near-threshold errors and larger errors; none are dropped. For example, teddy
158-159 has 98 PnP inliers spanning 11 cells but reaches 0.1194 m / 3.733 degrees
maximum error across the two GT alignments. More inliers and broader support
alone cannot certify an accurate pose.

The next useful step is to diagnose these seven cases from existing outputs:
separate incorrect/clustered correspondences, depth discontinuity or RGB-depth
registration effects, and model/calibration inconsistency. Keep PnP fixed until
that evidence identifies a concrete change. Do not add another matcher or tune
PnP thresholds based only on the remaining rejection count. No additional
experiment, BA, complete-sequence or online integration is executed in V19.

## Fixed implementation

New entry point: `python/run_rootsift_comparison.py`, with separate `match` and
`fit` commands. Data/source manifest, V18 baseline study, LightGlue checkout,
weights, native binary and new output directories are explicit. Existing output
directories are refused. Existing project files are not edited.

Both matchers share one feature file per image, including identical pixel
coordinates, descriptors, scales and orientations. Extraction uses OpenCV SIFT
with a 2048-feature cap, contrast threshold 0.0066667, edge threshold 10 and four
octave layers; official radius-zero duplicate filtering and response top-k are
applied. Images remain 640x480, OpenCV integer BGR-to-gray, with no resizing or
pixel offset. Official RootSIFT conversion is used. NN uses Euclidean distance,
strict ratio 0.8 and mutual nearest neighbors. LightGlue uses the existing
official SIFT weights and default 9 layers, depth confidence 0.95, width
confidence 0.99, filter threshold 0.1, flash enabled and mp disabled.

The cached official source commit is
`eb42fee2d71449efb0aa5c10549752b5d75384d8`; weight SHA-256 is
`5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0`.
Weights are loaded explicitly from the supplied file; no download is performed.
Feature tensors use XY [1,N,2], descriptors [1,N,128], scale/orientation [1,N],
orientation in radians and explicit WH [1,2]. RootSIFT feature-row IDs are retained
through matching and depth filtering. Both match lists are sorted by source ID
before the unchanged native PnP call.

Source-depth eligibility is unchanged from V18: 3x3 neighborhood, all nonzero,
depth range <= 0.05 m and scale 5000. Target depth does not gate either matcher.
Historical intrinsics [517.3,516.5,318.6,255.3], zero additional distortion, the
original native PnP binary and requested seed 17 remain fixed. All eligible
observations are fitted; no held-out reprojection metric is claimed. Matching
and fitting do not read dataset GT. The existing evaluator opens GT only after
the new fit is sealed. These are the same selected development pairs, not an
independent generalization set.

## Measured cost

Across 25 unique images, shared SIFT extraction takes 0.701 s. NN matching takes
0.245 s in total (median 7.04 ms/pair); LightGlue matching including device
transfer and result copy takes 1.065 s (median 15.84 ms/pair). Model loading and
GPU transfer take 0.239 s. The combined matching phase takes 3.023 s, including
both arms, feature caching and pair-output writes. No warm-up run is excluded.

LightGlue's measured Torch peak is 128.74 MiB allocated / 158 MiB reserved,
including resident model memory, within the 11 GiB allocator cap on the local
12 GB GPU. This is not a whole-card sampled peak. Matching-process peak RSS is
1,522.23 MiB. The NN arm runs on CPU; its recorded CUDA resident memory belongs
to the already-loaded shared model.

The 78 isolated CPU pose workers take 18.59 s as a phase. Summed native solver
time is 0.0895 s for NN and 0.0984 s for LightGlue; worker sums are 9.103 / 9.246 s,
including Python startup and bridge/file overhead. These phase measurements
exclude top-level interpreter startup and GT evaluation. They do not establish
online tracking speed. Full per-pair extraction, matching and pose components
are retained rather than reporting GPU matching alone as total pipeline cost.

## Artifacts and reproduction

- [All 156 evaluation rows](../results/rootsift_matching_results.csv)
- [Scene and aggregate counts](../results/rootsift_matching_summary.csv)
- [Per-pair observation support and costs](../results/rootsift_matching_diagnostics.csv)
- [Machine-readable summary](../results/rootsift_matching_summary.json)

Use the existing LightGlue environment for `match`, and the existing geometry
environment plus native runtime library paths for `fit` and `evaluate`.
Set `PYTHONPATH` to this project's `python` directory and the BLAS/OpenMP thread
environment variables to 1. Output paths must be new.

```sh
python python/run_rootsift_comparison.py match \
  --baseline-study "$V18_STUDY" --sources "$SOURCES_JSON" \
  --lightglue-source "$LIGHTGLUE_SOURCE" --weights "$SIFT_WEIGHTS" \
  --output "$NEW_MATCHING"
python python/run_rootsift_comparison.py fit \
  --matching "$NEW_MATCHING" --baseline-study "$V18_STUDY" \
  --native-binary "$NATIVE_PNP" --output "$NEW_STUDY"
python python/run_camera_pnp_comparison.py evaluate \
  --study "$NEW_STUDY" --sources "$SOURCES_JSON" --output "$NEW_EVALUATION"
```

Targeted checks cover NN mutuality, insufficient neighbors and source-depth-only
row adaptation. Runtime checks cover feature dimensions/norms, match index
uniqueness, input identity and exact raw fallback. No full regression suite is
rerun. Raw features, matches, all rejections and process logs remain in the
external V19 evidence directory named in the JSON summary. No commit or push is
made.

Cumulative engineering contribution: the reusable native PnP bridge and ORB
controls now extend to a controlled learned-matching path. The measured matching
gain identifies a stronger candidate; the retained seven damaged pairs define
the concrete remaining robustness problem.
