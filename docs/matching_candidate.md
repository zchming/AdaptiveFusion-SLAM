# Learned-matching candidate status

Status: **offline candidate; disabled in production tracking**.

The candidate combines cached OpenCV RootSIFT features (2048 budget), official
SIFT-LightGlue weights, F-MAGSAC++ verification and the unchanged native PnP
backend. It is an integration of existing components, with no algorithm novelty
claim. Source depth is the original registered TUM depth; temporal compensation
is not part of the candidate.

On the 39 development pairs, the V21 candidate returns17 accurate poses versus
15 for V19 LightGlue, and damages5 previously accurate raw poses versus7. Its
interpolated translation P95 is11.38cm versus15.40cm. Three equal-count controls
return15/18/15 accurate poses; the strongest control damages3 raw-accurate pairs.
The candidate retains6/8 of V19's original rescued pairs. These selected-pair
results support retaining it, but do not establish stable trajectory benefit.

Frozen settings: original640x480 coordinates, historical intrinsics
[517.3,516.5,318.6,255.3], no additional distortion, source-depth eligibility
3x3 all nonzero with range<=0.05m and scale5000; F-MAGSAC3px/confidence0.999/
10000iterations; original PnP binary/settings and requested seed17. H is
diagnostic only. No estimated or ground-truth motion filters the matches.

Official LightGlue commit: `eb42fee2d71449efb0aa5c10549752b5d75384d8`.
SIFT weight SHA-256:
`5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0`.
Source, weights, datasets and full raw outputs remain outside Git.

Evidence and reproduction:

- [V21 comparison and equal-count controls](magsac_comparison.md)
- [Shared RootSIFT comparison](rootsift_matching_comparison.md)
- [Time-compensation control and retained negative results](depth_time_comparison.md)
- [Native PnP adapter and camera controls](camera_pnp_comparison.md)

The frozen candidate was then evaluated on48 preselected pairs/96 disjoint
images from the same four sequences, with a1second buffer from development
images. F-MAGSAC returns46/48 accurate versus45/48 LightGlue, with one gain and
no losses against LightGlue. Equal-count controls return45/43/43. One originally
accurate raw pair is still damaged. Parameters were not tuned after evaluation.
This is within-sequence heldout evidence, not unseen-sequence or full-trajectory
benefit. See [the heldout validation](matching_holdout_validation.md).

Publication-facing JSON summaries use `${AFS_EVIDENCE_ROOT}` followed by the
original cache directory basename; set this variable to the local evidence root.
Machine-specific location maps and original summaries are retained locally.

No BA or online-tracking integration is authorized by this candidate record.
Commit and push remain separate user decisions.
