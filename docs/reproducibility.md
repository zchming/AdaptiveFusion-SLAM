# Reproduction and provenance

The runnable service entry is documented in [auxiliary_service.md](auxiliary_service.md).
[configs/auxiliary.json](../configs/auxiliary.json) freezes matching, confidence,
graph and resource parameters without private machine paths. Runtime manifests
add explicit local paths, input inventories, binary/model hashes and call budgets.
Every run requires a new output directory.

## Evaluation separation

- Native raw trajectories are compared against the frozen reference only after
  the run is sealed. Reference poses are not input to the live correction worker.
- Ground truth is used only in the separate historical ATE/RPE evaluation. The
  sparse/prefetch resource comparisons use no GT and retain all output frames.
- RPY/360 developed the default integration; floor/plant/desk2 tested that frozen
  service. Prefetch was developed on RPY and then tested on the latter three
  recordings with matching and confidence unchanged.
- `raw.tum` and `corrected.tum` are separate streams. No correction is written
  into tracking, map state or local bundle adjustment.

## Compact published evidence

| File | Evidence |
|---|---|
| [trajectory_summary.csv](../results/trajectory_summary.csv) | Five-recording native/auxiliary accuracy, coverage, RPE and every-frame controls |
| [auxiliary_latency.csv](../results/auxiliary_latency.csv) | Every independent CPU/prefetch run, failures, tails and correction freshness |
| [prefetch_development.json](../results/prefetch_development.json) | Both repetitions of every development profile |
| [provenance.json](../results/provenance.json) | Source summary hashes, upstream model/source pin and tested versions |
| [stage_18_v0.1_release.md](../results/stage_18_v0.1_release.md) | Original risk-adaptive mapping and degradation results |

Historical pair-level CSVs and protocols remain available in `results/`.
Large replay caches, full worker logs, images, weights and upstream checkouts
are kept outside tracked files. Evidence cache names are recorded in provenance
so a local evidence archive can be mapped to the published summaries.

## Historical documentation

The earlier audit notes and experiment diary are preserved in Git at
`6dd2bd718bbf3e63c1833bbdce6dbb2c8e502cbe`. View a prior document with:

```bash
git show 6dd2bd718bbf3e63c1833bbdce6dbb2c8e502cbe:docs/progress.md
```

The source workspace and its uncommitted experiment documents are preserved
separately, with a content-hashed archive made before this release cleanup.
The concise documentation reorganizes usage and findings; it does not rewrite
public history or erase the historical result files.

## Local checks

After building, run `ctest --test-dir "$BUILD" --output-on-failure`. For Python
checks, expose the package and native executables explicitly:

```bash
export PYTHONPATH="$PWD/python"
export AFS_NATIVE_PNP="$BUILD/native_pnp_pair"
export AFS_CERES_REPLAY="$BUILD/replay_local_ba"
export AFS_ODOMETRY="$BUILD/run_rgbd_odometry"
python -B -m unittest discover -v -s tests/python
```

Set the platform's shared-library search path when using a nonstandard native
installation. Without the native executable variables, six integration checks
are skipped with their individual reasons.
