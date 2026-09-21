# Stage 16 Experiment: Saved Model and Online Risk Loop

## Model Round Trip

A temporal logistic model trained on 202 synthetic windows is saved with a
versioned header and loaded into a fresh predictor. Held-out probabilities
match the original model within `1e-12`.

## Causal Fixture Run

The saved model is passed to `run_rgbd_odometry` on the repository's three-frame
fixture. The fixture is only 2-by-2 pixels, so it initializes once and then loses
tracking twice.

```text
frame 0: applied low,      next low
frame 1: applied low,      tracking lost -> next critical
frame 2: applied critical, tracking lost -> next critical
```

Frame 1's failure changes only frame 2's policy, proving that the online loop
does not use future/current results retroactively. At frame 2, new landmarks are
disabled and the map is frozen.

## Verification

```text
Saved history length:             5
Saved temporal feature dimension: 21
Probability round-trip tolerance: < 1e-12
Hysteresis 0.61 -> 0.58:          remains high
Hysteresis 0.58 -> 0.54:          drops to medium
CTest:                            14/14 passed
```

Files `stage_16_simulated_risk_model.txt` and
`stage_16_simulated_predictions.csv` are reproducibility artifacts. They are
synthetic and are not calibrated for deployment on real RGB-D sequences.
