# Stage 14 Experiment: Temporal Risk Prediction Baseline

## Setup

The predictor encodes each five-frame window into latest, mean, and slope for
seven geometric-health signals. Logistic regression uses training-only
standardization, balanced class weights, L2 regularization, 1,500 batch-gradient
iterations, learning rate 0.05, and decision threshold 0.5.

Training contains 14 synthetic episodes: ten with failures and four stable.
The held-out set contains eight separate episodes: five with failures and three
stable. Failure timing and degradation duration vary by episode.

## Result

```text
Training samples:          202
Held-out test samples:     119
Positive test samples:      15
True positives:             15
False negatives:             0
False positives:             5
True negatives:             99
Accuracy:              0.957983
AUROC:                 0.997436
AUPRC:                 0.983824
Brier score:           0.0366757
Mean warning lead:             2 frames
CTest:                     13/13 passed
```

Positive risks range from `0.940214` to approximately `1.0`. Negative risks
range from `0.0100751` to `0.960348`; the overlap produces five conservative
false warnings at threshold 0.5. Individual held-out predictions are stored in
`results/stage_14_simulated_risk_predictions.csv`.

This controlled experiment verifies learning, held-out evaluation, probability
output, and metric computation. The synthetic trends are deliberately regular,
so these numbers must not be presented as performance on real RGB-D data.
