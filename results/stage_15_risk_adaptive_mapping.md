# Stage 15 Experiment: Risk-Adaptive Map Protection

## Policy Thresholds

- Low: `[0.00, 0.30)`
- Medium: `[0.30, 0.60)`
- High: `[0.60, 0.85)`
- Critical: `[0.85, 1.00]`

## Controlled Input

Twelve consecutive frames expose 100 candidate landmarks each. Risks are:

```text
0.05, 0.08, 0.12, 0.18, 0.28, 0.35,
0.48, 0.62, 0.72, 0.86, 0.92, 0.97
```

For this mechanism test, candidates at high and critical risk are designated
unreliable. The baseline always writes candidates; the adaptive policy follows
its map permissions.

## Result

```text
Baseline total points:             1200
Baseline unreliable points:         500
Adaptive total points:               700
Adaptive unreliable points:            0
Suppressed risky candidates:         500
Early-keyframe requests:                4
Critical frozen frames:                 3
CTest:                               14/14 passed
```

Per-frame decisions are stored in `stage_15_risk_adaptive_mapping.csv`. Unit
integration also confirms forced ORB at high risk, trusted-reference retention
at critical risk, early keyframe selection, and actual suppression inside
`SparseMap`.

This result verifies that the policy mechanisms obey predicted risk. It does
not prove that every real high-risk observation is bad or that suppressing all
of them improves ATE. Those claims require a loaded real model and paired
sequence experiments.
