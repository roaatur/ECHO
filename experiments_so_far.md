# Experiment Assessment — Spatial Feature Forecasting for Adaptive YOLO
 
The thesis is: predict when nano@320 will miss detections using cheap spatial features, so an MPC controller can preemptively switch to medium@640. The core requirement is that **spatial features correlate with miss count within a video sequence** — not just across sequences.

**This requirement has not been met.** Every experiment points to the same fundamental problem.

---

## Experiment-by-Experiment

### 1. Early LSTM Experiments (recall_gap target)

| Experiment | R² | rho |
|---|---|---|
| Image-only (14 feat) | -0.115 | 0.084 |
| Both models (32 feat) | 0.165 | 0.530 |
| LSTM_nano (23 feat) | -0.017 | 0.357 |
| LSTM_medium (23 feat) | -0.098 | 0.215 |
| Nano recall (23 feat) | -0.021 | 0.265 |

**Verdict:** Negative R² across the board except "Both models" which cheats by using detection outputs. Image-only spatial features cannot predict recall_gap. These experiments already signaled the problem early on.

### 2. Spatial Global LSTM (15 feat, fn target) — R² = 0.553

This was the high-water mark. But the diagnostic work revealed **why** it looked good: the model learned which sequences are hard vs easy (inter-sequence), not which frames within a sequence are hard (intra-sequence). The R² = 0.553 was Simpson's paradox — pooling across sequences with different difficulty levels creates a correlation that doesn't exist within any individual sequence.

### 3. Per-Sequence Normalized LSTM (latest) — norm R² = 0.201, raw R² = 0.791

This experiment correctly isolated the question. The raw R² of 0.791 looks great but is misleading — the model receives the sequence mean for free via denormalization. The **real number is the normalized R² = 0.201** — that's the actual intra-sequence predictive power. The model explains ~20% of within-sequence fn variation, and the per-sequence plots show it mostly predicts a flat line near zero while the actual fn oscillates wildly around it. The num_gt auxiliary head completely collapsed (R² = -38 to -295 on some sequences).

### 4. DETRAC Diagnostic (within-seq Spearman + windowed)

This was the most informative experiment. Key numbers:

- **Per-frame within-seq rho:** best = 0.324 (FAST keypoint count vs fn_nano)
- **90-frame windowed:** best = 0.414 (same feature)
- Windowing helps but tops out around 0.4
- recoverable_fn is slightly weaker than fn_nano as a target
- High-GT filter (num_gt >= 5) doesn't help

**The ceiling for any spatial feature on DETRAC is rho ~ 0.3–0.4 within-sequence.** That's a weak signal — you'd need rho > 0.5–0.6 for a useful regression model.

### 5. Dirichlet Binary Classification (easy/hard)

| Model | Acc | AUC | Params |
|---|---|---|---|
| C_MLP_stats | 0.705 | 0.714 | 1,538 |
| B_MLP_mean | 0.677 | 0.707 | 578 |
| A_SmallLSTM | 0.670 | 0.802 | 6,338 |
| **fast_thresh** | **0.771** | **—** | **0** |
| majority | 0.561 | — | 0 |

The trained models **lose to a simple FAST keypoint threshold** (0.771 vs 0.705 best). The uncertainty quantification didn't work — epistemic uncertainty is nearly identical for correct and wrong predictions on B and C. Only A_SmallLSTM shows a gap, but its accuracy is worst. The calibration curve is also poorly calibrated.

**Verdict from own results:** "C_MLP_stats beats majority, close to FAST. Uncertainty NOT informative."

### 6. METEOR Dashcam Validation — The Critical Test

This was supposed to save the thesis. The hypothesis: DETRAC is fixed-camera, so the scene doesn't change within a sequence — but a dashcam (METEOR) has a moving camera, so spatial features should vary meaningfully and correlate with miss count.

**The results are devastating:**

- **Max within-sequence |rho| = 0.082** across all 41 features
- That's *worse* than DETRAC (~0.3), not better
- Elastic Net: R² = 0.000, selected 1 feature out of 41 (dark_channel) with coefficient ~1e-18
- Delta features (frame-to-frame differences) also near zero
- Even the best feature (colorfulness) only hits rho = 0.082

The per-clip boxplots show the medians sitting right at zero with massive spread — the features are essentially uncorrelated with recoverable_fn on a per-sequence basis.

---

## The Core Problem

Across DETRAC (fixed camera, 15 features), DETRAC (fixed camera, 40 features), and METEOR (dashcam, 41 features), the story is consistent:

**Spatial image features do not predict detection miss count at the frame level within a video sequence.**

The reasons are likely:

1. **Miss count is dominated by object-level factors** — occlusion patterns, object size, object density, partial visibility — that aren't captured by global/regional image statistics.
2. **The features measure scene complexity, not detection difficulty.** A high-texture, bright scene can be easy (large well-separated objects) or hard (many small occluded objects). The features can't distinguish.
3. **The global R² on DETRAC was a confounder** — sequences filmed at night/rain have both different spatial statistics AND more misses, but the causal link is sequence identity, not the features themselves.
4. **METEOR being worse** suggests that camera motion adds noise to spatial features without adding predictive signal.

---

## What Works vs What Doesn't

### Works
- A simple FAST keypoint threshold gives 77% binary accuracy on DETRAC — but that's a crude sequence-level classifier, not a frame-level predictor.
- The model can tell which sequences are globally hard vs easy (R² = 0.553 / 0.791 with per-seq norm trick).

### Doesn't Work
- Frame-level regression of fn within sequences (the actual operational requirement).
- Evidential uncertainty as a useful signal for MPC.
- Dashcam as a path to stronger within-sequence correlation.
- All 41 spatial features on METEOR (rho < 0.1).

---

## Summary

The spatial-features-only approach has been thoroughly tested across 2 datasets, 3 target formulations (fn, recall_gap, recoverable_fn), multiple architectures (LSTM, MLP, evidential, Dirichlet), 15–41 features, and various normalization/windowing strategies. The within-sequence signal consistently comes in at rho = 0.0–0.4 on DETRAC and ~0.0 on METEOR. This is likely insufficient for a reliable MPC-based model switching system.

