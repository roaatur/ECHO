# 40 Candidate Spatial Features for Scene Complexity Prediction
## ECHO Project — Feature Selection Survey
### Target: recoverable_fn = tp_medium − tp_nano

---

## Why These Features?

Nano@320 misses objects that Medium@640 catches primarily because:
1. **Small objects vanish** at 320px resolution (biggest factor — 90% of gap is resolution)
2. **Low contrast objects** lose distinguishing detail when downsampled
3. **Cluttered/textured scenes** create aliasing artifacts at low resolution
4. **Poor visibility** (dark, fog, rain) means fine details matter more
5. **Motion blur** compounds with low resolution
6. **Dense scenes** have more objects to lose

Good candidate features should detect these conditions from the raw frame, cheaply, without running any detector.

---

## Category A: Edge & Texture (8 features)

These capture scene structural complexity — more structure = more potential for small objects to be lost at low resolution.

| # | Feature | Computation | Cost | Why it might predict recoverable_fn |
|---|---------|-------------|------|-------------------------------------|
| A1 | **Edge density (Canny)** | `mean(Canny(gray, 50, 150) > 0)` | ~2ms | More edges = more structure = more objects to miss at low res |
| A2 | **Gradient magnitude mean** | `mean(sqrt(Sobel_x² + Sobel_y²))` | ~2ms | Average structural intensity |
| A3 | **Gradient magnitude std** | `std(sqrt(Sobel_x² + Sobel_y²))` | ~0ms (computed with A2) | High variation = mix of busy/empty regions = small objects hiding in busy areas |
| A4 | **Laplacian variance** | `var(Laplacian(gray))` | ~1ms | Sharpness/blur proxy — blurry frames lose more at low res |
| A5 | **High-freq energy ratio** | DCT → ratio of high-freq coefficients to total energy | ~3ms | High-frequency content is exactly what gets destroyed at 320px downsampling |
| A6 | **Local Binary Pattern (LBP) entropy** | Compute LBP histogram → entropy | ~5ms | Texture diversity — uniform textures lose less at low res than diverse textures |
| A7 | **Gabor energy (multi-orientation)** | Mean response of 4-orientation Gabor bank | ~8ms | Captures oriented edges at multiple scales, more informative than isotropic Canny |
| A8 | **Gradient histogram entropy** | Histogram of gradient orientations → entropy | ~2ms | Many edge orientations = complex scene = more detection difficulty |

---

## Category B: Luminance & Visibility (7 features)

Dark, foggy, and low-contrast conditions make objects harder to detect — especially at low resolution where fine details are lost.

| # | Feature | Computation | Cost | Why it might predict recoverable_fn |
|---|---------|-------------|------|-------------------------------------|
| B1 | **Brightness mean** | `mean(gray)` | <1ms | Dark frames lose more detail at low res |
| B2 | **Brightness std** | `std(gray)` | <1ms | Low contrast = objects blend into background |
| B3 | **Dark channel prior** | `min over RGB channels in local patch` | ~3ms | Fog/haze proxy (He et al., 2009) — fog degrades detection, especially at low res |
| B4 | **Percentage of dark pixels** | `mean(gray < 40)` | <1ms | Underexposed regions lose detail first at low res |
| B5 | **Percentage of bright pixels** | `mean(gray > 220)` | <1ms | Overexposed/glare regions (headlights at night) |
| B6 | **Local contrast (RMS)** | `std(gray) / mean(gray)` (Weber/Michelson variant) | <1ms | Normalized contrast — low = hard scene for any detector |
| B7 | **Luminance range** | `max(gray) - min(gray)` after removing 1% outliers | <1ms | Dynamic range proxy |

---

## Category C: Color & Saturation (4 features)

Color information helps distinguish objects from background. Low saturation (fog, overcast) means objects are harder to separate.

| # | Feature | Computation | Cost | Why it might predict recoverable_fn |
|---|---------|-------------|------|-------------------------------------|
| C1 | **Saturation mean** | `mean(HSV[:,:,1])` | ~1ms | Low saturation = washed out = harder detection |
| C2 | **Saturation std** | `std(HSV[:,:,1])` | ~0ms | High variation = some objects visually distinct, others not |
| C3 | **Colorfulness** | Hasler & Süsstrunk metric: `sqrt(σ_rgyb² + μ_rgyb²)` | ~2ms | Perceptual color diversity — colorful scenes have more distinguishable objects |
| C4 | **Hue entropy** | Histogram of H channel → Shannon entropy | ~1ms | More hue variety = more visually distinct objects |

---

## Category D: Spatial Frequency & Information (5 features)

These measure how much "information" is in the image. More information = more potential objects = more to lose.

| # | Feature | Computation | Cost | Why it might predict recoverable_fn |
|---|---------|-------------|------|-------------------------------------|
| D1 | **Shannon entropy** | Grayscale histogram → `-Σ p·log2(p)` | <1ms | Information content — high entropy = complex scene |
| D2 | **2D entropy** | Joint histogram of pixel + neighbor → 2D entropy | ~3ms | Captures spatial structure, not just intensity distribution |
| D3 | **Spatial frequency** | `sqrt(RF² + CF²)` where RF=row freq, CF=col freq | ~2ms | Overall activity level of the image |
| D4 | **Average gradient** | `mean(|dx| + |dy|) / 2` | ~1ms | Image clarity and detail level — NR-IQA standard metric |
| D5 | **BRISQUE features** (subset) | MSCN coefficients → GGD fit → shape + variance | ~5ms | No-reference quality features strongly correlated with detection degradation |

---

## Category E: Spatial Structure & Density (6 features)

These capture where things are in the frame and how dense/cluttered the scene is.

| # | Feature | Computation | Cost | Why it might predict recoverable_fn |
|---|---------|-------------|------|-------------------------------------|
| E1 | **Connected components** | Otsu threshold → `connectedComponents()` | ~3ms | Proxy for number of distinct objects/regions |
| E2 | **Small component count** | Connected components with area < 1% of image | ~0ms (from E1) | Small blobs = small objects = likely missed at 320px |
| E3 | **Small component ratio** | Small components / total components | ~0ms | What fraction of the scene is small stuff |
| E4 | **Top-third edge density** | Canny on top 1/3 of frame | ~1ms | Distant objects (horizon) — nano misses these most |
| E5 | **Middle-third edge density** | Canny on middle 1/3 | ~1ms | Mid-range objects |
| E6 | **Bottom-third edge density** | Canny on bottom 1/3 | ~1ms | Near objects — usually caught by both models |

---

## Category F: Temporal / Motion (5 features)

Critical for dashcam — camera movement creates scene transitions that change detection difficulty.

| # | Feature | Computation | Cost | Why it might predict recoverable_fn |
|---|---------|-------------|------|-------------------------------------|
| F1 | **Optical flow magnitude mean** | `calcOpticalFlowFarneback()` → `mean(mag)` | ~15ms | Camera motion = scene changing = features should change with difficulty |
| F2 | **Optical flow magnitude std** | `std(mag)` from above | ~0ms (from F1) | Mixed motion = some objects moving differently = harder detection |
| F3 | **Optical flow direction entropy** | Histogram of flow angles → entropy | ~1ms | Random motion directions = chaotic scene |
| F4 | **Temporal diff magnitude** | `mean(|frame_t - frame_{t-1}|)` | ~2ms | Cheap proxy for motion without full optical flow |
| F5 | **Temporal diff ratio** | `mean(|frame_t - frame_{t-1}| > threshold)` | ~1ms | Fraction of frame that changed — captures how dynamic the scene is |

---

## Category G: Scale & Resolution Sensitivity (5 features)

These directly target the resolution gap — features that measure what gets lost when downsampling from 640 to 320.

| # | Feature | Computation | Cost | Why it might predict recoverable_fn |
|---|---------|-------------|------|-------------------------------------|
| G1 | **FAST keypoint count** | `FastFeatureDetector.detect()` | ~3ms | Interest points — more keypoints = more detail = more lost at 320px |
| G2 | **FAST keypoint count at 320px** | Downsample to 320 → FAST → count | ~2ms | Directly measures surviving detail at nano's resolution |
| G3 | **Keypoint loss ratio** | `1 - (G2 / G1)` | ~0ms | Fraction of detail lost at 320px — directly measures what nano can't see |
| G4 | **Multi-scale edge ratio** | Edge density at 640px / edge density at 320px | ~3ms | How much edge structure survives downsampling |
| G5 | **Small edge blob count** | Number of edge-connected components < 16px² | ~2ms | Tiny structures that will vanish at 320px |

---

## Summary Table: All 40 Features

| # | Feature | Category | Approx Cost | Expected ρ with recoverable_fn |
|---|---------|----------|-------------|-------------------------------|
| A1 | Edge density | Edge/Texture | 2ms | Medium |
| A2 | Gradient magnitude mean | Edge/Texture | 2ms | Medium |
| A3 | Gradient magnitude std | Edge/Texture | 0ms | Medium |
| A4 | Laplacian variance | Edge/Texture | 1ms | Medium-High |
| A5 | High-freq energy ratio | Edge/Texture | 3ms | High |
| A6 | LBP entropy | Edge/Texture | 5ms | Medium |
| A7 | Gabor energy | Edge/Texture | 8ms | Medium |
| A8 | Gradient histogram entropy | Edge/Texture | 2ms | Medium |
| B1 | Brightness mean | Luminance | <1ms | Low-Medium |
| B2 | Brightness std | Luminance | <1ms | Medium |
| B3 | Dark channel prior | Luminance | 3ms | Medium-High |
| B4 | % dark pixels | Luminance | <1ms | Low-Medium |
| B5 | % bright pixels | Luminance | <1ms | Low |
| B6 | Local contrast (RMS) | Luminance | <1ms | Medium |
| B7 | Luminance range | Luminance | <1ms | Low |
| C1 | Saturation mean | Color | 1ms | Medium |
| C2 | Saturation std | Color | 0ms | Medium |
| C3 | Colorfulness | Color | 2ms | Medium |
| C4 | Hue entropy | Color | 1ms | Low-Medium |
| D1 | Shannon entropy | Info/Frequency | <1ms | Medium |
| D2 | 2D entropy | Info/Frequency | 3ms | Medium-High |
| D3 | Spatial frequency | Info/Frequency | 2ms | Medium |
| D4 | Average gradient | Info/Frequency | 1ms | Medium |
| D5 | BRISQUE (subset) | Info/Frequency | 5ms | High |
| E1 | Connected components | Spatial | 3ms | Medium-High |
| E2 | Small component count | Spatial | 0ms | High |
| E3 | Small component ratio | Spatial | 0ms | High |
| E4 | Top-third edge density | Spatial | 1ms | High |
| E5 | Middle-third edge density | Spatial | 1ms | Medium |
| E6 | Bottom-third edge density | Spatial | 1ms | Low |
| F1 | Optical flow magnitude | Temporal | 15ms | High |
| F2 | Optical flow std | Temporal | 0ms | Medium-High |
| F3 | Optical flow direction entropy | Temporal | 1ms | Medium |
| F4 | Temporal diff magnitude | Temporal | 2ms | Medium |
| F5 | Temporal diff ratio | Temporal | 1ms | Medium |
| G1 | FAST keypoint count | Scale | 3ms | Medium-High |
| G2 | FAST keypoints at 320px | Scale | 2ms | Medium |
| G3 | Keypoint loss ratio | Scale | 0ms | Very High |
| G4 | Multi-scale edge ratio | Scale | 3ms | Very High |
| G5 | Small edge blob count | Scale | 2ms | High |

---

## Feature Selection Strategy

### Phase 1: Spearman ρ (filter method)
- Compute within-sequence Spearman ρ for all 40 features vs recoverable_fn
- Drop anything with |ρ| < 0.15 (weaker than DETRAC's average)
- Keep top 15-20 features

### Phase 2: Elastic Net (embedded method)
- Train Elastic Net (L1 + L2 regularization) on surviving features → recoverable_fn
- α = 0.5 (equal L1/L2) as starting point
- L1 component will zero out redundant features automatically
- Evaluate with per-sequence R² (not overall R²!)
- Cross-validate α and L1/L2 ratio

### Phase 3: Mutual Information (nonlinear check)
- Compute MI between top features and recoverable_fn
- Catches nonlinear relationships Spearman might miss
- Compare MI ranking to Spearman ranking — large discrepancies = interesting features

### Why Elastic Net over plain Lasso or Ridge?
- Lasso (L1 only): can't handle correlated features well — if A1, A2, A3 are correlated, it picks one arbitrarily
- Ridge (L2 only): keeps all features, no selection
- Elastic Net: selects features while handling correlated groups — picks one representative from each correlated group

### Final Selection Criteria
- Features must survive both Spearman (>0.3) and Elastic Net (non-zero coefficient)
- Prefer features that are cheap (<5ms) — deployment budget matters
- Prefer features from different categories — avoid redundancy
- Target: 5-8 final features for the LSTM

---

## Top Predictions (Before Seeing Data)

Features I expect to rank highest on METEOR dashcam:

1. **G3 — Keypoint loss ratio**: Directly measures what 320px loses. Should have highest ρ.
2. **G4 — Multi-scale edge ratio**: Same logic — resolution-gap proxy.
3. **E2 — Small component count**: Small objects = exactly what nano misses.
4. **F1 — Optical flow magnitude**: Dashcam-specific — scene transitions drive difficulty changes.
5. **A5 — High-freq energy ratio**: High-freq content is what downsampling destroys.
6. **E4 — Top-third edge density**: Distant objects cluster at horizon, lost at 320px.
7. **B3 — Dark channel prior**: Fog/haze makes everything harder, especially at low res.
8. **A4 — Laplacian variance**: Blur proxy — already in your chapter 4 feature set.

---

## Implementation Notes

### Total compute budget for all 40 features: ~90ms per frame
- This is less than 1 YOLO-medium inference (250ms)
- At 10fps subsampling: 10 features extractions/sec is feasible
- After selection, 5-8 features should take ~15-25ms total

### Optical flow (F1-F3) requires consecutive frame pairs
- First frame of each sequence: set to 0
- Consider subsampling flow computation (every 3rd frame) if too slow
- Temporal diff (F4-F5) is a cheap alternative if flow is too expensive

### BRISQUE (D5) needs scikit-image or custom implementation
- Can use `brisque` pip package or compute MSCN features manually
- Only extract the GGD shape and variance parameters (2 features from full BRISQUE)
- Skip the full quality score, just use the features

### Keypoint loss ratio (G3) computation:
```python
fast = cv2.FastFeatureDetector_create()
kp_full = len(fast.detect(frame_gray, None))
small = cv2.resize(frame_gray, (320, 180))  # match nano resolution
kp_small = len(fast.detect(small, None))
keypoint_loss_ratio = 1 - (kp_small / max(kp_full, 1))
```
