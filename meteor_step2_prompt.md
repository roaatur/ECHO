# METEOR Step 2 — Feature Extraction & Spearman Analysis

## CRITICAL CHANGE: Target Metric

**Do NOT use fn_nano = num_gt - tp_nano as the target.**

Use **recoverable_fn = tp_medium - tp_nano** instead.

Why: fn_nano includes objects that NEITHER model can detect (non-COCO classes like auto-rickshaw, extremely occluded objects). These inflate the target with noise the LSTM can never predict — switching to medium won't recover them. recoverable_fn only counts what switching actually recovers, which is what the MPC decision depends on.

You already have tp_nano and tp_medium per frame from step 1. Just compute:
```python
df['recoverable_fn'] = df['tp_medium'] - df['tp_nano']
# Clamp negatives to 0 (noise where nano randomly beats medium on a frame)
df['recoverable_fn'] = df['recoverable_fn'].clip(lower=0)
```

Also compute and report the distribution of recoverable_fn: mean, std, percentiles (25th, 50th, 75th, 90th, 95th), and what fraction of frames have recoverable_fn = 0.

---

## Feature Extraction: 40 Features in 7 Categories

Extract ALL of the following per frame. Group them by category for clean code.

### Category A: Edge & Texture (8 features)
```python
import cv2
import numpy as np

def extract_edge_texture(gray):
    features = {}
    
    # A1: Edge density (Canny)
    edges = cv2.Canny(gray, 50, 150)
    features['edge_density'] = np.mean(edges > 0)
    
    # A2-A3: Gradient statistics
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    features['gradient_mag_mean'] = np.mean(mag)
    features['gradient_mag_std'] = np.std(mag)
    
    # A4: Laplacian variance (sharpness)
    features['laplacian_var'] = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # A5: High-frequency energy ratio (DCT-based)
    # Use center crop for speed (256x256)
    h, w = gray.shape
    cy, cx = h//2, w//2
    crop = gray[cy-128:cy+128, cx-128:cx+128].astype(np.float32)
    dct = cv2.dct(crop)
    total_energy = np.sum(dct**2)
    # High-freq = bottom-right quadrant of DCT
    hf_energy = np.sum(dct[128:, 128:]**2)
    features['highfreq_energy_ratio'] = hf_energy / max(total_energy, 1e-10)
    
    # A6: LBP entropy
    # Simple LBP: compare center pixel to 8 neighbors
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for i, (dy, dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        lbp |= ((gray >= shifted).astype(np.uint8) << i)
    hist = np.histogram(lbp, bins=256, range=(0, 256))[0].astype(np.float64)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    features['lbp_entropy'] = -np.sum(hist * np.log2(hist))
    
    # A7: Gabor energy (4 orientations)
    gabor_energy = 0
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0)
        filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
        gabor_energy += np.mean(np.abs(filtered))
    features['gabor_energy'] = gabor_energy / 4.0
    
    # A8: Gradient histogram entropy (orientation diversity)
    angle = np.arctan2(gy, gx)  # reuse from A2-A3
    # Only count pixels with significant gradient
    mask = mag > np.percentile(mag, 50)
    if mask.sum() > 0:
        hist_angle = np.histogram(angle[mask], bins=36, range=(-np.pi, np.pi))[0].astype(np.float64)
        hist_angle = hist_angle / hist_angle.sum()
        hist_angle = hist_angle[hist_angle > 0]
        features['gradient_hist_entropy'] = -np.sum(hist_angle * np.log2(hist_angle))
    else:
        features['gradient_hist_entropy'] = 0.0
    
    return features
```

### Category B: Luminance & Visibility (7 features)
```python
def extract_luminance(gray):
    features = {}
    
    # B1-B2: Brightness statistics
    features['brightness_mean'] = np.mean(gray)
    features['brightness_std'] = np.std(gray)
    
    # B3: Dark channel prior (fog/haze proxy)
    # Min over RGB in 15x15 patch
    # NOTE: needs BGR input, not gray — pass frame_bgr separately
    # features['dark_channel'] computed below with color input
    
    # B4-B5: Exposure extremes
    features['pct_dark_pixels'] = np.mean(gray < 40)
    features['pct_bright_pixels'] = np.mean(gray > 220)
    
    # B6: Local contrast (RMS)
    mean_val = np.mean(gray).astype(np.float64)
    features['local_contrast_rms'] = np.std(gray) / max(mean_val, 1e-10)
    
    # B7: Luminance range (robust)
    p1, p99 = np.percentile(gray, [1, 99])
    features['luminance_range'] = p99 - p1
    
    return features

def compute_dark_channel(frame_bgr, patch_size=15):
    # Min across channels
    min_channel = np.min(frame_bgr, axis=2)
    # Min filter (erosion with square kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return np.mean(dark_channel)
```

### Category C: Color & Saturation (4 features)
```python
def extract_color(frame_bgr):
    features = {}
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    
    # C1-C2: Saturation statistics
    features['saturation_mean'] = np.mean(hsv[:,:,1])
    features['saturation_std'] = np.std(hsv[:,:,1])
    
    # C3: Colorfulness (Hasler & Susstrunk)
    B, G, R = frame_bgr[:,:,0].astype(float), frame_bgr[:,:,1].astype(float), frame_bgr[:,:,2].astype(float)
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    features['colorfulness'] = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    
    # C4: Hue entropy
    hue = hsv[:,:,0]
    hist_h = np.histogram(hue, bins=36, range=(0, 180))[0].astype(np.float64)
    hist_h = hist_h / hist_h.sum()
    hist_h = hist_h[hist_h > 0]
    features['hue_entropy'] = -np.sum(hist_h * np.log2(hist_h))
    
    return features
```

### Category D: Spatial Frequency & Information (5 features)
```python
def extract_info(gray):
    features = {}
    
    # D1: Shannon entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    features['shannon_entropy'] = -np.sum(hist * np.log2(hist))
    
    # D2: 2D entropy (pixel + right neighbor joint histogram)
    g1 = gray[:, :-1].flatten()
    g2 = gray[:, 1:].flatten()
    hist2d = np.histogram2d(g1, g2, bins=64, range=[[0,256],[0,256]])[0]
    hist2d = hist2d / hist2d.sum()
    hist2d = hist2d[hist2d > 0]
    features['entropy_2d'] = -np.sum(hist2d * np.log2(hist2d))
    
    # D3: Spatial frequency
    gray_f = gray.astype(np.float64)
    rf = np.sqrt(np.mean(np.diff(gray_f, axis=1)**2))  # row frequency
    cf = np.sqrt(np.mean(np.diff(gray_f, axis=0)**2))  # column frequency
    features['spatial_frequency'] = np.sqrt(rf**2 + cf**2)
    
    # D4: Average gradient
    gx = np.abs(np.diff(gray_f, axis=1))
    gy = np.abs(np.diff(gray_f, axis=0))
    features['average_gradient'] = (np.mean(gx) + np.mean(gy)) / 2.0
    
    # D5: BRISQUE-like features (MSCN statistics)
    # Compute MSCN (Mean Subtracted Contrast Normalized) coefficients
    C = 1  # stability constant
    mu = cv2.GaussianBlur(gray_f, (7,7), 7/6)
    sigma = cv2.GaussianBlur((gray_f - mu)**2, (7,7), 7/6)
    sigma = np.sqrt(sigma) + C
    mscn = (gray_f - mu) / sigma
    features['mscn_mean'] = np.mean(mscn)
    features['mscn_std'] = np.std(mscn)
    
    return features
```
NOTE: D5 outputs 2 sub-features (mscn_mean, mscn_std). Total unique features = 41.

### Category E: Spatial Structure & Density (6 features)
```python
def extract_spatial(gray):
    features = {}
    h, w = gray.shape
    
    # E1-E3: Connected components
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    features['connected_components'] = num_labels
    
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    small_threshold = 0.01 * h * w  # 1% of image area
    small_count = np.sum(areas < small_threshold)
    features['small_component_count'] = small_count
    features['small_component_ratio'] = small_count / max(num_labels - 1, 1)
    
    # E4-E6: Strip edge densities
    strip_h = h // 3
    for i, name in enumerate(['top', 'middle', 'bottom']):
        strip = gray[i*strip_h:(i+1)*strip_h, :]
        strip_edges = cv2.Canny(strip, 50, 150)
        features[f'{name}_edge_density'] = np.mean(strip_edges > 0)
    
    return features
```

### Category F: Temporal / Motion (5 features)
```python
def extract_temporal(gray_curr, gray_prev):
    """Requires previous frame. For first frame of each clip, pass gray_prev=None."""
    features = {}
    
    if gray_prev is None:
        features['flow_magnitude_mean'] = 0.0
        features['flow_magnitude_std'] = 0.0
        features['flow_direction_entropy'] = 0.0
        features['temporal_diff_magnitude'] = 0.0
        features['temporal_diff_ratio'] = 0.0
        return features
    
    # F4-F5: Temporal difference (cheap, compute first)
    diff = cv2.absdiff(gray_curr, gray_prev)
    features['temporal_diff_magnitude'] = np.mean(diff)
    features['temporal_diff_ratio'] = np.mean(diff > 25)  # threshold=25
    
    # F1-F3: Optical flow (expensive — downsample for speed)
    small_curr = cv2.resize(gray_curr, (320, 240))
    small_prev = cv2.resize(gray_prev, (320, 240))
    flow = cv2.calcOpticalFlowFarneback(small_prev, small_curr, None,
                                         0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    features['flow_magnitude_mean'] = np.mean(mag)
    features['flow_magnitude_std'] = np.std(mag)
    
    # Flow direction entropy
    angle = np.arctan2(flow[...,1], flow[...,0])
    mask = mag > 1.0  # only significant motion
    if mask.sum() > 100:
        hist_dir = np.histogram(angle[mask], bins=36, range=(-np.pi, np.pi))[0].astype(np.float64)
        hist_dir = hist_dir / hist_dir.sum()
        hist_dir = hist_dir[hist_dir > 0]
        features['flow_direction_entropy'] = -np.sum(hist_dir * np.log2(hist_dir))
    else:
        features['flow_direction_entropy'] = 0.0
    
    return features
```

### Category G: Scale & Resolution Sensitivity (5 features)
```python
def extract_scale(gray):
    """These directly measure what 320px downsampling loses."""
    features = {}
    h, w = gray.shape
    
    # G1: FAST keypoint count at full resolution
    fast = cv2.FastFeatureDetector_create()
    kp_full = fast.detect(gray, None)
    features['fast_keypoints_full'] = len(kp_full)
    
    # G2: FAST keypoint count at 320px (nano resolution)
    # Maintain aspect ratio: 1920x1080 -> 320x180
    small = cv2.resize(gray, (320, int(h * 320 / w)))
    kp_small = fast.detect(small, None)
    features['fast_keypoints_320'] = len(kp_small)
    
    # G3: Keypoint loss ratio
    features['keypoint_loss_ratio'] = 1.0 - (len(kp_small) / max(len(kp_full), 1))
    
    # G4: Multi-scale edge ratio
    edges_full = cv2.Canny(gray, 50, 150)
    edges_small = cv2.Canny(small, 50, 150)
    ed_full = np.mean(edges_full > 0)
    ed_small = np.mean(edges_small > 0)
    features['multiscale_edge_ratio'] = ed_full / max(ed_small, 1e-10)
    
    # G5: Small edge blob count
    # Connected components of edge map, count tiny ones
    num_labels_e, _, stats_e, _ = cv2.connectedComponentsWithStats(edges_full)
    small_edge_blobs = np.sum(stats_e[1:, cv2.CC_STAT_AREA] < 16)
    features['small_edge_blob_count'] = small_edge_blobs
    
    return features
```

---

## Spearman Analysis

Run the same within-sequence Spearman analysis as before, but:

1. **Use recoverable_fn as target, not fn_nano**
2. **Compute for all 40 features**
3. **Also compute global Spearman ρ for comparison (to check for Simpson's paradox again)**

```python
from scipy.stats import spearmanr

feature_cols = [all 40 feature column names]
target = 'recoverable_fn'

results = []
for feature in feature_cols:
    # Global
    valid = df[[feature, target]].dropna()
    if valid[feature].std() > 0 and valid[target].std() > 0:
        global_rho, _ = spearmanr(valid[feature], valid[target])
    else:
        global_rho = 0
    
    # Within-sequence
    per_seq_rhos = []
    for seq, group in df.groupby('sequence'):
        if len(group) < 30:
            continue
        if group[target].std() > 0 and group[feature].std() > 0:
            rho, _ = spearmanr(group[feature], group[target])
            if not np.isnan(rho):
                per_seq_rhos.append(rho)
    
    within_rho = np.mean(per_seq_rhos) if per_seq_rhos else 0
    within_median = np.median(per_seq_rhos) if per_seq_rhos else 0
    
    results.append({
        'feature': feature,
        'category': ...,  # label A-G
        'global_rho': round(global_rho, 3),
        'within_seq_rho_mean': round(within_rho, 3),
        'within_seq_rho_median': round(within_median, 3),
        'signal_drop': round(global_rho - within_rho, 3),
        'n_valid_seqs': len(per_seq_rhos)
    })

results_df = pd.DataFrame(results).sort_values('within_seq_rho_mean', ascending=False)
print(results_df.to_string())
```

**ALSO** run Elastic Net as a second selection method:
```python
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

# Per-sequence normalization first (subtract sequence means)
for col in feature_cols + [target]:
    df[col + '_norm'] = df.groupby('sequence')[col].transform(lambda x: x - x.mean())

X = df[[f + '_norm' for f in feature_cols]].values
y = df[target + '_norm'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ElasticNet with CV to find best alpha and l1_ratio
enet = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], alphas=50, cv=5, max_iter=10000)
enet.fit(X_scaled, y)

print(f"Best alpha: {enet.alpha_:.4f}, Best l1_ratio: {enet.l1_ratio_:.2f}")
print(f"R²: {enet.score(X_scaled, y):.3f}")

# Feature importance
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': enet.coef_
}).sort_values('coefficient', key=abs, ascending=False)
print(coef_df[coef_df['coefficient'] != 0].to_string())
```

---

## Important Notes

1. **Subsample to 10fps** if processing is too slow. Every 3rd frame at 30fps. Still ~600 frames per clip. This is fine for Spearman.

2. **LBP (A6) negative index trick**: The code uses negative indices for neighbor offsets. If using np.roll, note that `−1` means different things in Python vs the literal character. Use `-1` (minus sign). If LBP is too slow, skip it — it's medium priority.

3. **Optical flow (F1-F3) is the most expensive feature** (~15ms per frame). Downsample to 320x240 before computing flow. Still captures scene transitions.

4. **Dark channel (B3)** needs the BGR frame, not grayscale. Pass both to the extraction function.

5. **DCT (A5)** uses a 256x256 center crop for speed. Full-frame DCT on 1080p is unnecessary.

6. **Save all raw feature values to CSV/parquet** so we don't need to re-extract for different analyses.

7. **Report the recoverable_fn distribution** before the Spearman table — we need to know: mean, std, % of frames with recoverable_fn=0, and percentiles.

8. **Sort the final Spearman table by within_seq_rho_mean descending** — that's the metric that matters.

---

## Success Criteria (same as before, but for recoverable_fn)

- **within_seq ρ > 0.3 for 5+ features** → proceed to full pipeline
- **within_seq ρ 0.2-0.3** → marginal, try delta features (Δ_feature = feature_t - feature_{t-1})
- **within_seq ρ < 0.2** → fundamental limitation persists
