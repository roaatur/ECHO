#!/usr/bin/env python3
"""Extract comprehensive spatial features on ALL 100 DETRAC sequences (~138K frames).

Feature categories:
  A. Resolution-sensitivity (why nano@320 fails — information lost at 2× downsample)
  B. Density proxies (num_gt is the oracle — approximate it without detection)
  C. Scene complexity / texture
  D. Existing validated features (from prior experiments)

All features are class-agnostic (no detection needed at extraction time).
All features are computed at full 960×540 resolution.

Output: new_features/new_all_features.csv
"""

import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(ROOT / "res_experiment"))
from res_evaluate_detrac import IMG_BASE  # noqa: E402

DETRAC_CSV = ROOT / "res_experiment" / "detrac_results.csv"
OUT_CSV = SCRIPT_DIR / "new_all_features.csv"

FRAME_H, FRAME_W = 540, 960
STRIP_H = FRAME_H // 3  # 180 rows per strip

# Reusable OpenCV objects
DARK_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
FAST_DETECTOR = cv2.FastFeatureDetector_create(threshold=20)
FAST_DETECTOR_320 = cv2.FastFeatureDetector_create(threshold=20)
MOG2_DICT = {}  # per-sequence background subtractors

# Precomputed FFT masks (same for every frame of same size)
_FFT_MASKS = {}

def _get_fft_masks(h, w):
    """Cache low-pass masks for FFT energy computation."""
    key = (h, w)
    if key not in _FFT_MASKS:
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        r_nyquist = min(h, w) // 4
        r_half = min(h, w) // 8
        _FFT_MASKS[key] = {
            "low_nyquist": dist_sq <= r_nyquist ** 2,
            "very_low": dist_sq <= r_half ** 2,
        }
    return _FFT_MASKS[key]


# ═══════════════════════════════════════════════════════════════════
# CATEGORY A: Resolution-Sensitivity Features
# These measure content that is lost when downsampling 640 → 320px.
# ═══════════════════════════════════════════════════════════════════

def extract_resolution_features(gray, gray_f, bgr):
    """Resolution-sensitivity features — the thesis core."""
    h, w = gray.shape
    feats = {}

    # A1: FFT high-frequency energy ratio
    # Content above Nyquist frequency of 320px is destroyed by 2× downsample
    f = np.fft.fft2(gray_f)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) ** 2
    total_energy = mag.sum() + 1e-10
    masks = _get_fft_masks(h, w)
    low_energy = mag[masks["low_nyquist"]].sum()
    feats["fft_high_freq_energy_ratio"] = float((total_energy - low_energy) / total_energy)

    # A2: FFT energy in the critical band (between Nyquist/2 and Nyquist)
    very_low_energy = mag[masks["very_low"]].sum()
    mid_band_energy = low_energy - very_low_energy
    feats["fft_critical_band_ratio"] = float(mid_band_energy / total_energy)

    # A3: Laplacian variance (sharpness — classic focus measure)
    lap = cv2.Laplacian(gray_f, cv2.CV_64F)
    feats["laplacian_variance"] = float(lap.var())

    # A4: Laplacian mean absolute (less sensitive to outliers)
    feats["laplacian_mean_abs"] = float(np.abs(lap).mean())

    # A5: Multi-scale edge density ratio (fine / coarse edges)
    edges_fine = cv2.Canny(gray, 50, 150)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3.0)
    edges_coarse = cv2.Canny(blurred, 50, 150)
    fine_density = edges_fine.mean() / 255.0
    coarse_density = edges_coarse.mean() / 255.0
    feats["edge_density_fine"] = float(fine_density)
    feats["edge_density_coarse"] = float(coarse_density)
    feats["edge_fine_coarse_ratio"] = float(fine_density / (coarse_density + 1e-8))

    # A6: FAST keypoint count at full res (960×540)
    kps_full = FAST_DETECTOR.detect(gray, None)
    n_kps_full = len(kps_full)
    feats["fast_keypoints_full"] = float(n_kps_full / (h * w))

    # A7: FAST keypoint count at half-res (480×270) — simulates nano scale
    gray_half = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    kps_half = FAST_DETECTOR_320.detect(gray_half, None)
    n_kps_half = len(kps_half)
    feats["fast_keypoints_half"] = float(n_kps_half / (gray_half.shape[0] * gray_half.shape[1]))

    # A8: Keypoint loss ratio (what fraction of structure survives downsampling)
    feats["keypoint_loss_ratio"] = float(1.0 - n_kps_half / (n_kps_full + 1e-8))

    # A9: Spatial frequency (sqrt of mean squared gradient in both directions)
    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    sf = np.sqrt(np.mean(gx ** 2) + np.mean(gy ** 2))
    feats["spatial_frequency"] = float(sf)

    # A10: Gradient magnitude statistics
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    feats["gradient_magnitude_mean"] = float(grad_mag.mean())
    feats["gradient_magnitude_std"] = float(grad_mag.std())

    # A11: DCT high-frequency energy (block-based, like JPEG)
    # Process 8×8 blocks across the image, average high-freq coefficient energy
    # Use downsampled grid for speed: sample every 4th block
    dct_high_total = 0.0
    dct_total = 0.0
    n_blocks = 0
    step = 32  # sample every 4th 8×8 block position
    for by in range(0, h - 7, step):
        for bx in range(0, w - 7, step):
            block = gray_f[by:by + 8, bx:bx + 8]
            dct_block = cv2.dct(block.astype(np.float32))
            energy = float(np.sum(dct_block ** 2))
            # High-freq = outside top-left 4×4 quadrant
            high_energy = float(np.sum(dct_block[4:, :] ** 2) +
                                np.sum(dct_block[:4, 4:] ** 2))
            dct_high_total += high_energy
            dct_total += energy
            n_blocks += 1
    feats["dct_high_freq_ratio"] = float(dct_high_total / (dct_total + 1e-10))

    # A12: Gabor energy at critical frequency (Nyquist band for 2× downsample)
    # σ and λ tuned so the filter responds to features at 320px Nyquist
    # Use 2 orientations (H + V) on half-res for speed — still captures the signal
    gabor_energy = 0.0
    gray_gabor = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    gray_gabor_f = gray_gabor.astype(np.float64)
    for theta in [0, np.pi / 2]:
        kern = cv2.getGaborKernel((15, 15), sigma=2.5, theta=theta,
                                  lambd=4.0, gamma=0.5, psi=0)
        resp = cv2.filter2D(gray_gabor_f, cv2.CV_64F, kern)
        gabor_energy += np.mean(resp ** 2)
    feats["gabor_nyquist_energy"] = float(gabor_energy / 2.0)

    # A13: SSIM between full-res and half-res (upscaled back)
    # Measures actual information loss from downsampling
    gray_half_up = cv2.resize(gray_half, (w, h), interpolation=cv2.INTER_LINEAR)
    # Simplified SSIM: luminance + contrast components
    mu1, mu2 = gray_f.mean(), gray_half_up.astype(np.float64).mean()
    sig1 = gray_f.std()
    sig2 = gray_half_up.astype(np.float64).std()
    sig12 = np.mean((gray_f - mu1) * (gray_half_up.astype(np.float64) - mu2))
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sig12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sig1 ** 2 + sig2 ** 2 + C2))
    feats["downsample_ssim"] = float(ssim_val)
    # Lower SSIM = more information lost = more fn_nano expected
    feats["downsample_info_loss"] = float(1.0 - ssim_val)

    return feats


# ═══════════════════════════════════════════════════════════════════
# CATEGORY B: Density Proxy Features
# num_gt is the oracle — these approximate object count without detection.
# ═══════════════════════════════════════════════════════════════════

def extract_density_features(gray, gray_f, bgr, seq_name, prev_gray):
    """Density proxy features — approximate num_gt without running a detector."""
    h, w = gray.shape
    feats = {}

    # B1: MOG2 background subtraction (per-sequence stateful model)
    if seq_name not in MOG2_DICT:
        mog = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=True
        )
        MOG2_DICT[seq_name] = mog
    else:
        mog = MOG2_DICT[seq_name]

    fg_mask = mog.apply(bgr)
    # fg_mask: 255=foreground, 127=shadow, 0=background
    fg_binary = (fg_mask == 255).astype(np.uint8)
    shadow_binary = (fg_mask == 127).astype(np.uint8)
    total_px = h * w

    feats["foreground_pixel_ratio"] = float(fg_binary.sum() / total_px)
    feats["shadow_pixel_ratio"] = float(shadow_binary.sum() / total_px)

    # B2: Connected components on foreground mask (blob count ≈ object count)
    # Morphological cleanup first
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_clean = cv2.morphologyEx(fg_binary, cv2.MORPH_CLOSE, kernel)
    fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg_clean, connectivity=8
    )
    # Filter out tiny noise blobs (< 200 px) and huge blobs (> 30% of frame)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        valid = (areas >= 200) & (areas < 0.3 * total_px)
        feats["foreground_blob_count"] = int(valid.sum())
        if valid.sum() > 0:
            valid_areas = areas[valid]
            feats["foreground_blob_area_mean"] = float(valid_areas.mean() / total_px)
            feats["foreground_blob_area_std"] = float(valid_areas.std() / total_px)
            # Perimeter-to-area ratio (high = many small objects)
            perimeters = stats[1:, cv2.CC_STAT_WIDTH] * 2 + stats[1:, cv2.CC_STAT_HEIGHT] * 2
            valid_pa = perimeters[valid].astype(float) / (valid_areas + 1e-8)
            feats["foreground_blob_pa_ratio"] = float(valid_pa.mean())
        else:
            feats["foreground_blob_area_mean"] = 0.0
            feats["foreground_blob_area_std"] = 0.0
            feats["foreground_blob_pa_ratio"] = 0.0
    else:
        feats["foreground_blob_count"] = 0
        feats["foreground_blob_area_mean"] = 0.0
        feats["foreground_blob_area_std"] = 0.0
        feats["foreground_blob_pa_ratio"] = 0.0

    # B3: Foreground edge length (edges within foreground only)
    edges = cv2.Canny(gray, 50, 150)
    fg_edges = cv2.bitwise_and(edges, edges, mask=fg_clean)
    feats["foreground_edge_density"] = float(fg_edges.sum() / (255.0 * total_px))

    # B4: Frame differencing (simple motion indicator, density proxy)
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        feats["temporal_diff_mean"] = float(diff.mean())
        feats["temporal_diff_std"] = float(diff.astype(np.float64).std())
        # Motion area: pixels with significant change
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        feats["motion_pixel_ratio"] = float(motion_mask.sum() / (255.0 * total_px))
    else:
        feats["temporal_diff_mean"] = 0.0
        feats["temporal_diff_std"] = 0.0
        feats["motion_pixel_ratio"] = 0.0

    return feats


# ═══════════════════════════════════════════════════════════════════
# CATEGORY C: Scene Complexity / Texture Features
# ═══════════════════════════════════════════════════════════════════

def extract_complexity_features(gray, gray_f, bgr):
    """Scene complexity and texture features."""
    h, w = gray.shape
    feats = {}

    # C1: Image entropy (histogram-based)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-10)
    feats["image_entropy"] = float(scipy_entropy(hist_norm, base=2))

    # C2: GLCM-derived features (co-occurrence texture)
    # Compute simplified GLCM at distance=1, angle=0 on downsampled image for speed
    gray_ds = cv2.resize(gray, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    # Quantize to 32 levels for manageable GLCM
    gray_q = (gray_ds // 8).astype(np.int32)
    levels = 32
    glcm = np.zeros((levels, levels), dtype=np.float64)
    # Vectorized GLCM: horizontal pairs (d=1, angle=0)
    left, right = gray_q[:, :-1].ravel(), gray_q[:, 1:].ravel()
    np.add.at(glcm, (left, right), 1)
    np.add.at(glcm, (right, left), 1)
    # Vertical pairs (d=1, angle=90)
    top_p, bot_p = gray_q[:-1, :].ravel(), gray_q[1:, :].ravel()
    np.add.at(glcm, (top_p, bot_p), 1)
    np.add.at(glcm, (bot_p, top_p), 1)
    glcm_sum = glcm.sum() + 1e-10
    glcm_norm = glcm / glcm_sum

    # GLCM contrast
    ii, jj = np.meshgrid(range(levels), range(levels), indexing='ij')
    feats["glcm_contrast"] = float(np.sum(glcm_norm * (ii - jj) ** 2))

    # GLCM homogeneity (inverse difference moment)
    feats["glcm_homogeneity"] = float(np.sum(glcm_norm / (1 + np.abs(ii - jj))))

    # GLCM energy (angular second moment)
    feats["glcm_energy"] = float(np.sum(glcm_norm ** 2))

    # GLCM entropy
    glcm_flat = glcm_norm.flatten()
    glcm_flat = glcm_flat[glcm_flat > 0]
    feats["glcm_entropy"] = float(-np.sum(glcm_flat * np.log2(glcm_flat)))

    # C3: Colorfulness (Hasler & Süsstrunk 2003)
    B, G, R = bgr[:, :, 0].astype(float), bgr[:, :, 1].astype(float), bgr[:, :, 2].astype(float)
    rg = R - G
    yb = 0.5 * (R + G) - B
    feats["colorfulness"] = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2) +
        0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )

    # C4: Mean brightness and contrast
    feats["mean_brightness"] = float(gray.mean())
    feats["brightness_std"] = float(gray_f.std())

    # C5: Dark channel mean (weather/haze proxy — He et al. 2009)
    min_channel = np.min(bgr, axis=2)
    dark = cv2.erode(min_channel, DARK_KERNEL)
    feats["dark_channel_mean"] = float(dark.mean())

    # C6: Small edge components (from Canny — clutter indicator)
    edges = cv2.Canny(gray, 50, 150)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        feats["small_edge_components"] = int(np.sum(areas < 500))
        feats["total_edge_components"] = int(n_labels - 1)
    else:
        feats["small_edge_components"] = 0
        feats["total_edge_components"] = 0

    # C7: Overall edge density
    feats["overall_edge_density"] = float(edges.mean() / 255.0)

    # C8: RMS contrast
    feats["rms_contrast"] = float(np.sqrt(np.mean((gray_f - gray_f.mean()) ** 2)))

    # C9: Saturation statistics (HSV)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(float)
    feats["saturation_mean"] = float(sat.mean())
    feats["saturation_std"] = float(sat.std())

    return feats


# ═══════════════════════════════════════════════════════════════════
# CATEGORY D: Per-Strip Features (spatial distribution)
# ═══════════════════════════════════════════════════════════════════

def extract_strip_features(gray):
    """Per-strip features capturing spatial distribution of complexity."""
    strips = {
        "top": gray[0:STRIP_H, :],
        "mid": gray[STRIP_H:2 * STRIP_H, :],
        "bot": gray[2 * STRIP_H:FRAME_H, :],
    }
    feats = {}
    strip_vals = {}
    for prefix, strip in strips.items():
        gray_f = strip.astype(np.float64)

        # Laplacian variance
        lap_var = float(cv2.Laplacian(gray_f, cv2.CV_64F).var())
        feats[f"{prefix}_laplacian_var"] = lap_var

        # Gradient magnitude std
        gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
        grad_std = float(np.sqrt(gx ** 2 + gy ** 2).std())
        feats[f"{prefix}_gradient_std"] = grad_std

        # Edge density
        edges = cv2.Canny(strip, 50, 150)
        ed = float(edges.mean() / 255.0)
        feats[f"{prefix}_edge_density"] = ed

        # FFT high-freq energy (use cached masks)
        h_s, w_s = strip.shape
        f = np.fft.fft2(gray_f)
        fshift = np.fft.fftshift(f)
        mag_s = np.abs(fshift) ** 2
        total_e = mag_s.sum() + 1e-10
        strip_masks = _get_fft_masks(h_s, w_s)
        low_e = mag_s[strip_masks["very_low"]].sum()  # use very_low for strips
        feats[f"{prefix}_high_freq_energy"] = float((total_e - low_e) / total_e)

        strip_vals[prefix] = {"edge_density": ed, "gradient_std": grad_std,
                              "laplacian_var": lap_var}

    # Ratios (top/bottom) — spatial distribution indicators
    for metric in ["edge_density", "gradient_std", "laplacian_var"]:
        top_v = strip_vals["top"][metric]
        bot_v = strip_vals["bot"][metric]
        feats[f"ratio_top_bot_{metric}"] = float(top_v / (bot_v + 1e-8))

    return feats


# ═══════════════════════════════════════════════════════════════════
# BRISQUE-like features (MSCN — Mean Subtracted Contrast Normalized)
# No-reference image quality — Mittal et al. 2012
# ═══════════════════════════════════════════════════════════════════

def extract_brisque_features(gray_f):
    """Simplified BRISQUE: MSCN coefficient statistics.

    MSCN = (I - mu) / (sigma + 1) where mu and sigma are local.
    The distribution shape correlates with image quality.
    """
    feats = {}

    # Local mean and variance via Gaussian blur
    kernel_size = 7
    mu = cv2.GaussianBlur(gray_f, (kernel_size, kernel_size), 1.166)
    mu_sq = cv2.GaussianBlur(gray_f ** 2, (kernel_size, kernel_size), 1.166)
    sigma = np.sqrt(np.abs(mu_sq - mu ** 2))

    # MSCN coefficients
    mscn = (gray_f - mu) / (sigma + 1.0)

    feats["mscn_mean"] = float(mscn.mean())
    feats["mscn_std"] = float(mscn.std())
    feats["mscn_kurtosis"] = float(
        np.mean((mscn - mscn.mean()) ** 4) / (mscn.std() ** 4 + 1e-10) - 3.0
    )
    feats["mscn_skewness"] = float(
        np.mean((mscn - mscn.mean()) ** 3) / (mscn.std() ** 3 + 1e-10)
    )

    # Pairwise products (adjacent MSCN coefficients — captures structural distortion)
    # Horizontal pairs
    h_pairs = mscn[:, :-1] * mscn[:, 1:]
    feats["mscn_h_pair_mean"] = float(h_pairs.mean())
    feats["mscn_h_pair_std"] = float(h_pairs.std())

    # Vertical pairs
    v_pairs = mscn[:-1, :] * mscn[1:, :]
    feats["mscn_v_pair_mean"] = float(v_pairs.mean())
    feats["mscn_v_pair_std"] = float(v_pairs.std())

    return feats


# ═══════════════════════════════════════════════════════════════════
# Master extraction function
# ═══════════════════════════════════════════════════════════════════

def extract_all_features(bgr, prev_gray, seq_name):
    """Extract all features from one BGR frame.

    Returns (feature_dict, gray).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float64)

    feats = {}

    # Category A: Resolution-sensitivity
    feats.update(extract_resolution_features(gray, gray_f, bgr))

    # Category B: Density proxies
    feats.update(extract_density_features(gray, gray_f, bgr, seq_name, prev_gray))

    # Category C: Scene complexity
    feats.update(extract_complexity_features(gray, gray_f, bgr))

    # Category D: Per-strip spatial distribution
    feats.update(extract_strip_features(gray))

    # Category E: BRISQUE-like (MSCN)
    feats.update(extract_brisque_features(gray_f))

    return feats, gray


# ═══════════════════════════════════════════════════════════════════
# Main extraction loop (crash-safe, per-sequence append)
# ═══════════════════════════════════════════════════════════════════

def run_extraction():
    """Extract features for all 100 sequences."""
    print("=" * 70)
    print("NEW FEATURE EXTRACTION — COMPREHENSIVE SET")
    print("=" * 70)

    # Load detection results (nano_320 only for target + medium for recoverable_fn)
    print("\nLoading detrac_results.csv ...", flush=True)
    full_df = pd.read_csv(DETRAC_CSV)

    nano_df = full_df[full_df["model"] == "nano_320"][
        ["sequence", "frame_id", "fn", "tp", "weather", "num_gt"]
    ].copy()
    nano_df = nano_df.rename(columns={"fn": "fn_nano", "tp": "tp_nano"})

    medium_df = full_df[full_df["model"] == "medium_640"][
        ["sequence", "frame_id", "tp"]
    ].copy()
    medium_df = medium_df.rename(columns={"tp": "tp_medium"})

    # Merge to compute recoverable_fn
    det_df = nano_df.merge(medium_df, on=["sequence", "frame_id"], how="left")
    det_df["recoverable_fn"] = (det_df["tp_medium"] - det_df["tp_nano"]).clip(lower=0)

    all_sequences = sorted(det_df["sequence"].unique())
    total_seqs = len(all_sequences)
    total_frames = len(det_df)
    print(f"  {total_frames:,} frames across {total_seqs} sequences\n", flush=True)

    # Determine feature columns from a test extraction
    # (run on a dummy image to get column names)
    test_img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    test_feats, _ = extract_all_features(test_img, None, "__test__")
    FEATURE_COLS = sorted(test_feats.keys())
    META_COLS = ["sequence", "frame_id", "weather", "fn_nano", "num_gt",
                 "tp_nano", "tp_medium", "recoverable_fn"]
    COL_ORDER = META_COLS + FEATURE_COLS
    del MOG2_DICT["__test__"]

    print(f"  Extracting {len(FEATURE_COLS)} features per frame")
    print(f"  Feature columns: {FEATURE_COLS[:5]}... (+ {len(FEATURE_COLS) - 5} more)\n")

    # Crash safety: check which sequences are already done
    done_sequences = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV, nrows=0)
        # Verify column compatibility
        if list(existing.columns) == COL_ORDER:
            existing_full = pd.read_csv(OUT_CSV)
            done_sequences = set(existing_full["sequence"].unique())
            done_frames = len(existing_full)
            print(f"  Resuming — {len(done_sequences)} sequences already done, "
                  f"skipping them.\n", flush=True)
        else:
            print("  WARNING: Existing CSV has different columns. Starting fresh.\n")
            OUT_CSV.unlink()
            done_frames = 0
    else:
        done_frames = 0

    grouped = det_df.groupby("sequence")
    seq_done = len(done_sequences)
    frames_done = done_frames
    times = []

    for seq_name in all_sequences:
        if seq_name in done_sequences:
            continue

        seq_df = grouped.get_group(seq_name).sort_values("frame_id")
        img_dir = IMG_BASE / seq_name
        weather = seq_df["weather"].iloc[0]
        n_frames = len(seq_df)

        # Reset MOG2 for this sequence (fresh background model)
        if seq_name in MOG2_DICT:
            del MOG2_DICT[seq_name]

        print(f"  [{seq_done + 1:3d}/{total_seqs}] {seq_name} ({weather}): "
              f"{n_frames} frames ...", end="", flush=True)
        t0 = time.time()
        prev_gray = None
        records = []

        for _, row in seq_df.iterrows():
            frame_id = int(row["frame_id"])
            img_path = img_dir / f"img{frame_id:05d}.jpg"
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"\n    WARNING: Could not read {img_path}", flush=True)
                continue

            feats, gray = extract_all_features(bgr, prev_gray, seq_name)
            prev_gray = gray

            # Add metadata
            feats["sequence"] = seq_name
            feats["frame_id"] = frame_id
            feats["weather"] = weather
            feats["fn_nano"] = int(row["fn_nano"])
            feats["num_gt"] = int(row["num_gt"])
            feats["tp_nano"] = int(row["tp_nano"])
            feats["tp_medium"] = int(row["tp_medium"])
            feats["recoverable_fn"] = int(row["recoverable_fn"])
            records.append(feats)

        # Append this sequence to CSV
        seq_out = pd.DataFrame(records)[COL_ORDER]
        write_header = not OUT_CSV.exists()
        seq_out.to_csv(OUT_CSV, mode="a", index=False, header=write_header)

        elapsed = time.time() - t0
        fps = n_frames / elapsed if elapsed > 0 else 0
        times.append(elapsed / max(n_frames, 1))
        seq_done += 1
        frames_done += len(records)

        # Free MOG2 memory for completed sequence
        if seq_name in MOG2_DICT:
            del MOG2_DICT[seq_name]

        avg_per_frame = sum(times) / len(times)
        remaining_frames = total_frames - frames_done
        eta_min = remaining_frames * avg_per_frame / 60

        print(f"  {elapsed:.1f}s ({fps:.0f} fps)  "
              f"[{frames_done:,}/{total_frames:,} frames, ETA {eta_min:.1f}min]",
              flush=True)

    print(f"\n{'=' * 70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    df = pd.read_csv(OUT_CSV)
    print(f"  Output: {OUT_CSV}")
    print(f"  Shape:  {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Sequences: {df['sequence'].nunique()}")

    # Quick NaN audit
    feat_cols = [c for c in df.columns if c not in META_COLS]
    nan_counts = df[feat_cols].isna().sum()
    if nan_counts.any():
        print("\n  NaN warnings:")
        for col in feat_cols:
            n = nan_counts[col]
            if n > 0:
                print(f"    {col}: {n} NaN values")
    else:
        print(f"  No NaN values in any of {len(feat_cols)} feature columns.")

    return df


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_extraction()
