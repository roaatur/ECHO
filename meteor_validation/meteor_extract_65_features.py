#!/usr/bin/env python3
"""Extract DETRAC's full 65 features on METEOR clips.

Resizes each METEOR frame to 960×540 (matching DETRAC) and runs
the exact same extraction functions. This ensures apples-to-apples
comparison between DETRAC and METEOR feature sets.

Merges with existing detection results (meteor_detection_results.csv).

Output: meteor_validation/meteor_65_features.csv
"""

import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from meteor_common import (
    SCRIPT_DIR,
    SUBSAMPLE,
    discover_clips,
    read_video_frames,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ──────────────────────────────────────────────────────────
DET_CSV = SCRIPT_DIR / "meteor_detection_results.csv"
OUT_CSV = SCRIPT_DIR / "meteor_65_features.csv"

# ── Target resolution (match DETRAC) ──────────────────────────────
FRAME_H, FRAME_W = 540, 960
STRIP_H = FRAME_H // 3  # 180 rows per strip

# ── Reusable OpenCV objects ────────────────────────────────────────
DARK_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
FAST_DETECTOR = cv2.FastFeatureDetector_create(threshold=20)
FAST_DETECTOR_320 = cv2.FastFeatureDetector_create(threshold=20)
MOG2_DICT = {}  # per-clip background subtractors

# Precomputed FFT masks
_FFT_MASKS = {}

def _get_fft_masks(h, w):
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
# CATEGORY A: Resolution-Sensitivity Features (17)
# ═══════════════════════════════════════════════════════════════════

def extract_resolution_features(gray, gray_f, bgr):
    h, w = gray.shape
    feats = {}

    # A1: FFT high-frequency energy ratio
    f = np.fft.fft2(gray_f)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) ** 2
    total_energy = mag.sum() + 1e-10
    masks = _get_fft_masks(h, w)
    low_energy = mag[masks["low_nyquist"]].sum()
    feats["fft_high_freq_energy_ratio"] = float((total_energy - low_energy) / total_energy)

    # A2: FFT critical band ratio
    very_low_energy = mag[masks["very_low"]].sum()
    mid_band_energy = low_energy - very_low_energy
    feats["fft_critical_band_ratio"] = float(mid_band_energy / total_energy)

    # A3-A4: Laplacian
    lap = cv2.Laplacian(gray_f, cv2.CV_64F)
    feats["laplacian_variance"] = float(lap.var())
    feats["laplacian_mean_abs"] = float(np.abs(lap).mean())

    # A5: Edge densities
    edges_fine = cv2.Canny(gray, 50, 150)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3.0)
    edges_coarse = cv2.Canny(blurred, 50, 150)
    fine_density = edges_fine.mean() / 255.0
    coarse_density = edges_coarse.mean() / 255.0
    feats["edge_density_fine"] = float(fine_density)
    feats["edge_density_coarse"] = float(coarse_density)
    feats["edge_fine_coarse_ratio"] = float(fine_density / (coarse_density + 1e-8))

    # A6-A8: FAST keypoints
    kps_full = FAST_DETECTOR.detect(gray, None)
    n_kps_full = len(kps_full)
    feats["fast_keypoints_full"] = float(n_kps_full / (h * w))

    gray_half = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    kps_half = FAST_DETECTOR_320.detect(gray_half, None)
    n_kps_half = len(kps_half)
    feats["fast_keypoints_half"] = float(n_kps_half / (gray_half.shape[0] * gray_half.shape[1]))
    feats["keypoint_loss_ratio"] = float(1.0 - n_kps_half / (n_kps_full + 1e-8))

    # A9: Spatial frequency
    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    sf = np.sqrt(np.mean(gx ** 2) + np.mean(gy ** 2))
    feats["spatial_frequency"] = float(sf)

    # A10: Gradient magnitude
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    feats["gradient_magnitude_mean"] = float(grad_mag.mean())
    feats["gradient_magnitude_std"] = float(grad_mag.std())

    # A11: DCT high-frequency
    dct_high_total = 0.0
    dct_total = 0.0
    step = 32
    for by in range(0, h - 7, step):
        for bx in range(0, w - 7, step):
            block = gray_f[by:by + 8, bx:bx + 8]
            dct_block = cv2.dct(block.astype(np.float32))
            energy = float(np.sum(dct_block ** 2))
            high_energy = float(np.sum(dct_block[4:, :] ** 2) +
                                np.sum(dct_block[:4, 4:] ** 2))
            dct_high_total += high_energy
            dct_total += energy
    feats["dct_high_freq_ratio"] = float(dct_high_total / (dct_total + 1e-10))

    # A12: Gabor energy at critical frequency
    gabor_energy = 0.0
    gray_gabor = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    gray_gabor_f = gray_gabor.astype(np.float64)
    for theta in [0, np.pi / 2]:
        kern = cv2.getGaborKernel((15, 15), sigma=2.5, theta=theta,
                                  lambd=4.0, gamma=0.5, psi=0)
        resp = cv2.filter2D(gray_gabor_f, cv2.CV_64F, kern)
        gabor_energy += np.mean(resp ** 2)
    feats["gabor_nyquist_energy"] = float(gabor_energy / 2.0)

    # A13: SSIM between full-res and half-res
    gray_half_up = cv2.resize(gray_half, (w, h), interpolation=cv2.INTER_LINEAR)
    mu1, mu2 = gray_f.mean(), gray_half_up.astype(np.float64).mean()
    sig1 = gray_f.std()
    sig2 = gray_half_up.astype(np.float64).std()
    sig12 = np.mean((gray_f - mu1) * (gray_half_up.astype(np.float64) - mu2))
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sig12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sig1 ** 2 + sig2 ** 2 + C2))
    feats["downsample_ssim"] = float(ssim_val)
    feats["downsample_info_loss"] = float(1.0 - ssim_val)

    return feats


# ═══════════════════════════════════════════════════════════════════
# CATEGORY B: Density Proxy Features (10)
# ═══════════════════════════════════════════════════════════════════

def extract_density_features(gray, gray_f, bgr, clip_name, prev_gray):
    h, w = gray.shape
    feats = {}

    # MOG2 background subtraction
    if clip_name not in MOG2_DICT:
        mog = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=True)
        MOG2_DICT[clip_name] = mog
    else:
        mog = MOG2_DICT[clip_name]

    fg_mask = mog.apply(bgr)
    fg_binary = (fg_mask == 255).astype(np.uint8)
    shadow_binary = (fg_mask == 127).astype(np.uint8)
    total_px = h * w

    feats["foreground_pixel_ratio"] = float(fg_binary.sum() / total_px)
    feats["shadow_pixel_ratio"] = float(shadow_binary.sum() / total_px)

    # Connected components on foreground
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_clean = cv2.morphologyEx(fg_binary, cv2.MORPH_CLOSE, kernel)
    fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg_clean, connectivity=8)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        valid = (areas >= 200) & (areas < 0.3 * total_px)
        feats["foreground_blob_count"] = int(valid.sum())
        if valid.sum() > 0:
            valid_areas = areas[valid]
            feats["foreground_blob_area_mean"] = float(valid_areas.mean() / total_px)
            feats["foreground_blob_area_std"] = float(valid_areas.std() / total_px)
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

    # Foreground edge density
    edges = cv2.Canny(gray, 50, 150)
    fg_edges = cv2.bitwise_and(edges, edges, mask=fg_clean)
    feats["foreground_edge_density"] = float(fg_edges.sum() / (255.0 * total_px))

    # Temporal differencing
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        feats["temporal_diff_mean"] = float(diff.mean())
        feats["temporal_diff_std"] = float(diff.astype(np.float64).std())
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        feats["motion_pixel_ratio"] = float(motion_mask.sum() / (255.0 * total_px))
    else:
        feats["temporal_diff_mean"] = 0.0
        feats["temporal_diff_std"] = 0.0
        feats["motion_pixel_ratio"] = 0.0

    return feats


# ═══════════════════════════════════════════════════════════════════
# CATEGORY C: Scene Complexity / Texture (15)
# ═══════════════════════════════════════════════════════════════════

def extract_complexity_features(gray, gray_f, bgr):
    h, w = gray.shape
    feats = {}

    # Image entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-10)
    feats["image_entropy"] = float(scipy_entropy(hist_norm, base=2))

    # GLCM features
    gray_ds = cv2.resize(gray, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    gray_q = (gray_ds // 8).astype(np.int32)
    levels = 32
    glcm = np.zeros((levels, levels), dtype=np.float64)
    left, right = gray_q[:, :-1].ravel(), gray_q[:, 1:].ravel()
    np.add.at(glcm, (left, right), 1)
    np.add.at(glcm, (right, left), 1)
    top_p, bot_p = gray_q[:-1, :].ravel(), gray_q[1:, :].ravel()
    np.add.at(glcm, (top_p, bot_p), 1)
    np.add.at(glcm, (bot_p, top_p), 1)
    glcm_sum = glcm.sum() + 1e-10
    glcm_norm = glcm / glcm_sum

    ii, jj = np.meshgrid(range(levels), range(levels), indexing='ij')
    feats["glcm_contrast"] = float(np.sum(glcm_norm * (ii - jj) ** 2))
    feats["glcm_homogeneity"] = float(np.sum(glcm_norm / (1 + np.abs(ii - jj))))
    feats["glcm_energy"] = float(np.sum(glcm_norm ** 2))

    glcm_flat = glcm_norm.flatten()
    glcm_flat = glcm_flat[glcm_flat > 0]
    feats["glcm_entropy"] = float(-np.sum(glcm_flat * np.log2(glcm_flat)))

    # Colorfulness
    B, G, R = bgr[:, :, 0].astype(float), bgr[:, :, 1].astype(float), bgr[:, :, 2].astype(float)
    rg = R - G
    yb = 0.5 * (R + G) - B
    feats["colorfulness"] = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2) +
        0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    # Brightness
    feats["mean_brightness"] = float(gray.mean())
    feats["brightness_std"] = float(gray_f.std())

    # Dark channel
    min_channel = np.min(bgr, axis=2)
    dark = cv2.erode(min_channel, DARK_KERNEL)
    feats["dark_channel_mean"] = float(dark.mean())

    # Edge components
    edges = cv2.Canny(gray, 50, 150)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        feats["small_edge_components"] = int(np.sum(areas < 500))
        feats["total_edge_components"] = int(n_labels - 1)
    else:
        feats["small_edge_components"] = 0
        feats["total_edge_components"] = 0

    feats["overall_edge_density"] = float(edges.mean() / 255.0)
    feats["rms_contrast"] = float(np.sqrt(np.mean((gray_f - gray_f.mean()) ** 2)))

    # Saturation
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(float)
    feats["saturation_mean"] = float(sat.mean())
    feats["saturation_std"] = float(sat.std())

    return feats


# ═══════════════════════════════════════════════════════════════════
# CATEGORY D: Per-Strip Features (15)
# ═══════════════════════════════════════════════════════════════════

def extract_strip_features(gray):
    strips = {
        "top": gray[0:STRIP_H, :],
        "mid": gray[STRIP_H:2 * STRIP_H, :],
        "bot": gray[2 * STRIP_H:FRAME_H, :],
    }
    feats = {}
    strip_vals = {}

    for prefix, strip in strips.items():
        gray_f = strip.astype(np.float64)
        lap_var = float(cv2.Laplacian(gray_f, cv2.CV_64F).var())
        feats[f"{prefix}_laplacian_var"] = lap_var

        gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
        grad_std = float(np.sqrt(gx ** 2 + gy ** 2).std())
        feats[f"{prefix}_gradient_std"] = grad_std

        edges = cv2.Canny(strip, 50, 150)
        ed = float(edges.mean() / 255.0)
        feats[f"{prefix}_edge_density"] = ed

        h_s, w_s = strip.shape
        f = np.fft.fft2(gray_f)
        fshift = np.fft.fftshift(f)
        mag_s = np.abs(fshift) ** 2
        total_e = mag_s.sum() + 1e-10
        strip_masks = _get_fft_masks(h_s, w_s)
        low_e = mag_s[strip_masks["very_low"]].sum()
        feats[f"{prefix}_high_freq_energy"] = float((total_e - low_e) / total_e)

        strip_vals[prefix] = {"edge_density": ed, "gradient_std": grad_std,
                              "laplacian_var": lap_var}

    for metric in ["edge_density", "gradient_std", "laplacian_var"]:
        top_v = strip_vals["top"][metric]
        bot_v = strip_vals["bot"][metric]
        feats[f"ratio_top_bot_{metric}"] = float(top_v / (bot_v + 1e-8))

    return feats


# ═══════════════════════════════════════════════════════════════════
# CATEGORY E: BRISQUE-like (MSCN) Features (8)
# ═══════════════════════════════════════════════════════════════════

def extract_brisque_features(gray_f):
    feats = {}
    kernel_size = 7
    mu = cv2.GaussianBlur(gray_f, (kernel_size, kernel_size), 1.166)
    mu_sq = cv2.GaussianBlur(gray_f ** 2, (kernel_size, kernel_size), 1.166)
    sigma = np.sqrt(np.abs(mu_sq - mu ** 2))
    mscn = (gray_f - mu) / (sigma + 1.0)

    feats["mscn_mean"] = float(mscn.mean())
    feats["mscn_std"] = float(mscn.std())
    feats["mscn_kurtosis"] = float(
        np.mean((mscn - mscn.mean()) ** 4) / (mscn.std() ** 4 + 1e-10) - 3.0)
    feats["mscn_skewness"] = float(
        np.mean((mscn - mscn.mean()) ** 3) / (mscn.std() ** 3 + 1e-10))

    h_pairs = mscn[:, :-1] * mscn[:, 1:]
    feats["mscn_h_pair_mean"] = float(h_pairs.mean())
    feats["mscn_h_pair_std"] = float(h_pairs.std())

    v_pairs = mscn[:-1, :] * mscn[1:, :]
    feats["mscn_v_pair_mean"] = float(v_pairs.mean())
    feats["mscn_v_pair_std"] = float(v_pairs.std())

    return feats


# ═══════════════════════════════════════════════════════════════════
# Master extraction
# ═══════════════════════════════════════════════════════════════════

def extract_all_features(bgr, prev_gray, clip_name):
    """Extract all 65 features from one BGR frame (resized to 960×540)."""
    # Resize to match DETRAC resolution
    bgr = cv2.resize(bgr, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float64)

    feats = {}
    feats.update(extract_resolution_features(gray, gray_f, bgr))
    feats.update(extract_density_features(gray, gray_f, bgr, clip_name, prev_gray))
    feats.update(extract_complexity_features(gray, gray_f, bgr))
    feats.update(extract_strip_features(gray))
    feats.update(extract_brisque_features(gray_f))

    return feats, gray


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("METEOR — EXTRACT DETRAC's 65 FEATURES (at 960×540)")
    print("=" * 70)

    # Load detection results
    print("\nLoading detection results ...", flush=True)
    det_df = pd.read_csv(DET_CSV)

    # Separate nano and medium results
    nano_df = det_df[det_df["model"] == "nano_320"][
        ["clip", "frame_id", "fn_coco", "fn_total", "tp", "num_gt", "num_gt_coco"]
    ].copy()
    nano_df = nano_df.rename(columns={
        "fn_coco": "fn_coco_nano", "fn_total": "fn_total_nano", "tp": "tp_nano"})

    medium_df = det_df[det_df["model"] == "medium_640"][
        ["clip", "frame_id", "tp", "fn_coco", "fn_total"]
    ].copy()
    medium_df = medium_df.rename(columns={
        "tp": "tp_medium", "fn_coco": "fn_coco_medium", "fn_total": "fn_total_medium"})

    merged = nano_df.merge(medium_df, on=["clip", "frame_id"], how="left")
    merged["tp_medium"] = merged["tp_medium"].fillna(0).astype(int)
    merged["recoverable_fn"] = (merged["tp_medium"] - merged["tp_nano"]).clip(lower=0)

    all_clips = sorted(merged["clip"].unique())
    print(f"  {len(merged):,} frames across {len(all_clips)} clips\n")

    # Determine feature columns from a test run
    test_img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    test_feats, _ = extract_all_features(test_img, None, "__test__")
    FEATURE_COLS = sorted(test_feats.keys())
    del MOG2_DICT["__test__"]

    META_COLS = ["clip", "frame_id", "fn_coco_nano", "fn_total_nano",
                 "num_gt", "num_gt_coco", "tp_nano", "tp_medium",
                 "fn_coco_medium", "fn_total_medium", "recoverable_fn"]
    COL_ORDER = META_COLS + FEATURE_COLS

    print(f"  Extracting {len(FEATURE_COLS)} features per frame")
    print(f"  Resolution: {FRAME_W}×{FRAME_H} (matching DETRAC)\n")

    # Crash safety: check which clips are already done
    done_clips = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV, nrows=0)
        if list(existing.columns) == COL_ORDER:
            existing_full = pd.read_csv(OUT_CSV)
            done_clips = set(existing_full["clip"].unique())
            print(f"  Resuming — {len(done_clips)} clips already done.\n")
        else:
            print("  WARNING: Existing CSV has different columns. Starting fresh.\n")
            OUT_CSV.unlink()

    # Discover clips
    clips = discover_clips()
    clips_by_name = {c["name"]: c for c in clips}
    grouped = merged.groupby("clip")

    total_clips = len(all_clips)
    total_frames = len(merged)
    clips_done = len(done_clips)
    frames_done = 0
    times = []

    for clip_name in all_clips:
        if clip_name in done_clips:
            continue

        clip_info = clips_by_name.get(clip_name)
        if clip_info is None or clip_info["video"] is None:
            print(f"  SKIP: {clip_name} — no video found")
            continue

        clip_det = grouped.get_group(clip_name).sort_values("frame_id")
        det_lookup = clip_det.set_index("frame_id")
        n_expected = len(clip_det)

        # Reset MOG2 for this clip
        if clip_name in MOG2_DICT:
            del MOG2_DICT[clip_name]

        print(f"  [{clips_done + 1:3d}/{total_clips}] {clip_name}: "
              f"{n_expected} frames ...", end="", flush=True)
        t0 = time.time()
        prev_gray = None
        records = []

        for frame_id, bgr in read_video_frames(clip_info["video"], SUBSAMPLE):
            if frame_id not in det_lookup.index:
                # Still feed MOG2 to maintain background model
                bgr_resized = cv2.resize(bgr, (FRAME_W, FRAME_H),
                                          interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2GRAY)
                if clip_name not in MOG2_DICT:
                    mog = cv2.createBackgroundSubtractorMOG2(
                        history=200, varThreshold=50, detectShadows=True)
                    MOG2_DICT[clip_name] = mog
                MOG2_DICT[clip_name].apply(bgr_resized)
                prev_gray = gray
                continue

            feats, gray = extract_all_features(bgr, prev_gray, clip_name)
            prev_gray = gray

            row = det_lookup.loc[frame_id]
            feats["clip"] = clip_name
            feats["frame_id"] = int(frame_id)
            feats["fn_coco_nano"] = int(row["fn_coco_nano"])
            feats["fn_total_nano"] = int(row["fn_total_nano"])
            feats["num_gt"] = int(row["num_gt"])
            feats["num_gt_coco"] = int(row["num_gt_coco"])
            feats["tp_nano"] = int(row["tp_nano"])
            feats["tp_medium"] = int(row["tp_medium"])
            feats["fn_coco_medium"] = int(row["fn_coco_medium"])
            feats["fn_total_medium"] = int(row["fn_total_medium"])
            feats["recoverable_fn"] = int(row["recoverable_fn"])
            records.append(feats)

        # Append to CSV
        if records:
            seq_out = pd.DataFrame(records)[COL_ORDER]
            write_header = not OUT_CSV.exists()
            seq_out.to_csv(OUT_CSV, mode="a", index=False, header=write_header)

        elapsed = time.time() - t0
        fps = len(records) / elapsed if elapsed > 0 else 0
        times.append(elapsed / max(len(records), 1))
        clips_done += 1
        frames_done += len(records)

        if clip_name in MOG2_DICT:
            del MOG2_DICT[clip_name]

        avg_per_frame = sum(times) / len(times) if times else 0
        remaining = total_frames - frames_done
        eta_min = remaining * avg_per_frame / 60

        print(f"  {elapsed:.1f}s ({fps:.0f} fps) "
              f"[{frames_done:,}/{total_frames:,}, ETA {eta_min:.1f}min]",
              flush=True)

    print(f"\n{'=' * 70}")
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    df = pd.read_csv(OUT_CSV)
    print(f"  Output: {OUT_CSV}")
    print(f"  Shape:  {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Clips:  {df['clip'].nunique()}")

    feat_cols = [c for c in df.columns if c not in META_COLS]
    nan_counts = df[feat_cols].isna().sum()
    if nan_counts.any():
        print("\n  NaN warnings:")
        for col in feat_cols:
            n = nan_counts[col]
            if n > 0:
                print(f"    {col}: {n} NaN values")
    else:
        print(f"  No NaN in any of {len(feat_cols)} feature columns.")


if __name__ == "__main__":
    main()
