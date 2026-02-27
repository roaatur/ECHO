#!/usr/bin/env python3
"""D²-City Step 2 — Extract 65 spatial features from video frames.

Ports the exact feature set from new_features/new_extract_features.py to
D²-City's video source (VideoCapture instead of image files).

Key differences from DETRAC:
  - Video input (cv2.VideoCapture) instead of image directories
  - Dynamic frame dimensions (1280×720 typical, must not hardcode)
  - SUBSAMPLE=3 from 25 fps → process MOG2 on every frame, extract
    features only on sampled frames (preserves background model quality)
  - No weather label (set to "unknown")
  - "sequence" column is named "clip_id" in detection CSV; renamed to
    "sequence" in output for compatibility with downstream scripts

Output: D2_all_features.csv
  Columns: clip_id, frame_idx, fn_nano, num_gt_coco, tp_nano, tp_medium,
           recoverable_fn, <65 feature columns ...>

Crash-safe: appends per clip, skips clips already written.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D2_common import SCRIPT_DIR, SUBSAMPLE, discover_clips

# ── Output ───────────────────────────────────────────────────────────────────
DET_CSV = SCRIPT_DIR / "D2_detection_results.csv"
OUT_CSV = SCRIPT_DIR / "D2_all_features.csv"

META_COLS = ["clip_id", "frame_idx", "fn_nano", "num_gt_coco",
             "tp_nano", "tp_medium", "recoverable_fn"]

# Reusable detectors (created once, shared across all clips)
_FAST_FULL = cv2.FastFeatureDetector_create(threshold=20)
_FAST_HALF = cv2.FastFeatureDetector_create(threshold=20)
_DARK_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

# Per-clip MOG2 background subtractors (reset between clips)
_MOG2 = {}

# Cached FFT frequency masks keyed by (height, width)
_FFT_MASKS = {}


def _get_fft_masks(h, w):
    key = (h, w)
    if key not in _FFT_MASKS:
        cy, cx    = h // 2, w // 2
        Y, X      = np.ogrid[:h, :w]
        dist_sq   = (X - cx) ** 2 + (Y - cy) ** 2
        r_nyquist = min(h, w) // 4
        r_half    = min(h, w) // 8
        _FFT_MASKS[key] = {
            "low_nyquist": dist_sq <= r_nyquist ** 2,
            "very_low":    dist_sq <= r_half ** 2,
        }
    return _FFT_MASKS[key]


# ═══════════════════════════════════════════════════════════════════
# CATEGORY A: Resolution-Sensitivity Features
# ═══════════════════════════════════════════════════════════════════

def extract_resolution_features(gray, gray_f, bgr):
    h, w  = gray.shape
    feats = {}

    # A1: FFT high-frequency energy ratio
    f      = np.fft.fft2(gray_f)
    fshift = np.fft.fftshift(f)
    mag    = np.abs(fshift) ** 2
    total  = mag.sum() + 1e-10
    masks  = _get_fft_masks(h, w)
    low    = mag[masks["low_nyquist"]].sum()
    feats["fft_high_freq_energy_ratio"] = float((total - low) / total)

    # A2: FFT critical band (between Nyquist/2 and Nyquist)
    very_low = mag[masks["very_low"]].sum()
    feats["fft_critical_band_ratio"] = float((low - very_low) / total)

    # A3 & A4: Laplacian variance and mean abs
    lap = cv2.Laplacian(gray_f, cv2.CV_64F)
    feats["laplacian_variance"]  = float(lap.var())
    feats["laplacian_mean_abs"]  = float(np.abs(lap).mean())

    # A5: Multi-scale edge density (fine / coarse)
    edges_fine   = cv2.Canny(gray, 50, 150)
    edges_coarse = cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 3.0), 50, 150)
    fine_d   = edges_fine.mean()   / 255.0
    coarse_d = edges_coarse.mean() / 255.0
    feats["edge_density_fine"]        = float(fine_d)
    feats["edge_density_coarse"]      = float(coarse_d)
    feats["edge_fine_coarse_ratio"]   = float(fine_d / (coarse_d + 1e-8))

    # A6: FAST keypoints at full resolution (density = count / pixels)
    kps_full = _FAST_FULL.detect(gray, None)
    n_full   = len(kps_full)
    feats["fast_keypoints_full"] = float(n_full / (h * w))

    # A7: FAST keypoints at half resolution (simulates nano@320 scale)
    gray_half  = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    kps_half   = _FAST_HALF.detect(gray_half, None)
    n_half     = len(kps_half)
    h2, w2     = gray_half.shape
    feats["fast_keypoints_half"] = float(n_half / (h2 * w2))

    # A8: Keypoint loss ratio (structure that survives 2× downsample)
    feats["keypoint_loss_ratio"] = float(1.0 - n_half / (n_full + 1e-8))

    # A9 & A10: Spatial frequency + gradient magnitude statistics
    gx      = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    gy      = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    grad    = np.sqrt(gx ** 2 + gy ** 2)
    feats["spatial_frequency"]       = float(np.sqrt(np.mean(gx ** 2) + np.mean(gy ** 2)))
    feats["gradient_magnitude_mean"] = float(grad.mean())
    feats["gradient_magnitude_std"]  = float(grad.std())

    # A11: DCT high-frequency energy (block-based, sampled)
    dct_high = dct_total = 0.0
    step = 32
    for by in range(0, h - 7, step):
        for bx in range(0, w - 7, step):
            block     = gray_f[by:by + 8, bx:bx + 8]
            dct_block = cv2.dct(block.astype(np.float32))
            e         = float(np.sum(dct_block ** 2))
            hi        = float(np.sum(dct_block[4:, :] ** 2) +
                              np.sum(dct_block[:4, 4:] ** 2))
            dct_high += hi
            dct_total += e
    feats["dct_high_freq_ratio"] = float(dct_high / (dct_total + 1e-10))

    # A12: Gabor energy at Nyquist-critical frequency (2 orientations)
    gray_gabor_f = cv2.resize(gray, (w // 2, h // 2),
                              interpolation=cv2.INTER_AREA).astype(np.float64)
    gabor_e = 0.0
    for theta in [0, np.pi / 2]:
        kern    = cv2.getGaborKernel((15, 15), sigma=2.5, theta=theta,
                                     lambd=4.0, gamma=0.5, psi=0)
        resp    = cv2.filter2D(gray_gabor_f, cv2.CV_64F, kern)
        gabor_e += np.mean(resp ** 2)
    feats["gabor_nyquist_energy"] = float(gabor_e / 2.0)

    # A13: SSIM between full-res and half-res upscaled (information loss)
    gray_half_up = cv2.resize(gray_half, (w, h),
                              interpolation=cv2.INTER_LINEAR).astype(np.float64)
    mu1, mu2 = gray_f.mean(), gray_half_up.mean()
    s1,  s2  = gray_f.std(),  gray_half_up.std()
    s12      = np.mean((gray_f - mu1) * (gray_half_up - mu2))
    C1, C2   = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_val = ((2*mu1*mu2 + C1) * (2*s12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2))
    feats["downsample_ssim"]       = float(ssim_val)
    feats["downsample_info_loss"]  = float(1.0 - ssim_val)

    return feats


# ═══════════════════════════════════════════════════════════════════
# CATEGORY B: Density Proxy Features
# ═══════════════════════════════════════════════════════════════════

def extract_density_features(gray, gray_f, bgr, clip_id, prev_gray):
    h, w  = gray.shape
    feats = {}
    total_px = h * w

    # B1: MOG2 foreground segmentation (per-clip stateful)
    # Note: for a moving dashcam, MOG2 captures relative motion against an
    # adapting background, not static scene structure. The model is still
    # maintained per-clip as a proxy for "objects moving relative to camera".
    if clip_id not in _MOG2:
        _MOG2[clip_id] = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=True)
    fg_mask    = _MOG2[clip_id].apply(bgr)
    fg_binary  = (fg_mask == 255).astype(np.uint8)
    shadow_bin = (fg_mask == 127).astype(np.uint8)
    feats["foreground_pixel_ratio"] = float(fg_binary.sum() / total_px)
    feats["shadow_pixel_ratio"]     = float(shadow_bin.sum() / total_px)

    # B2: Foreground connected components (blob count ≈ object count)
    kern    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_clean = cv2.morphologyEx(fg_binary, cv2.MORPH_CLOSE, kern)
    fg_clean = cv2.morphologyEx(fg_clean,  cv2.MORPH_OPEN,  kern)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg_clean, connectivity=8)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        valid = (areas >= 200) & (areas < 0.3 * total_px)
        feats["foreground_blob_count"]    = int(valid.sum())
        va = areas[valid] if valid.sum() > 0 else np.array([0])
        feats["foreground_blob_area_mean"] = float(va.mean()   / total_px)
        feats["foreground_blob_area_std"]  = float(va.std()    / total_px)
        if valid.sum() > 0:
            perims  = (stats[1:, cv2.CC_STAT_WIDTH] * 2 +
                       stats[1:, cv2.CC_STAT_HEIGHT] * 2).astype(float)
            valid_pa = perims[valid] / (areas[valid] + 1e-8)
            feats["foreground_blob_pa_ratio"] = float(valid_pa.mean())
        else:
            feats["foreground_blob_pa_ratio"] = 0.0
    else:
        feats["foreground_blob_count"]     = 0
        feats["foreground_blob_area_mean"] = 0.0
        feats["foreground_blob_area_std"]  = 0.0
        feats["foreground_blob_pa_ratio"]  = 0.0

    # B3: Foreground edge density (edges within FG mask only)
    edges    = cv2.Canny(gray, 50, 150)
    fg_edges = cv2.bitwise_and(edges, edges, mask=fg_clean)
    feats["foreground_edge_density"] = float(fg_edges.sum() / (255.0 * total_px))

    # B4: Frame differencing (between consecutive SAMPLED frames)
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        feats["temporal_diff_mean"]  = float(diff.mean())
        feats["temporal_diff_std"]   = float(diff.astype(np.float64).std())
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        feats["motion_pixel_ratio"]  = float(motion_mask.sum() / (255.0 * total_px))
    else:
        feats["temporal_diff_mean"]  = 0.0
        feats["temporal_diff_std"]   = 0.0
        feats["motion_pixel_ratio"]  = 0.0

    return feats


# ═══════════════════════════════════════════════════════════════════
# CATEGORY C: Scene Complexity / Texture Features
# ═══════════════════════════════════════════════════════════════════

def extract_complexity_features(gray, gray_f, bgr):
    h, w  = gray.shape
    feats = {}

    # C1: Image entropy
    hist      = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-10)
    feats["image_entropy"] = float(scipy_entropy(hist_norm, base=2))

    # C2: GLCM texture (simplified, downsampled for speed)
    gray_ds = cv2.resize(gray, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    gray_q  = (gray_ds // 8).astype(np.int32)
    levels  = 32
    glcm    = np.zeros((levels, levels), dtype=np.float64)
    l, r    = gray_q[:, :-1].ravel(), gray_q[:, 1:].ravel()
    np.add.at(glcm, (l, r), 1); np.add.at(glcm, (r, l), 1)
    t, b    = gray_q[:-1, :].ravel(), gray_q[1:, :].ravel()
    np.add.at(glcm, (t, b), 1); np.add.at(glcm, (b, t), 1)
    glcm_n  = glcm / (glcm.sum() + 1e-10)
    ii, jj  = np.meshgrid(range(levels), range(levels), indexing="ij")
    feats["glcm_contrast"]    = float(np.sum(glcm_n * (ii - jj) ** 2))
    feats["glcm_homogeneity"] = float(np.sum(glcm_n / (1 + np.abs(ii - jj))))
    feats["glcm_energy"]      = float(np.sum(glcm_n ** 2))
    flat = glcm_n.flatten(); flat = flat[flat > 0]
    feats["glcm_entropy"]     = float(-np.sum(flat * np.log2(flat)))

    # C3: Colorfulness (Hasler & Süsstrunk 2003)
    B, G, R = (bgr[:, :, i].astype(float) for i in range(3))
    rg, yb  = R - G, 0.5 * (R + G) - B
    feats["colorfulness"] = float(
        np.sqrt(rg.std()**2 + yb.std()**2) +
        0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2))

    # C4: Brightness and contrast
    feats["mean_brightness"] = float(gray.mean())
    feats["brightness_std"]  = float(gray_f.std())

    # C5: Dark channel mean (haze proxy)
    min_ch = np.min(bgr, axis=2)
    feats["dark_channel_mean"] = float(cv2.erode(min_ch, _DARK_KERNEL).mean())

    # C6: Edge component statistics
    edges = cv2.Canny(gray, 50, 150)
    n_lab, _, stats_e, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    if n_lab > 1:
        areas_e = stats_e[1:, cv2.CC_STAT_AREA]
        feats["small_edge_components"] = int(np.sum(areas_e < 500))
        feats["total_edge_components"] = int(n_lab - 1)
    else:
        feats["small_edge_components"] = 0
        feats["total_edge_components"] = 0

    # C7: Overall edge density
    feats["overall_edge_density"] = float(edges.mean() / 255.0)

    # C8: RMS contrast
    feats["rms_contrast"] = float(np.sqrt(np.mean((gray_f - gray_f.mean()) ** 2)))

    # C9: Saturation statistics
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(float)
    feats["saturation_mean"] = float(sat.mean())
    feats["saturation_std"]  = float(sat.std())

    return feats


# ═══════════════════════════════════════════════════════════════════
# CATEGORY D: Per-Strip Features (spatial distribution)
# ═══════════════════════════════════════════════════════════════════

def extract_strip_features(gray):
    """Divide frame into top / mid / bot horizontal thirds."""
    h, w     = gray.shape
    strip_h  = h // 3         # dynamic — works for 720p and 1080p
    strips   = {
        "top": gray[0          :strip_h,      :],
        "mid": gray[strip_h    :2 * strip_h,  :],
        "bot": gray[2 * strip_h:h,            :],
    }
    feats      = {}
    strip_vals = {}

    for prefix, strip in strips.items():
        sf   = strip.astype(np.float64)
        sh, sw = strip.shape

        lap_var  = float(cv2.Laplacian(sf, cv2.CV_64F).var())
        gx_s     = cv2.Sobel(sf, cv2.CV_64F, 1, 0, ksize=3)
        gy_s     = cv2.Sobel(sf, cv2.CV_64F, 0, 1, ksize=3)
        grad_std = float(np.sqrt(gx_s**2 + gy_s**2).std())
        edges_s  = cv2.Canny(strip, 50, 150)
        ed       = float(edges_s.mean() / 255.0)

        fshift_s = np.fft.fftshift(np.fft.fft2(sf))
        mag_s    = np.abs(fshift_s) ** 2
        tot_e    = mag_s.sum() + 1e-10
        m_s      = _get_fft_masks(sh, sw)
        hi_e     = float((tot_e - mag_s[m_s["very_low"]].sum()) / tot_e)

        feats[f"{prefix}_laplacian_var"]    = lap_var
        feats[f"{prefix}_gradient_std"]     = grad_std
        feats[f"{prefix}_edge_density"]     = ed
        feats[f"{prefix}_high_freq_energy"] = hi_e
        strip_vals[prefix] = {"edge_density": ed, "gradient_std": grad_std,
                               "laplacian_var": lap_var}

    for metric in ["edge_density", "gradient_std", "laplacian_var"]:
        top_v = strip_vals["top"][metric]
        bot_v = strip_vals["bot"][metric]
        feats[f"ratio_top_bot_{metric}"] = float(top_v / (bot_v + 1e-8))

    return feats


# ═══════════════════════════════════════════════════════════════════
# CATEGORY E: BRISQUE-like (MSCN)
# ═══════════════════════════════════════════════════════════════════

def extract_brisque_features(gray_f):
    feats   = {}
    ks      = 7
    mu      = cv2.GaussianBlur(gray_f, (ks, ks), 1.166)
    mu_sq   = cv2.GaussianBlur(gray_f ** 2, (ks, ks), 1.166)
    sigma   = np.sqrt(np.abs(mu_sq - mu ** 2))
    mscn    = (gray_f - mu) / (sigma + 1.0)

    feats["mscn_mean"]     = float(mscn.mean())
    feats["mscn_std"]      = float(mscn.std())
    feats["mscn_kurtosis"] = float(
        np.mean((mscn - mscn.mean())**4) / (mscn.std()**4 + 1e-10) - 3.0)
    feats["mscn_skewness"] = float(
        np.mean((mscn - mscn.mean())**3) / (mscn.std()**3 + 1e-10))

    h_pairs = mscn[:, :-1] * mscn[:, 1:]
    feats["mscn_h_pair_mean"] = float(h_pairs.mean())
    feats["mscn_h_pair_std"]  = float(h_pairs.std())

    v_pairs = mscn[:-1, :] * mscn[1:, :]
    feats["mscn_v_pair_mean"] = float(v_pairs.mean())
    feats["mscn_v_pair_std"]  = float(v_pairs.std())

    return feats


# ═══════════════════════════════════════════════════════════════════
# Master extraction function
# ═══════════════════════════════════════════════════════════════════

def extract_all_features(bgr, prev_gray, clip_id):
    """Extract all 65 features from one BGR frame.

    Returns (feature_dict, gray).
    """
    gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float64)

    feats = {}
    feats.update(extract_resolution_features(gray, gray_f, bgr))
    feats.update(extract_density_features(gray, gray_f, bgr, clip_id, prev_gray))
    feats.update(extract_complexity_features(gray, gray_f, bgr))
    feats.update(extract_strip_features(gray))
    feats.update(extract_brisque_features(gray_f))
    return feats, gray


# ═══════════════════════════════════════════════════════════════════
# Main extraction loop
# ═══════════════════════════════════════════════════════════════════

def run_extraction():
    print("=" * 70)
    print("D²-CITY FEATURE EXTRACTION — 65-FEATURE SET")
    print("=" * 70)

    if not DET_CSV.exists():
        print(f"\n  ERROR: {DET_CSV} not found.")
        print("  Run D2_evaluate.py first.")
        return

    det_df = pd.read_csv(DET_CSV)
    print(f"\n  Detection results: {len(det_df):,} rows, "
          f"{det_df['clip_id'].nunique()} clips")

    # Determine feature column order from a test extraction
    test_img   = np.zeros((720, 1280, 3), dtype=np.uint8)
    test_feats, _ = extract_all_features(test_img, None, "__test__")
    del _MOG2["__test__"]
    FEATURE_COLS = sorted(test_feats.keys())
    COL_ORDER    = META_COLS + FEATURE_COLS
    print(f"  Features per frame: {len(FEATURE_COLS)}")

    # Crash-safety: which clips are already done?
    done_clips = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV, nrows=0)
        if list(existing.columns) == COL_ORDER:
            done_clips = set(pd.read_csv(OUT_CSV)["clip_id"].unique())
            print(f"  Resuming — {len(done_clips)} clips already done\n")
        else:
            print("  WARNING: Column mismatch in existing CSV. Starting fresh.")
            OUT_CSV.unlink()

    clips    = discover_clips()
    todo     = [c for c in clips if c["clip_id"] not in done_clips]
    n_total  = det_df["clip_id"].nunique()
    print(f"  Clips to process: {len(todo)} / {n_total}")

    # Group detection results by clip for fast lookup
    det_grouped = det_df.groupby("clip_id")

    times = []
    done_count = len(done_clips)

    for clip_info in todo:
        clip_id  = clip_info["clip_id"]
        vid_path = clip_info["video_path"]

        if clip_id not in det_grouped.groups:
            print(f"  [SKIP] {clip_id}: not in detection CSV")
            continue

        clip_det = det_grouped.get_group(clip_id).set_index("frame_idx")
        n_frames = len(clip_det)

        done_count += 1
        print(f"  [{done_count:3d}/{n_total}] {clip_id}: "
              f"{n_frames} sampled frames ...", end="", flush=True)

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"  [SKIP] cannot open video")
            continue

        t0        = time.time()
        prev_gray = None
        records   = []
        frame_abs = 0   # original video frame index

        # Reset MOG2 for this clip
        if clip_id in _MOG2:
            del _MOG2[clip_id]

        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            if frame_abs % SUBSAMPLE != 0:
                # Still apply MOG2 on non-sampled frames to keep BG model fresh
                gray_tmp = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                if clip_id in _MOG2:
                    _MOG2[clip_id].apply(bgr)
                frame_abs += 1
                continue

            sampled_idx = frame_abs // SUBSAMPLE

            # Only store features for frames that appear in detection CSV
            if sampled_idx not in clip_det.index:
                frame_abs += 1
                continue

            det_row = clip_det.loc[sampled_idx]

            feats, gray = extract_all_features(bgr, prev_gray, clip_id)
            prev_gray   = gray

            feats["clip_id"]        = clip_id
            feats["frame_idx"]      = sampled_idx
            feats["fn_nano"]        = int(det_row["fn_nano"])
            feats["num_gt_coco"]    = int(det_row["num_gt_coco"])
            feats["tp_nano"]        = int(det_row["tp_nano"])
            feats["tp_medium"]      = int(det_row["tp_medium"])
            feats["recoverable_fn"] = int(det_row["recoverable_fn"])
            records.append(feats)

            frame_abs += 1

        cap.release()
        if clip_id in _MOG2:
            del _MOG2[clip_id]

        if not records:
            print(f"  [SKIP] no matching frames")
            continue

        # Append to CSV
        seq_out    = pd.DataFrame(records)[COL_ORDER]
        write_hdr  = not OUT_CSV.exists()
        seq_out.to_csv(OUT_CSV, mode="a", index=False, header=write_hdr)

        elapsed = time.time() - t0
        fps     = len(records) / elapsed if elapsed > 0 else 0
        times.append(elapsed / max(len(records), 1))
        avg_t   = sum(times) / len(times)
        remaining = len(todo) - done_count + len(done_clips)
        eta     = remaining * avg_t * n_frames / 60

        print(f"  {len(records)} rows, {elapsed:.1f}s ({fps:.0f} fps) "
              f"[ETA {eta:.1f} min]", flush=True)

    print(f"\n{'=' * 70}")
    print("FEATURE EXTRACTION COMPLETE")
    print(f"{'=' * 70}")

    df = pd.read_csv(OUT_CSV)
    print(f"  Output: {OUT_CSV}")
    print(f"  Shape:  {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Clips:  {df['clip_id'].nunique()}")

    feat_cols  = [c for c in df.columns if c not in META_COLS]
    nan_counts = df[feat_cols].isna().sum()
    if nan_counts.any():
        print("\n  NaN warnings:")
        for col in feat_cols:
            n = nan_counts[col]
            if n > 0:
                print(f"    {col}: {n}")
    else:
        print(f"  No NaN values in any of {len(feat_cols)} feature columns.")

    return df


if __name__ == "__main__":
    run_extraction()
