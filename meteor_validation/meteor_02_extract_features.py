#!/usr/bin/env python3
"""Extract 41 spatial features (40 features, D5 gives 2 values) on METEOR clips.

All features are computed at 640px width (matching YOLO11m input resolution).
7 categories: A (Edge/Texture), B (Luminance), C (Color), D (Info),
              E (Spatial), F (Temporal), G (Scale/Resolution).

Merges with detection results from meteor_01_evaluate.py to include
tp_nano, tp_medium, recoverable_fn, fn_coco, fn_total, num_gt per frame.

Output: meteor_validation/meteor_spatial_features.csv
Crash-safe: appends per clip, resumes by skipping completed clips.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from meteor_common import (
    SCRIPT_DIR,
    SUBSAMPLE,
    discover_clips,
    read_video_frames,
)

# ── Paths ──────────────────────────────────────────────────────────
DET_CSV = SCRIPT_DIR / "meteor_detection_results.csv"
OUT_CSV = SCRIPT_DIR / "meteor_spatial_features.csv"

# ── Target resolution for feature extraction ──────────────────────
TARGET_W = 640

# ── Column definitions ────────────────────────────────────────────
FEATURE_COLS = [
    # A: Edge & Texture (8)
    "edge_density", "gradient_mag_mean", "gradient_mag_std", "laplacian_var",
    "highfreq_energy_ratio", "lbp_entropy", "gabor_energy", "gradient_hist_entropy",
    # B: Luminance & Visibility (7)
    "brightness_mean", "brightness_std", "dark_channel", "pct_dark_pixels",
    "pct_bright_pixels", "local_contrast_rms", "luminance_range",
    # C: Color & Saturation (4)
    "saturation_mean", "saturation_std", "colorfulness", "hue_entropy",
    # D: Spatial Frequency & Info (6 values)
    "shannon_entropy", "entropy_2d", "spatial_frequency", "average_gradient",
    "mscn_mean", "mscn_std",
    # E: Spatial Structure & Density (6)
    "connected_components", "small_component_count", "small_component_ratio",
    "top_edge_density", "middle_edge_density", "bottom_edge_density",
    # F: Temporal / Motion (5)
    "flow_magnitude_mean", "flow_magnitude_std", "flow_direction_entropy",
    "temporal_diff_magnitude", "temporal_diff_ratio",
    # G: Scale & Resolution Sensitivity (5)
    "fast_keypoints_full", "fast_keypoints_320", "keypoint_loss_ratio",
    "multiscale_edge_ratio", "small_edge_blob_count",
]

META_COLS = [
    "clip", "frame_id", "weather", "road_type",
    "tp_nano", "tp_medium", "recoverable_fn",
    "fn_coco_nano", "fn_coco_medium",
    "fn_total_nano", "fn_total_medium",
    "num_gt", "num_gt_coco",
]
COL_ORDER = META_COLS + FEATURE_COLS

# Reusable OpenCV objects
FAST_DETECTOR = cv2.FastFeatureDetector_create()


# ── Category A: Edge & Texture ────────────────────────────────────

def extract_edge_texture(gray, gray_f, edges, gx, gy, mag):
    """A1-A8: Edge and texture features."""
    features = {}

    # A1: Edge density (Canny)
    features["edge_density"] = float(np.mean(edges > 0))

    # A2-A3: Gradient statistics
    features["gradient_mag_mean"] = float(np.mean(mag))
    features["gradient_mag_std"] = float(np.std(mag))

    # A4: Laplacian variance (sharpness)
    features["laplacian_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # A5: High-frequency energy ratio (DCT on 256x256 center crop)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    half = 128
    # Clamp crop to available size
    y1 = max(cy - half, 0)
    y2 = min(cy + half, h)
    x1 = max(cx - half, 0)
    x2 = min(cx + half, w)
    crop = gray_f[y1:y2, x1:x2]
    # Ensure even dimensions for DCT
    ch, cw = crop.shape
    crop = crop[:ch - ch % 2, :cw - cw % 2]
    if crop.size > 0:
        dct = cv2.dct(crop.astype(np.float32))
        total_energy = np.sum(dct ** 2)
        hh, hw = dct.shape[0] // 2, dct.shape[1] // 2
        hf_energy = np.sum(dct[hh:, hw:] ** 2)
        features["highfreq_energy_ratio"] = float(hf_energy / max(total_energy, 1e-10))
    else:
        features["highfreq_energy_ratio"] = 0.0

    # A6: LBP entropy (np.roll-based)
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for i, (dy, dx) in enumerate(
        [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    ):
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        lbp |= ((gray >= shifted).astype(np.uint8) << i)
    hist = np.histogram(lbp, bins=256, range=(0, 256))[0].astype(np.float64)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    features["lbp_entropy"] = float(-np.sum(hist * np.log2(hist)))

    # A7: Gabor energy (4 orientations)
    gabor_energy = 0.0
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0)
        filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
        gabor_energy += np.mean(np.abs(filtered))
    features["gabor_energy"] = float(gabor_energy / 4.0)

    # A8: Gradient histogram entropy (orientation diversity)
    angle = np.arctan2(gy, gx)
    mask = mag > np.percentile(mag, 50)
    if mask.sum() > 0:
        hist_angle = np.histogram(angle[mask], bins=36, range=(-np.pi, np.pi))[0].astype(np.float64)
        hist_angle = hist_angle / hist_angle.sum()
        hist_angle = hist_angle[hist_angle > 0]
        features["gradient_hist_entropy"] = float(-np.sum(hist_angle * np.log2(hist_angle)))
    else:
        features["gradient_hist_entropy"] = 0.0

    return features


# ── Category B: Luminance & Visibility ────────────────────────────

def extract_luminance(gray, frame_bgr):
    """B1-B7: Luminance and visibility features."""
    features = {}

    # B1-B2: Brightness statistics
    features["brightness_mean"] = float(np.mean(gray))
    features["brightness_std"] = float(np.std(gray))

    # B3: Dark channel prior (fog/haze proxy)
    min_channel = np.min(frame_bgr, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_channel, kernel)
    features["dark_channel"] = float(np.mean(dark_channel))

    # B4-B5: Exposure extremes
    features["pct_dark_pixels"] = float(np.mean(gray < 40))
    features["pct_bright_pixels"] = float(np.mean(gray > 220))

    # B6: Local contrast (RMS)
    mean_val = float(np.mean(gray))
    features["local_contrast_rms"] = float(np.std(gray)) / max(mean_val, 1e-10)

    # B7: Luminance range (robust, excluding 1% outliers)
    p1, p99 = np.percentile(gray, [1, 99])
    features["luminance_range"] = float(p99 - p1)

    return features


# ── Category C: Color & Saturation ────────────────────────────────

def extract_color(frame_bgr, hsv):
    """C1-C4: Color and saturation features."""
    features = {}

    # C1-C2: Saturation statistics
    features["saturation_mean"] = float(np.mean(hsv[:, :, 1]))
    features["saturation_std"] = float(np.std(hsv[:, :, 1]))

    # C3: Colorfulness (Hasler & Susstrunk)
    B = frame_bgr[:, :, 0].astype(np.float64)
    G = frame_bgr[:, :, 1].astype(np.float64)
    R = frame_bgr[:, :, 2].astype(np.float64)
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    features["colorfulness"] = float(
        np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)
    )

    # C4: Hue entropy
    hue = hsv[:, :, 0]
    hist_h = np.histogram(hue, bins=36, range=(0, 180))[0].astype(np.float64)
    hist_h = hist_h / hist_h.sum()
    hist_h = hist_h[hist_h > 0]
    features["hue_entropy"] = float(-np.sum(hist_h * np.log2(hist_h)))

    return features


# ── Category D: Spatial Frequency & Information ───────────────────

def extract_info(gray, gray_f):
    """D1-D5: Information and frequency features (6 values)."""
    features = {}

    # D1: Shannon entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    features["shannon_entropy"] = float(-np.sum(hist * np.log2(hist)))

    # D2: 2D entropy (pixel + right neighbor joint histogram)
    g1 = gray[:, :-1].flatten()
    g2 = gray[:, 1:].flatten()
    hist2d = np.histogram2d(g1, g2, bins=64, range=[[0, 256], [0, 256]])[0]
    hist2d = hist2d / hist2d.sum()
    hist2d = hist2d[hist2d > 0]
    features["entropy_2d"] = float(-np.sum(hist2d * np.log2(hist2d)))

    # D3: Spatial frequency
    rf = np.sqrt(np.mean(np.diff(gray_f, axis=1) ** 2))
    cf = np.sqrt(np.mean(np.diff(gray_f, axis=0) ** 2))
    features["spatial_frequency"] = float(np.sqrt(rf ** 2 + cf ** 2))

    # D4: Average gradient
    gx_d = np.abs(np.diff(gray_f, axis=1))
    gy_d = np.abs(np.diff(gray_f, axis=0))
    features["average_gradient"] = float((np.mean(gx_d) + np.mean(gy_d)) / 2.0)

    # D5: BRISQUE-like features (MSCN coefficients)
    C = 1.0
    mu = cv2.GaussianBlur(gray_f, (7, 7), 7 / 6)
    sigma = cv2.GaussianBlur((gray_f - mu) ** 2, (7, 7), 7 / 6)
    sigma = np.sqrt(sigma) + C
    mscn = (gray_f - mu) / sigma
    features["mscn_mean"] = float(np.mean(mscn))
    features["mscn_std"] = float(np.std(mscn))

    return features


# ── Category E: Spatial Structure & Density ───────────────────────

def extract_spatial(gray, edges):
    """E1-E6: Spatial structure and density features."""
    features = {}
    h, w = gray.shape

    # E1-E3: Connected components
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    features["connected_components"] = num_labels

    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
        small_threshold = 0.01 * h * w
        small_count = int(np.sum(areas < small_threshold))
    else:
        small_count = 0
    features["small_component_count"] = small_count
    features["small_component_ratio"] = float(small_count / max(num_labels - 1, 1))

    # E4-E6: Strip edge densities (top/middle/bottom thirds)
    strip_h = h // 3
    for i, name in enumerate(["top", "middle", "bottom"]):
        strip = gray[i * strip_h:(i + 1) * strip_h, :]
        strip_edges = cv2.Canny(strip, 50, 150)
        features[f"{name}_edge_density"] = float(np.mean(strip_edges > 0))

    return features


# ── Category F: Temporal / Motion ─────────────────────────────────

def extract_temporal(gray_curr, gray_prev):
    """F1-F5: Temporal and motion features.

    gray_prev=None for first frame of each clip -> all zeros.
    Both inputs are at 640px resolution.
    """
    features = {}

    if gray_prev is None:
        features["flow_magnitude_mean"] = 0.0
        features["flow_magnitude_std"] = 0.0
        features["flow_direction_entropy"] = 0.0
        features["temporal_diff_magnitude"] = 0.0
        features["temporal_diff_ratio"] = 0.0
        return features

    # F4-F5: Temporal difference (cheap, at 640px)
    diff = cv2.absdiff(gray_curr, gray_prev)
    features["temporal_diff_magnitude"] = float(np.mean(diff))
    features["temporal_diff_ratio"] = float(np.mean(diff > 25))

    # F1-F3: Optical flow (downsample to 320x180 for speed)
    h, w = gray_curr.shape
    small_w, small_h = 320, int(h * 320 / w)
    small_curr = cv2.resize(gray_curr, (small_w, small_h))
    small_prev = cv2.resize(gray_prev, (small_w, small_h))
    flow = cv2.calcOpticalFlowFarneback(
        small_prev, small_curr, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    features["flow_magnitude_mean"] = float(np.mean(flow_mag))
    features["flow_magnitude_std"] = float(np.std(flow_mag))

    # Flow direction entropy
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    mask = flow_mag > 1.0
    if mask.sum() > 100:
        hist_dir = np.histogram(angle[mask], bins=36, range=(-np.pi, np.pi))[0].astype(np.float64)
        hist_dir = hist_dir / hist_dir.sum()
        hist_dir = hist_dir[hist_dir > 0]
        features["flow_direction_entropy"] = float(-np.sum(hist_dir * np.log2(hist_dir)))
    else:
        features["flow_direction_entropy"] = 0.0

    return features


# ── Category G: Scale & Resolution Sensitivity ───────────────────

def extract_scale(gray, edges):
    """G1-G5: Scale and resolution sensitivity features.

    Input gray is at 640px. Measures what gets lost going to 320px.
    """
    features = {}
    h, w = gray.shape

    # G1: FAST keypoint count at 640px
    kp_full = FAST_DETECTOR.detect(gray, None)
    features["fast_keypoints_full"] = len(kp_full)

    # G2: FAST keypoint count at 320px (nano resolution)
    small_w = 320
    small_h = int(h * 320 / w)
    small = cv2.resize(gray, (small_w, small_h))
    kp_small = FAST_DETECTOR.detect(small, None)
    features["fast_keypoints_320"] = len(kp_small)

    # G3: Keypoint loss ratio (640->320)
    features["keypoint_loss_ratio"] = float(1.0 - len(kp_small) / max(len(kp_full), 1))

    # G4: Multi-scale edge ratio (640px / 320px edge density)
    ed_full = float(np.mean(edges > 0))
    edges_small = cv2.Canny(small, 50, 150)
    ed_small = float(np.mean(edges_small > 0))
    features["multiscale_edge_ratio"] = ed_full / max(ed_small, 1e-10)

    # G5: Small edge blob count (edge connected components < 16px^2)
    num_labels_e, _, stats_e, _ = cv2.connectedComponentsWithStats(edges)
    if num_labels_e > 1:
        features["small_edge_blob_count"] = int(np.sum(stats_e[1:, cv2.CC_STAT_AREA] < 16))
    else:
        features["small_edge_blob_count"] = 0

    return features


# ── Main dispatcher ───────────────────────────────────────────────

def extract_all_features(frame_bgr, prev_gray):
    """Extract all 41 feature values from one BGR frame.

    Frame is resized to 640px wide before extraction.

    Args:
        frame_bgr: numpy array (H, W, 3) BGR at original resolution
        prev_gray: previous frame's grayscale at 640px (None for first frame)

    Returns:
        (features_dict, current_gray) -- gray kept for next frame's temporal features.
    """
    # Resize to 640px wide, maintaining aspect ratio
    oh, ow = frame_bgr.shape[:2]
    target_h = int(oh * TARGET_W / ow)
    frame_bgr = cv2.resize(frame_bgr, (TARGET_W, target_h))

    # Shared intermediates (computed once)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float64)
    edges = cv2.Canny(gray, 50, 150)
    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Extract all categories
    features = {}
    features.update(extract_edge_texture(gray, gray_f, edges, gx, gy, mag))
    features.update(extract_luminance(gray, frame_bgr))
    features.update(extract_color(frame_bgr, hsv))
    features.update(extract_info(gray, gray_f))
    features.update(extract_spatial(gray, edges))
    features.update(extract_temporal(gray, prev_gray))
    features.update(extract_scale(gray, edges))

    return features, gray


# ── Main extraction loop ──────────────────────────────────────────

def main():
    t0 = time.time()

    # ── Load detection results ─────────────────────────────────────
    if not DET_CSV.exists():
        print(f"ERROR: detection results not found at {DET_CSV}", flush=True)
        print("Run meteor_01_evaluate.py first.", flush=True)
        return

    print("Loading detection results...", flush=True)
    det_df = pd.read_csv(DET_CSV)

    # Pivot to get nano and medium metrics side-by-side per frame
    nano_df = det_df[det_df["model"] == "nano_320"].copy()
    medium_df = det_df[det_df["model"] == "medium_640"].copy()

    nano_df = nano_df.rename(columns={
        "tp": "tp_nano",
        "fn_coco": "fn_coco_nano",
        "fn_total": "fn_total_nano",
    })[["clip", "frame_id", "tp_nano", "fn_coco_nano", "fn_total_nano",
        "num_gt", "num_gt_coco", "weather", "road_type"]]

    medium_df = medium_df.rename(columns={
        "tp": "tp_medium",
        "fn_coco": "fn_coco_medium",
        "fn_total": "fn_total_medium",
    })[["clip", "frame_id", "tp_medium", "fn_coco_medium", "fn_total_medium"]]

    merged_det = nano_df.merge(medium_df, on=["clip", "frame_id"], how="inner")
    merged_det["recoverable_fn"] = (merged_det["tp_medium"] - merged_det["tp_nano"]).clip(lower=0)
    det_by_clip = merged_det.groupby("clip")

    all_clips_with_det = set(merged_det["clip"].unique())
    print(f"  {len(merged_det)} frames across {len(all_clips_with_det)} clips", flush=True)

    # recoverable_fn distribution preview
    rfn = merged_det["recoverable_fn"]
    print(f"  recoverable_fn: mean={rfn.mean():.2f}, std={rfn.std():.2f}, "
          f"median={rfn.median():.0f}, max={rfn.max():.0f}, "
          f"zero_pct={100 * (rfn == 0).mean():.1f}%\n", flush=True)

    # ── Discover clips ─────────────────────────────────────────────
    clips = discover_clips(max_clips=50)
    if not clips:
        return

    clips = [c for c in clips if c["name"] in all_clips_with_det]
    n_clips = len(clips)
    print(f"Processing {n_clips} clips with detection results\n", flush=True)

    # ── Crash-safe resume ──────────────────────────────────────────
    done_clips = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV, nrows=0)
        # Check column compatibility
        if set(COL_ORDER).issubset(set(existing.columns)):
            existing_full = pd.read_csv(OUT_CSV)
            done_clips = set(existing_full["clip"].unique())
            print(f"Resuming: {len(done_clips)} clips already extracted\n", flush=True)
        else:
            print("WARNING: existing CSV has incompatible columns, starting fresh.\n",
                  flush=True)
            OUT_CSV.unlink()

    total_frames = len(merged_det)
    frames_done = 0
    times = []

    for clip_idx, clip_info in enumerate(clips):
        clip_name = clip_info["name"]

        if clip_name in done_clips:
            # Count frames for ETA
            if clip_name in all_clips_with_det:
                frames_done += len(det_by_clip.get_group(clip_name))
            print(f"  Skipping {clip_name} -- already extracted", flush=True)
            continue

        video_path = clip_info["video"]
        if video_path is None or not video_path.exists():
            print(f"  Skipping {clip_name} -- no video", flush=True)
            continue

        if clip_name not in all_clips_with_det:
            continue
        clip_det = det_by_clip.get_group(clip_name)
        det_lookup = clip_det.set_index("frame_id").to_dict("index")

        n_frames_expected = len(clip_det)
        weather = clip_info["weather"]
        road_type = clip_info["road_type"]

        print(f"  [{clip_idx + 1:3d}/{n_clips}] {clip_name} ({weather}): "
              f"{n_frames_expected} frames ...", end="", flush=True)
        clip_t0 = time.time()

        prev_gray = None
        records = []

        for frame_id, frame_bgr in read_video_frames(video_path, SUBSAMPLE):
            feats, prev_gray = extract_all_features(frame_bgr, prev_gray)

            det_row = det_lookup.get(frame_id)
            if det_row is not None:
                feats["clip"] = clip_name
                feats["frame_id"] = frame_id
                feats["weather"] = weather
                feats["road_type"] = road_type
                feats["tp_nano"] = int(det_row["tp_nano"])
                feats["tp_medium"] = int(det_row["tp_medium"])
                feats["recoverable_fn"] = int(det_row["recoverable_fn"])
                feats["fn_coco_nano"] = int(det_row["fn_coco_nano"])
                feats["fn_coco_medium"] = int(det_row["fn_coco_medium"])
                feats["fn_total_nano"] = int(det_row["fn_total_nano"])
                feats["fn_total_medium"] = int(det_row["fn_total_medium"])
                feats["num_gt"] = int(det_row["num_gt"])
                feats["num_gt_coco"] = int(det_row["num_gt_coco"])
                records.append(feats)

        # Append this clip to CSV
        if records:
            clip_out = pd.DataFrame(records)[COL_ORDER]
            write_header = not OUT_CSV.exists()
            clip_out.to_csv(OUT_CSV, mode="a", index=False, header=write_header)

        elapsed = time.time() - clip_t0
        fps = len(records) / elapsed if elapsed > 0 else 0
        ms_per_frame = (elapsed / max(len(records), 1)) * 1000
        times.append(elapsed / max(len(records), 1))
        frames_done += len(records)

        # ETA
        avg_per_frame = sum(times) / len(times) if times else 0
        remaining = total_frames - frames_done
        eta_min = remaining * avg_per_frame / 60

        print(f"  {elapsed:.1f}s ({fps:.0f} fps, {ms_per_frame:.0f}ms/f)  "
              f"[{frames_done:,}/{total_frames:,} frames, ETA {eta_min:.1f}min]",
              flush=True)

    # ── Validation ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}", flush=True)
    print(f"Feature extraction complete. Total time: {time.time() - t0:.1f}s\n",
          flush=True)

    if OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV)
        print(f"Output: {len(df):,} rows x {len(df.columns)} cols -> {OUT_CSV}\n",
              flush=True)

        # Feature ranges
        print(f"{'Feature':<30s} {'Min':>10s} {'Max':>10s} {'Mean':>10s}")
        print("-" * 60)
        for col in FEATURE_COLS:
            if col in df.columns:
                lo, hi, mu = df[col].min(), df[col].max(), df[col].mean()
                print(f"{col:<30s} {lo:10.4f} {hi:10.4f} {mu:10.4f}")

        # NaN check
        feat_cols_present = [c for c in FEATURE_COLS if c in df.columns]
        nan_counts = df[feat_cols_present].isna().sum()
        if nan_counts.any():
            print("\nWARNING: NaN values found:")
            for col in feat_cols_present:
                n = nan_counts[col]
                if n > 0:
                    print(f"  {col}: {n}")
        else:
            print("\nNo NaN values in features.")

        # recoverable_fn summary
        print(f"\nrecoverable_fn distribution:")
        rfn = df["recoverable_fn"]
        print(f"  mean={rfn.mean():.2f}, std={rfn.std():.2f}")
        print(f"  percentiles: 25th={rfn.quantile(0.25):.0f}, "
              f"50th={rfn.quantile(0.5):.0f}, "
              f"75th={rfn.quantile(0.75):.0f}, "
              f"90th={rfn.quantile(0.9):.0f}, "
              f"95th={rfn.quantile(0.95):.0f}")
        print(f"  zero_pct={100 * (rfn == 0).mean():.1f}%")


if __name__ == "__main__":
    main()
