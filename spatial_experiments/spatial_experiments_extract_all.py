#!/usr/bin/env python3
"""Extract 15 selected spatial features on ALL 100 DETRAC sequences (~138K frames).

Features were selected by spatial_experiments_feature_selection.py based on
Spearman correlation with nano@320 miss count (fn).  This script computes
only the 15 winners — 3 per strip × 3 strips + 6 global — and writes a
single CSV ready for LSTM training.

Output: spatial_experiments_all_features.csv  (~138K rows × 20 cols)
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(ROOT / "res_experiment"))
from res_evaluate_detrac import IMG_BASE          # noqa: E402

DETRAC_CSV = ROOT / "res_experiment" / "detrac_results.csv"
OUT_CSV = SCRIPT_DIR / "spatial_experiments_all_features.csv"

FRAME_H = 540
STRIP_H = FRAME_H // 3          # 180 rows per strip

# Reusable OpenCV objects (same params as feature_selection.py)
DARK_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
FAST_DETECTOR = cv2.FastFeatureDetector_create(threshold=20)

# The 15 selected feature names (for column ordering)
FEATURE_COLS = [
    "top_laplacian_variance", "top_small_component_count", "top_gradient_magnitude_std",
    "mid_laplacian_variance", "mid_small_component_count", "mid_gradient_magnitude_std",
    "bot_laplacian_variance", "bot_small_component_count", "bot_gradient_magnitude_std",
    "global_total_small_components", "global_overall_edge_density",
    "global_FAST_keypoint_count", "global_mean_brightness",
    "global_dark_channel_mean", "global_temporal_diff_mean",
]
META_COLS = ["sequence", "frame_id", "weather", "fn", "num_gt"]
COL_ORDER = META_COLS + FEATURE_COLS


# ── Feature extractors ─────────────────────────────────────────────

def extract_strip_features_selected(gray_strip):
    """3 features from a 180-row grayscale strip."""
    gray_f = gray_strip.astype(np.float64)

    # Small component count (from Canny edges)
    edges = cv2.Canny(gray_strip, 50, 150)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    small_count = int(np.sum(stats[1:, cv2.CC_STAT_AREA] < 500)) if n_labels > 1 else 0

    # Gradient magnitude std (Sobel)
    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    grad_std = float(np.sqrt(gx ** 2 + gy ** 2).std())

    # Laplacian variance (sharpness)
    lap_var = float(cv2.Laplacian(gray_f, cv2.CV_64F).var())

    return {
        "laplacian_variance": round(lap_var, 4),
        "small_component_count": small_count,
        "gradient_magnitude_std": round(grad_std, 4),
    }


def extract_global_features_selected(bgr, gray, prev_gray):
    """6 global features from the full frame."""
    h, w = gray.shape

    # Edge density + small components (share Canny result)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.mean() / 255.0)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    small_count = int(np.sum(stats[1:, cv2.CC_STAT_AREA] < 500)) if n_labels > 1 else 0

    # Mean brightness
    mean_bright = float(gray.mean())

    # Dark channel mean (weather proxy)
    min_channel = np.min(bgr, axis=2)
    dark = cv2.erode(min_channel, DARK_KERNEL)
    dark_mean = float(dark.mean())

    # Temporal diff
    if prev_gray is not None:
        temporal_diff = float(cv2.absdiff(gray, prev_gray).mean())
    else:
        temporal_diff = 0.0

    # FAST keypoints
    kps = FAST_DETECTOR.detect(gray, None)
    fast_count = len(kps) / (h * w)

    return {
        "total_small_components": small_count,
        "overall_edge_density": round(edge_density, 6),
        "FAST_keypoint_count": round(fast_count, 6),
        "mean_brightness": round(mean_bright, 4),
        "dark_channel_mean": round(dark_mean, 4),
        "temporal_diff_mean": round(temporal_diff, 4),
    }


def extract_frame_features(bgr, prev_gray):
    """Extract 15 selected features from one BGR frame.

    Returns (feature_dict, gray) — gray kept for temporal chain.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Strip features (top / mid / bottom)
    strips = {
        "top": gray[0:STRIP_H, :],
        "mid": gray[STRIP_H:2 * STRIP_H, :],
        "bot": gray[2 * STRIP_H:FRAME_H, :],
    }
    features = {}
    for prefix, strip in strips.items():
        sf = extract_strip_features_selected(strip)
        for k, v in sf.items():
            features[f"{prefix}_{k}"] = v

    # Global features
    gf = extract_global_features_selected(bgr, gray, prev_gray)
    for k, v in gf.items():
        features[f"global_{k}"] = v

    return features, gray


# ── Main extraction loop ───────────────────────────────────────────

def run_extraction():
    """Extract features for all 100 sequences.  Crash-safe: appends per sequence."""

    # Load target variable from detrac_results.csv (nano_320 rows only)
    print("Loading detrac_results.csv ...", flush=True)
    full_df = pd.read_csv(DETRAC_CSV)
    nano_df = full_df[full_df["model"] == "nano_320"].copy()
    nano_df = nano_df[["sequence", "frame_id", "fn", "weather", "num_gt"]]
    all_sequences = sorted(nano_df["sequence"].unique())
    total_seqs = len(all_sequences)
    total_frames = len(nano_df)
    print(f"  {total_frames} frames across {total_seqs} sequences\n", flush=True)

    # Crash safety: check which sequences are already done
    done_sequences = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        done_sequences = set(existing["sequence"].unique())
        print(f"  Resuming — {len(done_sequences)} sequences already extracted, "
              f"skipping them.\n", flush=True)
        done_frames = len(existing)
    else:
        done_frames = 0

    grouped = nano_df.groupby("sequence")
    seq_done = len(done_sequences)
    frames_done = done_frames
    times = []                     # elapsed per frame, for ETA

    for seq_name in all_sequences:
        if seq_name in done_sequences:
            print(f"  Skipping {seq_name} — already extracted", flush=True)
            continue

        seq_df = grouped.get_group(seq_name).sort_values("frame_id")
        img_dir = IMG_BASE / seq_name
        weather = seq_df["weather"].iloc[0]
        n_frames = len(seq_df)

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
                continue

            feats, gray = extract_frame_features(bgr, prev_gray)
            prev_gray = gray

            feats["sequence"] = seq_name
            feats["frame_id"] = frame_id
            feats["weather"] = weather
            feats["fn"] = int(row["fn"])
            feats["num_gt"] = int(row["num_gt"])
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

        # ETA
        avg_per_frame = sum(times) / len(times)
        remaining_frames = total_frames - frames_done
        eta_s = remaining_frames * avg_per_frame
        eta_min = eta_s / 60

        print(f"  {elapsed:.1f}s ({fps:.0f} fps)  "
              f"[{frames_done:,}/{total_frames:,} frames, ETA {eta_min:.1f}min]",
              flush=True)

    # Load final CSV for validation
    df = pd.read_csv(OUT_CSV)
    print(f"\nDone — {len(df):,} rows x {len(df.columns)} cols -> {OUT_CSV}\n",
          flush=True)
    return df


# ── Validation ─────────────────────────────────────────────────────

def validate(df):
    """Sanity-check the output DataFrame."""

    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # Shape
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Sequences: {df['sequence'].nunique()}")

    # NaN check
    nan_counts = df[FEATURE_COLS].isna().sum()
    if nan_counts.any():
        print("\nNaN counts:")
        for col in FEATURE_COLS:
            n = nan_counts[col]
            if n > 0:
                print(f"  {col}: {n}")
        # Fill temporal_diff NaN with 0
        if df["global_temporal_diff_mean"].isna().any():
            df["global_temporal_diff_mean"] = df["global_temporal_diff_mean"].fillna(0.0)
            df.to_csv(OUT_CSV, index=False)
            print("  -> Filled temporal_diff NaN with 0 and re-saved CSV")
    else:
        print("\nNo NaN values in any feature column.")

    # Feature ranges
    print(f"\n{'Feature':<40s} {'Min':>10s} {'Max':>10s} {'Mean':>10s}")
    print("-" * 70)
    for col in FEATURE_COLS:
        lo, hi, mu = df[col].min(), df[col].max(), df[col].mean()
        print(f"{col:<40s} {lo:10.4f} {hi:10.4f} {mu:10.4f}")

    # Spearman correlation vs fn
    print(f"\n{'Feature':<40s} {'Spearman rho':>12s} {'p-value':>12s}")
    print("-" * 64)
    corrs = []
    for col in FEATURE_COLS:
        rho, p = spearmanr(df[col], df["fn"])
        corrs.append((col, rho, p))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, rho, p in corrs:
        print(f"{col:<40s} {rho:12.4f} {p:12.2e}")

    print("\nCorrelation on full dataset (100 sequences) — compare with "
          "10-sequence selection results above.\n")


# ── Entry point ────────────────────────────────────────────────────

def main():
    df = run_extraction()
    validate(df)


if __name__ == "__main__":
    main()
