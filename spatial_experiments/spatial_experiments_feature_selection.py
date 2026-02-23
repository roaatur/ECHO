"""Extract ~40 candidate image features and correlate with nano@320 miss counts.

Computes strip-based (top/mid/bottom) and global features for each frame in
the 10 benchmark sequences, then ranks them by Spearman correlation and AUC
against per-frame false negatives.

Output:
  spatial_experiments_candidate_features.csv  — ~14,560 rows × ~43 columns
  spatial_experiments_feature_correlation.pdf — 3 pages
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(ROOT / "res_experiment"))
from res_evaluate_detrac import IMG_BASE

NANO320_CSV = SCRIPT_DIR / "spatial_experiments_nano320_results.csv"
OUT_CSV = SCRIPT_DIR / "spatial_experiments_candidate_features.csv"
OUT_PDF = SCRIPT_DIR / "spatial_experiments_feature_correlation.pdf"

FRAME_W, FRAME_H = 960, 540
STRIP_H = FRAME_H // 3  # 180 rows per strip

# Reusable OpenCV objects
DARK_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
FAST_DETECTOR = cv2.FastFeatureDetector_create(threshold=20)


# ── Strip features (9 per strip) ─────────────────────────────────
def extract_strip_features(gray_strip):
    """Extract 9 features from a grayscale strip (H×W uint8)."""
    gray_f = gray_strip.astype(np.float64)

    # Edge density
    edges = cv2.Canny(gray_strip, 50, 150)
    edge_density = edges.mean() / 255.0

    # Gradient magnitude (Sobel)
    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    gradient_mean = mag.mean()
    gradient_std = mag.std()

    # Small connected components from edge map
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    # Skip label 0 (background)
    small_count = int(np.sum(stats[1:, cv2.CC_STAT_AREA] < 500)) if n_labels > 1 else 0

    # Texture entropy (256-bin histogram)
    hist = cv2.calcHist([gray_strip], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-10)
    tex_entropy = float(entropy(hist_norm, base=2))

    # High frequency energy (FFT)
    h, w = gray_strip.shape
    f = np.fft.fft2(gray_f)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift) ** 2
    total_energy = magnitude.sum()
    # Low-pass mask: circle radius = min(h,w)/8
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 8
    Y, X = np.ogrid[:h, :w]
    low_mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2
    low_energy = magnitude[low_mask].sum()
    high_freq = (total_energy - low_energy) / (total_energy + 1e-10)

    # Laplacian variance (sharpness)
    lap_var = cv2.Laplacian(gray_f, cv2.CV_64F).var()

    # Brightness
    mean_bright = float(gray_strip.mean())

    # Contrast
    contrast = float(gray_f.std())

    return {
        "edge_density": round(edge_density, 6),
        "gradient_magnitude_mean": round(gradient_mean, 4),
        "gradient_magnitude_std": round(gradient_std, 4),
        "small_component_count": small_count,
        "texture_entropy": round(tex_entropy, 4),
        "high_freq_energy": round(high_freq, 6),
        "laplacian_variance": round(lap_var, 4),
        "mean_brightness": round(mean_bright, 4),
        "contrast_std": round(contrast, 4),
    }


# ── Global features (9) ──────────────────────────────────────────
def extract_global_features(bgr, gray, prev_gray):
    """Extract 9 global features from full frame."""
    gray_f = gray.astype(np.float64)
    h, w = gray.shape

    # Mean brightness & std
    mean_bright = float(gray.mean())
    bright_std = float(gray_f.std())

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0

    # Small components
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    small_count = int(np.sum(stats[1:, cv2.CC_STAT_AREA] < 500)) if n_labels > 1 else 0

    # Image entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-10)
    img_entropy = float(entropy(hist_norm, base=2))

    # Dark channel mean (weather proxy)
    min_channel = np.min(bgr, axis=2)
    dark = cv2.erode(min_channel, DARK_KERNEL)
    dark_mean = float(dark.mean())

    # Colorfulness (Hasler & Süsstrunk)
    B, G, R = bgr[:, :, 0].astype(float), bgr[:, :, 1].astype(float), bgr[:, :, 2].astype(float)
    rg = R - G
    yb = 0.5 * (R + G) - B
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) +
                         0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    # Temporal diff
    if prev_gray is not None:
        temporal_diff = float(cv2.absdiff(gray, prev_gray).mean())
    else:
        temporal_diff = 0.0

    # FAST keypoints
    kps = FAST_DETECTOR.detect(gray, None)
    fast_count = len(kps) / (h * w)

    return {
        "mean_brightness": round(mean_bright, 4),
        "brightness_std": round(bright_std, 4),
        "overall_edge_density": round(edge_density, 6),
        "total_small_components": small_count,
        "image_entropy": round(img_entropy, 4),
        "dark_channel_mean": round(dark_mean, 4),
        "colorfulness": round(colorfulness, 4),
        "temporal_diff_mean": round(temporal_diff, 4),
        "FAST_keypoint_count": round(fast_count, 6),
    }


# ── Combined per-frame feature extraction ────────────────────────
def extract_frame_features(bgr, prev_gray):
    """Extract all ~40 features from one BGR frame.

    Returns (feature_dict, gray) where gray is kept for temporal_diff.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Strip features (top / mid / bottom)
    strips = {
        "top": gray[0:STRIP_H, :],
        "mid": gray[STRIP_H:2 * STRIP_H, :],
        "bot": gray[2 * STRIP_H:FRAME_H, :],
    }
    features = {}
    strip_results = {}
    for prefix, strip in strips.items():
        sf = extract_strip_features(strip)
        strip_results[prefix] = sf
        for k, v in sf.items():
            features[f"{prefix}_{k}"] = v

    # Global features
    gf = extract_global_features(bgr, gray, prev_gray)
    for k, v in gf.items():
        features[f"global_{k}"] = v

    # Derived ratios (top / bottom)
    top = strip_results["top"]
    bot = strip_results["bot"]
    features["ratio_top_bottom_edge"] = round(
        top["edge_density"] / (bot["edge_density"] + 1e-6), 4)
    features["ratio_top_bottom_components"] = round(
        top["small_component_count"] / (bot["small_component_count"] + 1), 4)
    features["ratio_top_bottom_gradient"] = round(
        top["gradient_magnitude_mean"] / (bot["gradient_magnitude_mean"] + 1e-6), 4)
    features["ratio_top_bottom_entropy"] = round(
        top["texture_entropy"] / (bot["texture_entropy"] + 1e-6), 4)

    return features, gray


# ── Run extraction ────────────────────────────────────────────────
def run_extraction():
    """Extract features for all frames in the 10 benchmark sequences."""
    print("Loading nano@320 results...", flush=True)
    results_df = pd.read_csv(NANO320_CSV)
    print(f"  {len(results_df)} frames across "
          f"{results_df['sequence'].nunique()} sequences\n", flush=True)

    # Group by sequence, keep frame_id + fn
    grouped = results_df.groupby("sequence")

    all_records = []
    total_frames = len(results_df)
    done = 0

    for seq_name, seq_df in grouped:
        seq_df = seq_df.sort_values("frame_id")
        img_dir = IMG_BASE / seq_name
        weather = seq_df["weather"].iloc[0]

        print(f"  {seq_name} ({weather}): {len(seq_df)} frames ...", end="", flush=True)
        t0 = time.time()
        prev_gray = None

        for _, row in seq_df.iterrows():
            frame_id = int(row["frame_id"])
            fn = int(row["fn"])

            img_path = img_dir / f"img{frame_id:05d}.jpg"
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue

            features, gray = extract_frame_features(bgr, prev_gray)
            prev_gray = gray

            features["sequence"] = seq_name
            features["frame_id"] = frame_id
            features["weather"] = weather
            features["num_misses"] = fn
            all_records.append(features)
            done += 1

        elapsed = time.time() - t0
        print(f" {elapsed:.1f}s", flush=True)

    df = pd.DataFrame(all_records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows x {len(df.columns)} cols -> {OUT_CSV}", flush=True)
    return df


# ── Correlation analysis + PDF ────────────────────────────────────
def generate_pdf(df):
    """Generate 3-page correlation analysis PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # Identify feature columns (everything except metadata)
    meta_cols = {"sequence", "frame_id", "weather", "num_misses"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    target = df["num_misses"].values

    # ── Spearman correlations ─────────────────────────────────
    correlations = {}
    for col in feat_cols:
        vals = df[col].values.astype(float)
        if np.std(vals) < 1e-10:
            correlations[col] = 0.0
            continue
        rho, _ = spearmanr(vals, target)
        correlations[col] = rho

    corr_df = pd.DataFrame({
        "feature": list(correlations.keys()),
        "rho": list(correlations.values()),
    })
    corr_df["abs_rho"] = corr_df["rho"].abs()
    corr_df = corr_df.sort_values("abs_rho", ascending=True)

    # Print top 10
    print("\n=== Top 10 Features by |Spearman rho| ===", flush=True)
    for _, row in corr_df.tail(10).iloc[::-1].iterrows():
        print(f"  {row['feature']:40s}  rho={row['rho']:+.4f}", flush=True)

    # ── AUC for binary task (>= 3 misses) ─────────────────────
    binary_target = (target >= 3).astype(int)
    auc_scores = {}
    if binary_target.sum() > 0 and (1 - binary_target).sum() > 0:
        for col in feat_cols:
            vals = df[col].values.astype(float)
            if np.std(vals) < 1e-10:
                auc_scores[col] = 0.5
                continue
            try:
                auc = roc_auc_score(binary_target, vals)
                # Flip if AUC < 0.5 (feature inversely predictive)
                auc_scores[col] = max(auc, 1 - auc)
            except ValueError:
                auc_scores[col] = 0.5

    auc_df = pd.DataFrame({
        "feature": list(auc_scores.keys()),
        "auc": list(auc_scores.values()),
    }).sort_values("auc", ascending=True)

    print("\n=== Top 10 Features by AUC (>=3 misses) ===", flush=True)
    for _, row in auc_df.tail(10).iloc[::-1].iterrows():
        print(f"  {row['feature']:40s}  AUC={row['auc']:.4f}", flush=True)

    # ── PDF ────────────────────────────────────────────────────
    with PdfPages(str(OUT_PDF)) as pdf:

        # ── PAGE 1: Correlation bar chart ─────────────────────
        fig, ax = plt.subplots(figsize=(12, max(8, len(feat_cols) * 0.28)))

        colors = []
        for rho in corr_df["abs_rho"]:
            if rho > 0.3:
                colors.append("#2ecc71")
            elif rho > 0.2:
                colors.append("#f39c12")
            else:
                colors.append("#e74c3c")

        ax.barh(range(len(corr_df)), corr_df["rho"].values, color=colors)
        ax.set_yticks(range(len(corr_df)))
        ax.set_yticklabels(corr_df["feature"].values, fontsize=7)
        ax.set_xlabel("Spearman rho vs num_misses")
        ax.axvline(0, color="black", lw=0.5)
        ax.axvline(0.3, color="#2ecc71", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(-0.3, color="#2ecc71", ls="--", lw=0.8, alpha=0.5)

        fig.suptitle(
            f"Page 1: Feature Correlation with Nano@320 Misses  "
            f"(n={len(df):,} frames, {len(feat_cols)} features)",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── PAGE 2: Top 10 scatter plots ──────────────────────
        top10 = corr_df.tail(10).iloc[::-1]
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes_flat = axes.flatten()

        for idx, (_, row) in enumerate(top10.iterrows()):
            ax = axes_flat[idx]
            feat = row["feature"]
            rho = row["rho"]
            ax.scatter(df[feat].values, target, alpha=0.08, s=4,
                       color="steelblue", edgecolors="none")
            ax.set_title(f"{feat}\nrho={rho:+.3f}", fontsize=8)
            ax.set_xlabel(feat, fontsize=6)
            ax.set_ylabel("misses", fontsize=6)
            ax.tick_params(labelsize=6)

        fig.suptitle("Page 2: Top 10 Features vs Miss Count",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── PAGE 3: Binary AUC chart ──────────────────────────
        if len(auc_df) > 0:
            fig, ax = plt.subplots(figsize=(12, max(8, len(feat_cols) * 0.28)))

            auc_colors = []
            for auc in auc_df["auc"]:
                if auc > 0.65:
                    auc_colors.append("#2ecc71")
                elif auc > 0.55:
                    auc_colors.append("#f39c12")
                else:
                    auc_colors.append("#e74c3c")

            ax.barh(range(len(auc_df)), auc_df["auc"].values, color=auc_colors)
            ax.set_yticks(range(len(auc_df)))
            ax.set_yticklabels(auc_df["feature"].values, fontsize=7)
            ax.set_xlabel("AUC (binary: >= 3 misses)")
            ax.axvline(0.5, color="black", lw=0.5)
            ax.axvline(0.65, color="#2ecc71", ls="--", lw=0.8, alpha=0.5)
            ax.set_xlim(0.4, 1.0)

            n_pos = int(binary_target.sum())
            n_neg = int(len(binary_target) - n_pos)
            fig.suptitle(
                f"Page 3: AUC for Binary Classification (>= 3 misses)  "
                f"[pos={n_pos:,}, neg={n_neg:,}]",
                fontsize=13, fontweight="bold",
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"\nPDF saved -> {OUT_PDF}", flush=True)


# ── Main ──────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    df = run_extraction()
    generate_pdf(df)

    print(f"\nTotal time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
