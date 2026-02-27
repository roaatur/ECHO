#!/usr/bin/env python3
"""D²-City Step 3 — Feature selection: redundancy pruning + shifted-label validation.

Ports new_features/new_select_features.py to D²-City.

Steps:
  1. Compute within-clip feature-feature Spearman |ρ| matrix (top 30 features)
  2. Hierarchical clustering to merge features with |ρ| > 0.85 intra-sequence
  3. Keep the best representative per cluster (highest |ρ| vs fn_nano)
  4. Shifted-label Spearman validation: predict mean(fn_nano[t+30:t+60])
     from mean(feature[t:t+30]) — the actual 1-window-ahead forecasting task

The "sequence" unit here is a D²-City clip (moving dashcam).
Each clip is an independent time series; analysis is within-clip.

Output: D2_select_features_results.pdf
"""

import warnings
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D2_common import SCRIPT_DIR

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_CSV  = SCRIPT_DIR / "D2_all_features.csv"
ANALYSIS_CSV  = SCRIPT_DIR / "D2_analysis_summary.csv"   # from D2_analyze_correlation.py
OUT_PDF       = SCRIPT_DIR / "D2_select_features_results.pdf"

META_COLS = ["clip_id", "frame_idx", "fn_nano", "num_gt_coco",
             "tp_nano", "tp_medium", "recoverable_fn"]
TARGET    = "fn_nano"
WINDOW    = 30      # 30 sampled frames ≈ 3.6 s at 8.3 fps effective
MIN_FRAMES = 60     # need at least 2× WINDOW


# ── 1. Within-clip feature-feature correlation ────────────────────────────────

def within_clip_feature_corr(df, feature_cols):
    """Mean |ρ| matrix (within-clip Spearman) across all clips."""
    n        = len(feature_cols)
    corr_sum = np.zeros((n, n))
    n_valid  = 0

    for clip_id, cdf in df.groupby("clip_id"):
        if len(cdf) < MIN_FRAMES:
            continue
        data = cdf[feature_cols].values
        if (data.std(axis=0) < 1e-10).sum() > n // 2:
            continue
        rho_mat, _ = spearmanr(data)
        if rho_mat.ndim == 0:
            continue
        corr_sum += np.abs(np.nan_to_num(rho_mat, nan=0.0))
        n_valid  += 1

    if n_valid > 0:
        corr_sum /= n_valid
    np.fill_diagonal(corr_sum, 1.0)
    return pd.DataFrame(corr_sum, index=feature_cols, columns=feature_cols)


# ── 2. Shifted-label Spearman validation ─────────────────────────────────────

def shifted_label_spearman(df, feature_cols, target, window):
    """For each feature, compute mean |ρ| between mean(feature[t:t+w])
    and mean(target[t+w:t+2w]) using non-overlapping windows within each clip.
    """
    results = []
    for feat in feature_cols:
        rhos = []
        for clip_id, cdf in df.groupby("clip_id"):
            cdf = cdf.sort_values("frame_idx").reset_index(drop=True)
            n   = len(cdf)
            if n < 2 * window:
                continue
            feat_means, tgt_means = [], []
            for t in range(0, n - 2 * window + 1, window):
                feat_means.append(cdf[feat].iloc[t          : t + window].mean())
                tgt_means.append( cdf[target].iloc[t + window: t + 2*window].mean())
            if len(feat_means) < 5:
                continue
            fa, ta = np.array(feat_means), np.array(tgt_means)
            if fa.std() < 1e-10 or ta.std() < 1e-10:
                continue
            rho, _ = spearmanr(fa, ta)
            if not np.isnan(rho):
                rhos.append(rho)
        if rhos:
            ra = np.array(rhos)
            results.append({"feature": feat,
                             "shifted_abs_rho":    float(np.abs(ra).mean()),
                             "shifted_signed_rho": float(ra.mean()),
                             "shifted_std":        float(ra.std()),
                             "n_clips":            len(rhos)})
        else:
            results.append({"feature": feat, "shifted_abs_rho": 0.0,
                             "shifted_signed_rho": 0.0, "shifted_std": 0.0,
                             "n_clips": 0})
    return pd.DataFrame(results).sort_values("shifted_abs_rho", ascending=False)


# ── 3. Hierarchical clustering for redundancy pruning ────────────────────────

def select_features(corr_matrix, summary_df, threshold=0.85):
    features = list(corr_matrix.index)
    n        = len(features)
    dist     = np.clip((1.0 - corr_matrix.values + (1.0 - corr_matrix.values).T) / 2,
                       0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z         = linkage(condensed, method="average")
    clusters  = fcluster(Z, t=1.0 - threshold, criterion="distance")

    rho_map  = summary_df.set_index("feature")["mean_abs_rho"].to_dict()
    selected, cluster_info = [], {}

    for cid in sorted(set(clusters)):
        members = [features[i] for i in range(n) if clusters[i] == cid]
        members.sort(key=lambda f: rho_map.get(f, 0.0), reverse=True)
        best = members[0]
        selected.append(best)
        cluster_info[cid] = {"selected": best, "rho": rho_map.get(best, 0.0),
                              "members": members, "dropped": members[1:]}

    selected.sort(key=lambda f: rho_map.get(f, 0.0), reverse=True)
    return selected, clusters, Z, cluster_info


# ── 4. PDF report ─────────────────────────────────────────────────────────────

def generate_pdf(top_features, corr_matrix, Z, clusters, selected,
                 cluster_info, shifted_df, rho_map):
    with PdfPages(OUT_PDF) as pdf:

        # Page 1: Dendrogram
        fig, ax = plt.subplots(figsize=(16, 10))
        dendrogram(Z, labels=top_features, ax=ax,
                   leaf_rotation=90, leaf_font_size=7,
                   color_threshold=1.0 - 0.85)
        ax.axhline(y=1.0 - 0.85, color="red", ls="--", alpha=0.7,
                   label="Merge threshold (|ρ| > 0.85)")
        ax.set_ylabel("Distance (1 − |ρ|)")
        ax.set_title("D²-City Feature Redundancy Clustering\n"
                     "(Within-Clip Spearman, Top 30 Features)")
        ax.legend(fontsize=9)
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 2: Correlation heatmap
        fig, ax = plt.subplots(figsize=(16, 14))
        im = ax.imshow(corr_matrix.values, cmap="YlOrRd", vmin=0, vmax=1)
        labels = list(corr_matrix.index)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        for i, lab in enumerate(labels):
            if lab in selected:
                ax.get_yticklabels()[i].set_fontweight("bold")
                ax.get_yticklabels()[i].set_color("blue")
        plt.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title("D²-City Within-Clip Feature Correlation (Mean |ρ|)\n"
                     "Blue/Bold = Selected features")
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 3: Shifted-label validation
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        sdf   = shifted_df.sort_values("shifted_abs_rho", ascending=True)
        y_pos = range(len(sdf))

        ax = axes[0]
        ax.barh(y_pos, sdf["shifted_abs_rho"], color="#2196F3")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sdf["feature"], fontsize=8)
        ax.set_xlabel("Mean |ρ| (shifted-label, w=30 sampled frames)")
        ax.set_title(f"Shifted-Label Spearman\n"
                     f"predict mean(fn_nano[t+{WINDOW}:t+{2*WINDOW}]) "
                     f"from mean(feat[t:t+{WINDOW}])")

        ax = axes[1]
        feats     = sdf["feature"].tolist()
        same_rho  = [rho_map.get(f, 0) for f in feats]
        shift_rho = sdf["shifted_abs_rho"].values
        x = np.arange(len(feats))
        ax.barh(x - 0.2, same_rho,  0.35, label="Same-time |ρ|",  color="#9E9E9E")
        ax.barh(x + 0.2, shift_rho, 0.35, label="Shifted |ρ|",    color="#2196F3")
        ax.set_yticks(x)
        ax.set_yticklabels(feats, fontsize=8)
        ax.legend(fontsize=9)
        ax.set_xlabel("Mean |ρ|")
        ax.set_title("Same-Time vs Shifted-Label (D²-City)")
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 4: Summary text table
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis("off")
        lines = [
            "=" * 80,
            "D²-CITY FEATURE SELECTION SUMMARY",
            "=" * 80,
            "",
            f"Dataset     : D²-City validation (dashcam, moving camera)",
            f"Input       : 65 features, top 30 analyzed for redundancy",
            f"Method      : Hierarchical clustering (merge if within-clip |ρ| > 0.85)",
            f"Selected    : {len(selected)} non-redundant features",
            "",
            f"Validation  : Shifted-label Spearman (window = {WINDOW} sampled frames",
            f"              ≈ {WINDOW * 3 / 25:.1f} s at 25 fps / SUBSAMPLE=3)",
            f"  Task      : predict mean(fn_nano[t+{WINDOW}:t+{2*WINDOW}]) "
            f"from mean(feature[t:t+{WINDOW}])",
            "",
            f"{'Feature':<35s} {'Same-time':>10s} {'Shifted':>10s} {'Dropped from cluster':<35s}",
            "-" * 90,
        ]
        for f in selected:
            same = rho_map.get(f, 0)
            row  = shifted_df[shifted_df["feature"] == f]
            shft = float(row["shifted_abs_rho"].values[0]) if len(row) else 0.0
            for cid, info in cluster_info.items():
                if info["selected"] == f:
                    dropped = ", ".join(info["dropped"][:3]) or "(singleton)"
                    break
            else:
                dropped = ""
            lines.append(f"{f:<35s} {same:10.4f} {shft:10.4f} {dropped:<35s}")

        lines += [
            "",
            "=" * 80,
            "COPY-PASTE FOR TRAINING SCRIPTS:",
            "=" * 80,
            "SELECTED_FEATURES = [",
        ]
        for f in selected:
            lines.append(f'    "{f}",')
        lines.append("]")

        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontfamily="monospace", fontsize=7, verticalalignment="top")
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

    print(f"  PDF: {OUT_PDF}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("D²-CITY FEATURE SELECTION: REDUNDANCY PRUNING + SHIFTED-LABEL")
    print("=" * 70)

    if not FEATURES_CSV.exists():
        print(f"\n  ERROR: {FEATURES_CSV} not found. Run D2_extract_features.py first.")
        return None

    if not ANALYSIS_CSV.exists():
        print(f"\n  ERROR: {ANALYSIS_CSV} not found. Run D2_analyze_correlation.py first.")
        return None

    df      = pd.read_csv(FEATURES_CSV)
    summary = pd.read_csv(ANALYSIS_CSV)

    feat_cols = [c for c in df.columns if c not in META_COLS]
    for c in feat_cols:
        if "temporal" in c or c == "motion_pixel_ratio":
            df[c] = df[c].fillna(0.0)

    print(f"\n  {len(df):,} frames, {df['clip_id'].nunique()} clips, "
          f"{len(feat_cols)} features")

    # Step 1: Feature-feature correlation (top 30 by within-clip |ρ|)
    print("\n[1/4] Within-clip feature-feature correlation (top 30) ...", flush=True)
    top30       = summary.head(30)["feature"].tolist()
    corr_matrix = within_clip_feature_corr(df, top30)

    print("\n  Highly correlated pairs (|ρ| > 0.85):")
    n_pairs = 0
    for i, f1 in enumerate(top30):
        for j, f2 in enumerate(top30):
            if i >= j:
                continue
            r = corr_matrix.loc[f1, f2]
            if r > 0.85:
                n_pairs += 1
                print(f"    {f1:35s} ↔ {f2:35s}  |ρ|={r:.3f}")
    print(f"  Total: {n_pairs} pairs above threshold")

    # Step 2: Cluster and select
    print("\n[2/4] Clustering ...", flush=True)
    selected, clusters, Z, cluster_info = select_features(corr_matrix, summary)

    rho_map = summary.set_index("feature")["mean_abs_rho"].to_dict()
    print(f"\n  {len(selected)} features selected from {max(clusters)} clusters:")
    for cid in sorted(cluster_info, key=lambda c: cluster_info[c]["rho"], reverse=True):
        info    = cluster_info[cid]
        dropped = ", ".join(info["dropped"][:3]) if info["dropped"] else "(sole member)"
        print(f"    {info['selected']:35s}  |ρ|={info['rho']:.4f}  "
              f"dropped: {dropped}")

    # Step 3: Shifted-label validation on selected features
    print("\n[3/4] Shifted-label Spearman (w=30) on selected features ...", flush=True)
    shifted_df = shifted_label_spearman(df, selected, TARGET, WINDOW)

    print(f"\n  {'Feature':<35s} {'Same-time |ρ|':>14s} {'Shifted |ρ|':>12s} {'Δ':>6s}")
    print(f"  {'-' * 67}")
    for _, row in shifted_df.iterrows():
        feat  = row["feature"]
        same  = rho_map.get(feat, 0)
        shift = row["shifted_abs_rho"]
        print(f"  {feat:<35s} {same:14.4f} {shift:12.4f} {shift - same:+6.3f}")

    # Step 4: PDF
    print("\n[4/4] Generating PDF ...", flush=True)
    generate_pdf(top30, corr_matrix, Z, clusters, selected,
                 cluster_info, shifted_df, rho_map)

    print(f"\n{'=' * 70}")
    print("SELECTED_FEATURES = [")
    for f in selected:
        print(f'    "{f}",')
    print("]")
    print(f"{'=' * 70}")

    return selected


if __name__ == "__main__":
    main()
