#!/usr/bin/env python3
"""Shifted-label Spearman correlation + hierarchical clustering on METEOR 65 features.

For each clip, computes windowed mean of future fn_coco_nano (window=10, horizon=10),
then correlates each feature with the shifted target within each clip.
Averages correlations across clips (Fisher-z transform).
Performs hierarchical clustering to select top non-redundant features.

Output: meteor_validation/meteor_65_correlation_results.pdf
        meteor_validation/meteor_65_selected_features.txt
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "meteor_65_features.csv"
OUT_PDF = SCRIPT_DIR / "meteor_65_correlation_results.pdf"
OUT_FEATURES = SCRIPT_DIR / "meteor_65_selected_features.txt"

META_COLS = ["clip", "frame_id", "fn_coco_nano", "fn_total_nano", "num_gt",
             "num_gt_coco", "tp_nano", "tp_medium", "fn_coco_medium",
             "fn_total_medium", "recoverable_fn"]
TARGET = "fn_coco_nano"
WINDOW = 10
HORIZON = 10


def build_shifted_targets(group, target_col, window, horizon):
    """For a clip's frames, compute mean target over [t+1 : t+horizon] for each t."""
    vals = group[target_col].values.astype(float)
    n = len(vals)
    targets = []
    for i in range(n):
        start = i + 1
        end = min(i + horizon + 1, n)
        if start >= n:
            targets.append(np.nan)
        else:
            targets.append(np.mean(vals[start:end]))
    return np.array(targets)


def main():
    print("Loading METEOR 65-feature CSV ...", flush=True)
    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  {len(df)} frames, {df['clip'].nunique()} clips, {len(feature_cols)} features",
          flush=True)

    # ── Step 1: Shifted-label within-clip Spearman correlations ──────
    print("\nComputing shifted-label within-clip Spearman correlations ...", flush=True)

    clip_corrs = {f: [] for f in feature_cols}
    clip_names = []

    for clip_name, grp in df.groupby("clip"):
        grp = grp.sort_values("frame_id").reset_index(drop=True)
        if len(grp) < WINDOW + HORIZON:
            continue

        shifted_target = build_shifted_targets(grp, TARGET, WINDOW, HORIZON)
        valid = ~np.isnan(shifted_target)

        if valid.sum() < 20:
            continue

        clip_names.append(clip_name)
        for feat in feature_cols:
            x = grp[feat].values[valid]
            y = shifted_target[valid]
            # Drop any NaN in features
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 10:
                clip_corrs[feat].append(np.nan)
                continue
            rho, _ = stats.spearmanr(x[mask], y[mask])
            clip_corrs[feat].append(rho)

    print(f"  Valid clips: {len(clip_names)}", flush=True)

    # Fisher-z average across clips
    results = []
    for feat in feature_cols:
        rhos = [r for r in clip_corrs[feat] if not np.isnan(r)]
        if len(rhos) < 3:
            results.append({"feature": feat, "mean_rho": 0.0, "abs_rho": 0.0,
                           "std_rho": 0.0, "n_clips": len(rhos)})
            continue
        # Fisher z-transform
        z_vals = np.arctanh(np.clip(rhos, -0.999, 0.999))
        mean_z = np.mean(z_vals)
        std_z = np.std(z_vals)
        mean_rho = np.tanh(mean_z)
        results.append({
            "feature": feat,
            "mean_rho": mean_rho,
            "abs_rho": abs(mean_rho),
            "std_rho": np.tanh(std_z),
            "n_clips": len(rhos),
        })

    res_df = pd.DataFrame(results).sort_values("abs_rho", ascending=False)
    print("\n  Top 20 features by |shifted-label ρ|:", flush=True)
    for i, row in res_df.head(20).iterrows():
        print(f"    {row['feature']:40s} ρ={row['mean_rho']:+.4f} (|ρ|={row['abs_rho']:.4f}, "
              f"std={row['std_rho']:.3f}, n={row['n_clips']})", flush=True)

    # ── Step 2: Same-time within-clip correlations (for comparison) ──
    print("\nComputing same-time within-clip Spearman correlations ...", flush=True)
    same_corrs = {f: [] for f in feature_cols}
    for clip_name, grp in df.groupby("clip"):
        grp = grp.sort_values("frame_id").reset_index(drop=True)
        if len(grp) < 20:
            continue
        target_vals = grp[TARGET].values.astype(float)
        for feat in feature_cols:
            x = grp[feat].values
            mask = ~(np.isnan(x) | np.isnan(target_vals))
            if mask.sum() < 10:
                same_corrs[feat].append(np.nan)
                continue
            rho, _ = stats.spearmanr(x[mask], target_vals[mask])
            same_corrs[feat].append(rho)

    same_results = []
    for feat in feature_cols:
        rhos = [r for r in same_corrs[feat] if not np.isnan(r)]
        if len(rhos) < 3:
            same_results.append({"feature": feat, "same_rho": 0.0, "same_abs": 0.0})
            continue
        z_vals = np.arctanh(np.clip(rhos, -0.999, 0.999))
        mean_rho = np.tanh(np.mean(z_vals))
        same_results.append({"feature": feat, "same_rho": mean_rho, "same_abs": abs(mean_rho)})

    same_df = pd.DataFrame(same_results)
    res_df = res_df.merge(same_df, on="feature")

    print("\n  Top 10 same-time correlations:", flush=True)
    for _, row in res_df.sort_values("same_abs", ascending=False).head(10).iterrows():
        print(f"    {row['feature']:40s} same-time ρ={row['same_rho']:+.4f}, "
              f"shifted ρ={row['mean_rho']:+.4f}", flush=True)

    # ── Step 3: Hierarchical clustering ──────────────────────────────
    print("\nHierarchical clustering to remove redundant features ...", flush=True)

    # Compute feature-feature Spearman correlation matrix
    feat_data = df[feature_cols].dropna()
    if len(feat_data) > 10000:
        feat_data = feat_data.sample(10000, random_state=42)

    corr_matrix = feat_data.corr(method="spearman").values
    # Distance = 1 - |correlation|
    dist_matrix = 1.0 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0)
    # Ensure symmetry and non-negative
    dist_matrix = np.clip((dist_matrix + dist_matrix.T) / 2, 0, None)

    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="average")

    # Threshold = 0.3 means features with |ρ| > 0.7 are in same cluster
    threshold = 0.3
    clusters = fcluster(Z, t=threshold, criterion="distance")
    n_clusters = len(set(clusters))
    print(f"  {n_clusters} clusters at threshold {threshold} (|ρ| > {1-threshold})", flush=True)

    # From each cluster, pick the feature with highest |shifted-label ρ|
    res_df["cluster"] = [clusters[feature_cols.index(f)] for f in res_df["feature"]]
    selected = []
    for cid in sorted(set(clusters)):
        cluster_feats = res_df[res_df["cluster"] == cid].sort_values("abs_rho", ascending=False)
        best = cluster_feats.iloc[0]
        selected.append(best)

    selected_df = pd.DataFrame(selected).sort_values("abs_rho", ascending=False)

    print(f"\n  Selected {len(selected_df)} non-redundant features:", flush=True)
    for i, (_, row) in enumerate(selected_df.iterrows()):
        marker = "***" if row["abs_rho"] >= 0.1 else "   "
        print(f"    {i+1:2d}. {row['feature']:40s} ρ={row['mean_rho']:+.4f} "
              f"(cluster {int(row['cluster'])}) {marker}", flush=True)

    # Select features with |ρ| >= some threshold
    # Use top 20 or all with |ρ| > 0.05
    strong = selected_df[selected_df["abs_rho"] >= 0.05]
    if len(strong) < 10:
        strong = selected_df.head(20)
    top_n = min(20, len(strong))
    final_features = strong.head(top_n)["feature"].tolist()

    print(f"\n  Final selected: {len(final_features)} features with |ρ| >= 0.05", flush=True)
    for i, f in enumerate(final_features):
        row = res_df[res_df["feature"] == f].iloc[0]
        print(f"    {i+1:2d}. {f:40s} ρ={row['mean_rho']:+.4f}", flush=True)

    # Save feature list
    with open(OUT_FEATURES, "w") as fh:
        for f in final_features:
            fh.write(f + "\n")
    print(f"\n  Feature list saved to {OUT_FEATURES}", flush=True)

    # ── Step 4: Compare with DETRAC-20 ───────────────────────────────
    detrac_20 = [
        "fast_keypoints_half", "edge_density_coarse", "gradient_magnitude_mean",
        "gabor_nyquist_energy", "fast_keypoints_full", "glcm_homogeneity",
        "overall_edge_density", "top_edge_density", "top_high_freq_energy",
        "top_gradient_std", "fft_critical_band_ratio", "total_edge_components",
        "top_laplacian_var", "glcm_entropy", "mid_edge_density",
        "dark_channel_mean", "mid_gradient_std", "foreground_blob_count",
        "mid_high_freq_energy", "rms_contrast",
    ]

    print("\n  DETRAC-20 features ranked by METEOR shifted-label |ρ|:", flush=True)
    for f in detrac_20:
        if f in res_df["feature"].values:
            row = res_df[res_df["feature"] == f].iloc[0]
            in_selected = "✓" if f in final_features else " "
            print(f"    {in_selected} {f:40s} ρ={row['mean_rho']:+.4f} (rank on METEOR: "
                  f"{list(res_df.sort_values('abs_rho', ascending=False)['feature']).index(f)+1})",
                  flush=True)
        else:
            print(f"      {f:40s} NOT IN CSV", flush=True)

    overlap = set(final_features) & set(detrac_20)
    print(f"\n  Overlap between METEOR-selected and DETRAC-20: {len(overlap)}/{len(final_features)}",
          flush=True)
    if overlap:
        print(f"    Shared: {sorted(overlap)}", flush=True)
    only_meteor = set(final_features) - set(detrac_20)
    if only_meteor:
        print(f"    METEOR-only: {sorted(only_meteor)}", flush=True)
    only_detrac = set(detrac_20) - set(final_features)
    if only_detrac:
        print(f"    DETRAC-only: {sorted(only_detrac)}", flush=True)

    # ── Step 5: Generate PDF ─────────────────────────────────────────
    print("\nGenerating PDF ...", flush=True)
    with PdfPages(str(OUT_PDF)) as pdf:
        # Page 1: Top 30 features bar chart (shifted-label)
        fig, ax = plt.subplots(figsize=(12, 10))
        top30 = res_df.sort_values("abs_rho", ascending=False).head(30)
        colors = ["#2196F3" if r >= 0 else "#F44336" for r in top30["mean_rho"]]
        y_pos = range(len(top30))
        ax.barh(y_pos, top30["mean_rho"], color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top30["feature"], fontsize=8)
        ax.set_xlabel("Mean within-clip Spearman ρ (shifted-label)")
        ax.set_title("METEOR: Top 30 Features by Shifted-Label Correlation\n"
                     f"(window={WINDOW}, horizon={HORIZON}, {len(clip_names)} clips)")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Shifted vs same-time scatter
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(res_df["same_rho"], res_df["mean_rho"], alpha=0.5, s=30)
        for _, row in res_df.head(15).iterrows():
            ax.annotate(row["feature"], (row["same_rho"], row["mean_rho"]),
                       fontsize=6, alpha=0.7)
        ax.set_xlabel("Same-time within-clip ρ")
        ax.set_ylabel("Shifted-label within-clip ρ")
        ax.set_title("METEOR: Same-Time vs Shifted-Label Correlations")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        # Diagonal
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Dendrogram of selected features
        from scipy.cluster.hierarchy import dendrogram
        fig, ax = plt.subplots(figsize=(14, 8))
        sel_indices = [feature_cols.index(f) for f in final_features]
        sel_corr = corr_matrix[np.ix_(sel_indices, sel_indices)]
        sel_dist = 1.0 - np.abs(sel_corr)
        np.fill_diagonal(sel_dist, 0)
        sel_dist = np.clip((sel_dist + sel_dist.T) / 2, 0, None)
        sel_condensed = squareform(sel_dist, checks=False)
        sel_Z = linkage(sel_condensed, method="average")
        dendrogram(sel_Z, labels=final_features, ax=ax, leaf_rotation=90, leaf_font_size=7)
        ax.set_title(f"METEOR: Dendrogram of {len(final_features)} Selected Features")
        ax.set_ylabel("Distance (1 - |ρ|)")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: DETRAC-20 vs METEOR-selected comparison
        fig, ax = plt.subplots(figsize=(10, 8))
        all_feats = sorted(set(final_features) | set(detrac_20))
        y_pos = range(len(all_feats))
        meteor_rhos = []
        labels = []
        bar_colors = []
        for f in all_feats:
            if f in res_df["feature"].values:
                rho = res_df[res_df["feature"] == f]["mean_rho"].values[0]
            else:
                rho = 0
            meteor_rhos.append(rho)
            in_m = f in final_features
            in_d = f in detrac_20
            if in_m and in_d:
                bar_colors.append("#4CAF50")  # green = both
                labels.append(f + " [BOTH]")
            elif in_m:
                bar_colors.append("#2196F3")  # blue = METEOR only
                labels.append(f + " [METEOR]")
            else:
                bar_colors.append("#FF9800")  # orange = DETRAC only
                labels.append(f + " [DETRAC]")
        ax.barh(y_pos, meteor_rhos, color=bar_colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("METEOR shifted-label ρ")
        ax.set_title("Feature Selection Comparison: METEOR vs DETRAC-20")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: Summary table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        summary = (
            f"METEOR 65-Feature Correlation Analysis\n"
            f"{'='*50}\n\n"
            f"Dataset: {len(df)} frames, {df['clip'].nunique()} clips\n"
            f"Window: {WINDOW} frames, Horizon: {HORIZON} frames\n"
            f"Valid clips for correlation: {len(clip_names)}\n\n"
            f"Clustering: threshold={threshold} (|ρ| > {1-threshold})\n"
            f"  → {n_clusters} clusters from {len(feature_cols)} features\n\n"
            f"Selected features: {len(final_features)}\n"
            f"  Best: {final_features[0]} (ρ={res_df[res_df['feature']==final_features[0]]['mean_rho'].values[0]:+.4f})\n\n"
            f"Overlap with DETRAC-20: {len(overlap)}/{len(final_features)}\n"
            f"  Shared: {', '.join(sorted(overlap)) if overlap else 'none'}\n"
            f"  METEOR-only: {len(only_meteor)}\n"
            f"  DETRAC-only: {len(only_detrac)}\n"
        )
        ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nPDF: {OUT_PDF}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
