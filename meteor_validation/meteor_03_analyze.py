#!/usr/bin/env python3
"""Spearman correlation + Elastic Net analysis for METEOR validation.

Reads meteor_spatial_features.csv (41 features), computes:
  - Within-sequence Spearman rho vs recoverable_fn (primary target)
  - Within-sequence Spearman rho vs fn_coco_nano (secondary, for comparison)
  - Delta features correlation
  - Elastic Net feature selection (per-sequence normalized)
  - 6-page PDF report

Output:
  - meteor_validation/meteor_correlation_results.csv
  - meteor_validation/meteor_validation_report.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

FEAT_CSV = SCRIPT_DIR / "meteor_spatial_features.csv"
CORR_CSV = SCRIPT_DIR / "meteor_correlation_results.csv"
PDF_PATH = SCRIPT_DIR / "meteor_validation_report.pdf"

# ── Feature columns (41 values) ──────────────────────────────────
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

# ── Feature -> Category mapping ──────────────────────────────────
FEATURE_CATEGORIES = {
    "edge_density": "A", "gradient_mag_mean": "A", "gradient_mag_std": "A",
    "laplacian_var": "A", "highfreq_energy_ratio": "A", "lbp_entropy": "A",
    "gabor_energy": "A", "gradient_hist_entropy": "A",
    "brightness_mean": "B", "brightness_std": "B", "dark_channel": "B",
    "pct_dark_pixels": "B", "pct_bright_pixels": "B",
    "local_contrast_rms": "B", "luminance_range": "B",
    "saturation_mean": "C", "saturation_std": "C", "colorfulness": "C",
    "hue_entropy": "C",
    "shannon_entropy": "D", "entropy_2d": "D", "spatial_frequency": "D",
    "average_gradient": "D", "mscn_mean": "D", "mscn_std": "D",
    "connected_components": "E", "small_component_count": "E",
    "small_component_ratio": "E", "top_edge_density": "E",
    "middle_edge_density": "E", "bottom_edge_density": "E",
    "flow_magnitude_mean": "F", "flow_magnitude_std": "F",
    "flow_direction_entropy": "F", "temporal_diff_magnitude": "F",
    "temporal_diff_ratio": "F",
    "fast_keypoints_full": "G", "fast_keypoints_320": "G",
    "keypoint_loss_ratio": "G", "multiscale_edge_ratio": "G",
    "small_edge_blob_count": "G",
}

CATEGORY_NAMES = {
    "A": "Edge/Texture", "B": "Luminance", "C": "Color",
    "D": "Info/Freq", "E": "Spatial", "F": "Temporal", "G": "Scale",
}

CATEGORY_COLORS = {
    "A": "#e74c3c", "B": "#f39c12", "C": "#2ecc71",
    "D": "#3498db", "E": "#9b59b6", "F": "#1abc9c", "G": "#e67e22",
}


# ── Correlation computation ───────────────────────────────────────

def compute_correlations(df, feature_cols, target="recoverable_fn"):
    """Compute global and within-sequence Spearman for each feature."""
    results = []
    for feature in feature_cols:
        if feature not in df.columns:
            continue

        # Global Spearman
        if df[feature].std() > 0 and df[target].std() > 0:
            global_rho, _ = spearmanr(df[feature], df[target])
        else:
            global_rho = 0.0

        # Within-sequence Spearman
        per_seq_rhos = []
        for _, group in df.groupby("clip"):
            if len(group) < 30:
                continue
            if group[target].std() > 0 and group[feature].std() > 0:
                rho, _ = spearmanr(group[feature], group[target])
                if not np.isnan(rho):
                    per_seq_rhos.append(rho)

        within_mean = np.mean(per_seq_rhos) if per_seq_rhos else 0.0
        within_median = np.median(per_seq_rhos) if per_seq_rhos else 0.0

        results.append({
            "feature": feature,
            "category": FEATURE_CATEGORIES.get(feature, "?"),
            "global_rho": round(global_rho, 4),
            "within_seq_rho_mean": round(within_mean, 4),
            "within_seq_rho_median": round(within_median, 4),
            "signal_drop": round(global_rho - within_mean, 4),
            "n_valid_seqs": len(per_seq_rhos),
        })

    return pd.DataFrame(results).sort_values(
        "within_seq_rho_mean", ascending=False, key=abs
    )


def compute_delta_correlations(df, feature_cols, target="recoverable_fn"):
    """Compute Spearman of delta_feature vs delta_target (frame-to-frame diffs)."""
    df_work = df.copy()

    # Compute deltas
    for feature in feature_cols:
        if feature in df_work.columns:
            df_work[f"delta_{feature}"] = df_work.groupby("clip")[feature].diff()
    df_work[f"delta_{target}"] = df_work.groupby("clip")[target].diff()
    df_clean = df_work.dropna(subset=[f"delta_{target}"])

    results = []
    for feature in feature_cols:
        delta_col = f"delta_{feature}"
        delta_target = f"delta_{target}"

        if delta_col not in df_clean.columns:
            continue

        if df_clean[delta_col].std() > 0 and df_clean[delta_target].std() > 0:
            global_rho, _ = spearmanr(df_clean[delta_col], df_clean[delta_target])
        else:
            global_rho = 0.0

        per_seq_rhos = []
        for _, group in df_clean.groupby("clip"):
            if len(group) < 10:
                continue
            if group[delta_col].std() > 0 and group[delta_target].std() > 0:
                rho, _ = spearmanr(group[delta_col], group[delta_target])
                if not np.isnan(rho):
                    per_seq_rhos.append(rho)

        within_rho = np.mean(per_seq_rhos) if per_seq_rhos else 0.0

        results.append({
            "feature": f"delta_{feature}",
            "category": FEATURE_CATEGORIES.get(feature, "?"),
            "global_rho": round(global_rho, 4),
            "within_seq_rho_mean": round(within_rho, 4),
            "n_valid_seqs": len(per_seq_rhos),
        })

    return pd.DataFrame(results).sort_values(
        "within_seq_rho_mean", ascending=False, key=abs
    )


def run_elastic_net(df, feature_cols, target="recoverable_fn"):
    """Run Elastic Net with per-sequence normalization.

    Returns dict with keys: alpha, l1_ratio, r2, coef_df
    """
    try:
        from sklearn.linear_model import ElasticNetCV
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("WARNING: scikit-learn not installed. Skipping Elastic Net.", flush=True)
        return None

    # Only use features that exist and have variance
    valid_feats = [f for f in feature_cols if f in df.columns and df[f].std() > 0]

    # Per-sequence normalization (subtract sequence means)
    df_norm = df.copy()
    for col in valid_feats + [target]:
        df_norm[col] = df_norm.groupby("clip")[col].transform(lambda x: x - x.mean())

    # Drop NaN rows
    df_norm = df_norm.dropna(subset=valid_feats + [target])

    X = df_norm[valid_feats].values
    y = df_norm[target].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ElasticNet with cross-validation
    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        n_alphas=50,
        cv=5,
        max_iter=10000,
        random_state=42,
    )
    enet.fit(X_scaled, y)

    r2 = enet.score(X_scaled, y)

    coef_df = pd.DataFrame({
        "feature": valid_feats,
        "category": [FEATURE_CATEGORIES.get(f, "?") for f in valid_feats],
        "coefficient": enet.coef_,
    }).sort_values("coefficient", key=abs, ascending=False)

    return {
        "alpha": enet.alpha_,
        "l1_ratio": enet.l1_ratio_,
        "r2": r2,
        "coef_df": coef_df,
        "n_nonzero": int(np.sum(enet.coef_ != 0)),
    }


# ── PDF report ────────────────────────────────────────────────────

def generate_pdf(df, corr_recov_df, corr_fn_df, delta_df, enet_result):
    """Generate 6-page validation report PDF."""

    with PdfPages(str(PDF_PATH)) as pdf:

        # ── Page 1: Within-seq rho bar chart (recoverable_fn) ─────
        fig, ax = plt.subplots(figsize=(12, 14))

        sorted_feats = corr_recov_df.sort_values("within_seq_rho_mean")
        y_pos = np.arange(len(sorted_feats))
        colors = []
        for _, row in sorted_feats.iterrows():
            rho = abs(row["within_seq_rho_mean"])
            if rho >= 0.3:
                colors.append("#2ecc71")
            elif rho >= 0.2:
                colors.append("#f39c12")
            else:
                colors.append("#e74c3c")

        bars = ax.barh(y_pos, sorted_feats["within_seq_rho_mean"],
                       color=colors, alpha=0.85, height=0.7)

        # Add category labels
        labels = [f"[{row['category']}] {row['feature']}"
                  for _, row in sorted_feats.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Mean within-sequence Spearman rho", fontsize=10)
        ax.axvline(0.3, color="green", ls="--", lw=0.8, label="rho=0.3 target")
        ax.axvline(-0.3, color="green", ls="--", lw=0.8)
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.legend(fontsize=9, loc="lower right")

        # Success verdict
        n_above_03 = (corr_recov_df["within_seq_rho_mean"].abs() >= 0.3).sum()
        if n_above_03 >= 5:
            verdict = f"SUCCESS: {n_above_03} features with |rho| >= 0.3"
            color = "green"
        elif n_above_03 > 0:
            verdict = f"MARGINAL: {n_above_03} features with |rho| >= 0.3"
            color = "orange"
        else:
            max_rho = corr_recov_df["within_seq_rho_mean"].abs().max()
            verdict = f"WEAK: max |rho| = {max_rho:.3f}"
            color = "red"

        fig.suptitle(f"Page 1: Within-Seq rho vs recoverable_fn -- {verdict}",
                     fontsize=13, fontweight="bold", color=color)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 2: Global vs within-seq scatter (colored by category) ─
        fig, ax = plt.subplots(figsize=(12, 10))

        for _, row in corr_recov_df.iterrows():
            cat = row["category"]
            c = CATEGORY_COLORS.get(cat, "#333")
            ax.scatter(row["global_rho"], row["within_seq_rho_mean"],
                       color=c, s=60, zorder=3, alpha=0.8)
            ax.annotate(row["feature"], (row["global_rho"], row["within_seq_rho_mean"]),
                        fontsize=5, ha="center", va="bottom")

        # Legend for categories
        for cat, cname in CATEGORY_NAMES.items():
            ax.scatter([], [], color=CATEGORY_COLORS[cat], s=60,
                       label=f"{cat}: {cname}")

        lims = [
            min(corr_recov_df["global_rho"].min(),
                corr_recov_df["within_seq_rho_mean"].min()) - 0.05,
            max(corr_recov_df["global_rho"].max(),
                corr_recov_df["within_seq_rho_mean"].max()) + 0.05,
        ]
        ax.plot(lims, lims, "k--", lw=0.5, label="y=x (no drop)")
        ax.axhline(0.3, color="green", ls=":", lw=0.8, label="rho=0.3")
        ax.axhline(-0.3, color="green", ls=":", lw=0.8)
        ax.set_xlabel("Global rho (pooled)", fontsize=10)
        ax.set_ylabel("Mean within-sequence rho", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")

        fig.suptitle("Page 2: Global vs Within-Sequence rho (by category)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 3: Per-clip rho boxplots for top 15 features ─────
        top15 = corr_recov_df.head(15)
        n_feats = len(top15)
        n_cols = 5
        n_rows = (n_feats + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten()

        for idx, (_, row) in enumerate(top15.iterrows()):
            feature = row["feature"]
            ax = axes[idx]
            per_clip_rhos = []
            for _, group in df.groupby("clip"):
                if len(group) < 30:
                    continue
                target = "recoverable_fn"
                if group[target].std() > 0 and group[feature].std() > 0:
                    rho, _ = spearmanr(group[feature], group[target])
                    if not np.isnan(rho):
                        per_clip_rhos.append(rho)

            if per_clip_rhos:
                bp = ax.boxplot(per_clip_rhos, vert=True, widths=0.5,
                                patch_artist=True)
                median = np.median(per_clip_rhos)
                color = ("#2ecc71" if abs(median) >= 0.3 else
                         "#f39c12" if abs(median) >= 0.2 else "#e74c3c")
                bp["boxes"][0].set_facecolor(color)
                bp["boxes"][0].set_alpha(0.6)

            ax.axhline(0.3, color="green", ls="--", lw=0.5)
            ax.axhline(-0.3, color="green", ls="--", lw=0.5)
            ax.axhline(0, color="k", lw=0.3, ls=":")
            ax.set_title(f"[{row['category']}] {feature}", fontsize=7)
            ax.set_ylim(-1, 1)
            ax.tick_params(labelsize=6)
            ax.set_xticks([])

        # Hide unused axes
        for idx in range(n_feats, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Page 3: Per-Clip rho Distribution (top 15, vs recoverable_fn)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 4: Delta features ────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))

        # Left: top 20 delta features bar chart
        top20_delta = delta_df.head(20).sort_values("within_seq_rho_mean")
        y_pos = np.arange(len(top20_delta))
        colors_d = []
        for _, row in top20_delta.iterrows():
            rho = abs(row["within_seq_rho_mean"])
            colors_d.append("#2ecc71" if rho >= 0.3 else
                            "#f39c12" if rho >= 0.2 else "#e74c3c")
        ax1.barh(y_pos, top20_delta["within_seq_rho_mean"],
                 color=colors_d, alpha=0.85)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top20_delta["feature"], fontsize=7)
        ax1.set_xlabel("Mean within-sequence Spearman rho")
        ax1.set_title("Delta Features vs delta_recoverable_fn (top 20)")
        ax1.axvline(0.3, color="green", ls="--", lw=0.8)
        ax1.axvline(-0.3, color="green", ls="--", lw=0.8)
        ax1.axvline(0, color="k", lw=0.5, ls="--")

        # Right: original vs delta scatter
        orig_within = corr_recov_df.set_index("feature")["within_seq_rho_mean"]
        delta_within = delta_df.set_index("feature")["within_seq_rho_mean"]
        for feat in FEATURE_COLS:
            if feat in orig_within.index and f"delta_{feat}" in delta_within.index:
                ov = orig_within[feat]
                dv = delta_within[f"delta_{feat}"]
                cat = FEATURE_CATEGORIES.get(feat, "?")
                ax2.scatter(ov, dv, color=CATEGORY_COLORS.get(cat, "#333"),
                            s=40, zorder=3, alpha=0.8)
                ax2.annotate(feat, (ov, dv), fontsize=4, ha="center", va="bottom")

        lims = [-0.6, 0.6]
        ax2.plot(lims, lims, "k--", lw=0.5, label="y=x")
        ax2.set_xlabel("Original within-seq rho")
        ax2.set_ylabel("Delta within-seq rho")
        ax2.set_title("Original vs Delta Feature Correlation")
        ax2.legend(fontsize=8)

        fig.suptitle("Page 4: Delta Features (Frame-to-Frame Differences)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 5: Elastic Net ───────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))

        if enet_result is not None:
            coef_df = enet_result["coef_df"]
            nonzero = coef_df[coef_df["coefficient"] != 0].copy()

            if len(nonzero) > 0:
                nonzero_sorted = nonzero.sort_values("coefficient")
                y_pos = np.arange(len(nonzero_sorted))
                bar_colors = [CATEGORY_COLORS.get(row["category"], "#333")
                              for _, row in nonzero_sorted.iterrows()]
                ax1.barh(y_pos, nonzero_sorted["coefficient"],
                         color=bar_colors, alpha=0.85)
                labels_en = [f"[{row['category']}] {row['feature']}"
                             for _, row in nonzero_sorted.iterrows()]
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(labels_en, fontsize=7)
                ax1.set_xlabel("Elastic Net Coefficient")
                ax1.set_title(f"Non-zero Coefficients ({enet_result['n_nonzero']}/{len(coef_df)})")
                ax1.axvline(0, color="k", lw=0.5, ls="--")
            else:
                ax1.text(0.5, 0.5, "All coefficients are zero",
                         ha="center", va="center", transform=ax1.transAxes, fontsize=14)
                ax1.set_title("Elastic Net Coefficients")

            # Right: text summary
            ax2.axis("off")
            summary_text = (
                f"Elastic Net Results\n"
                f"{'=' * 40}\n\n"
                f"Best alpha:    {enet_result['alpha']:.6f}\n"
                f"Best l1_ratio: {enet_result['l1_ratio']:.2f}\n"
                f"R-squared:     {enet_result['r2']:.4f}\n\n"
                f"Non-zero features: {enet_result['n_nonzero']} / {len(coef_df)}\n\n"
                f"Top features by |coefficient|:\n"
            )
            top_feats = nonzero.head(15) if len(nonzero) > 0 else pd.DataFrame()
            for _, row in top_feats.iterrows():
                summary_text += f"  [{row['category']}] {row['feature']:30s}  {row['coefficient']:+.4f}\n"

            ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
                     fontsize=10, verticalalignment="top", fontfamily="monospace")
        else:
            ax1.text(0.5, 0.5, "scikit-learn not available",
                     ha="center", va="center", transform=ax1.transAxes, fontsize=14)
            ax2.axis("off")
            ax2.text(0.5, 0.5, "Install scikit-learn:\npip install scikit-learn",
                     ha="center", va="center", transform=ax2.transAxes, fontsize=14)

        fig.suptitle("Page 5: Elastic Net Feature Selection",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 6: recoverable_fn distribution + target comparison ─
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # (0,0) recoverable_fn histogram
        ax = axes[0, 0]
        rfn = df["recoverable_fn"]
        ax.hist(rfn, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(rfn.mean(), color="red", ls="--", lw=1, label=f"mean={rfn.mean():.2f}")
        ax.set_xlabel("recoverable_fn (tp_medium - tp_nano)")
        ax.set_ylabel("Count")
        ax.set_title(f"recoverable_fn Distribution (zero_pct={100*(rfn==0).mean():.1f}%)")
        ax.legend(fontsize=9)

        # (0,1) Per-clip recoverable_fn means
        ax = axes[0, 1]
        clip_means = df.groupby("clip")["recoverable_fn"].mean().sort_values()
        clip_stds = df.groupby("clip")["recoverable_fn"].std().reindex(clip_means.index)
        x_pos = np.arange(len(clip_means))
        ax.bar(x_pos, clip_means.values, yerr=clip_stds.values,
               color="steelblue", alpha=0.7, capsize=2)
        ax.set_xlabel("Clip (sorted by mean)")
        ax.set_ylabel("recoverable_fn")
        ax.set_title("Mean recoverable_fn per Clip (+/- std)")
        ax.set_xticks([])

        # (1,0) Recall comparison
        ax = axes[1, 0]
        valid = df[df["num_gt_coco"] > 0].copy()
        valid["recall_nano"] = valid["tp_nano"] / valid["num_gt_coco"]
        valid["recall_medium"] = valid["tp_medium"] / valid["num_gt_coco"]
        mean_nano = valid["recall_nano"].mean()
        mean_medium = valid["recall_medium"].mean()
        bars = ax.bar(["Nano@320", "Medium@640"], [mean_nano, mean_medium],
                      color=["#e74c3c", "#2ecc71"], alpha=0.8)
        ax.set_ylabel("Mean Recall (COCO-matchable)")
        ax.set_title(f"Recall: Nano={mean_nano:.3f}, Medium={mean_medium:.3f}, "
                     f"Gap={mean_medium - mean_nano:.3f}")
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, [mean_nano, mean_medium]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=10)

        # (1,1) Stats table
        ax = axes[1, 1]
        ax.axis("off")
        stats_text = (
            f"recoverable_fn Statistics\n"
            f"{'=' * 35}\n\n"
            f"Total frames:   {len(df):,}\n"
            f"Total clips:    {df['clip'].nunique()}\n\n"
            f"Mean:           {rfn.mean():.3f}\n"
            f"Std:            {rfn.std():.3f}\n"
            f"Min:            {rfn.min():.0f}\n"
            f"Max:            {rfn.max():.0f}\n\n"
            f"Percentiles:\n"
            f"  25th:         {rfn.quantile(0.25):.0f}\n"
            f"  50th:         {rfn.quantile(0.50):.0f}\n"
            f"  75th:         {rfn.quantile(0.75):.0f}\n"
            f"  90th:         {rfn.quantile(0.90):.0f}\n"
            f"  95th:         {rfn.quantile(0.95):.0f}\n\n"
            f"Zero fraction:  {100*(rfn==0).mean():.1f}%\n\n"
        )

        # Dual-target comparison summary
        recov_top = corr_recov_df.head(5)
        fn_top = corr_fn_df.head(5)
        stats_text += (
            f"Target Comparison (top 5 |rho|):\n"
            f"{'=' * 35}\n"
            f"recoverable_fn:  {', '.join(f'{r:.3f}' for r in recov_top['within_seq_rho_mean'])}\n"
            f"fn_coco_nano:    {', '.join(f'{r:.3f}' for r in fn_top['within_seq_rho_mean'])}\n"
        )

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace")

        fig.suptitle("Page 6: recoverable_fn Distribution & Statistics",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"PDF saved -> {PDF_PATH}", flush=True)


# ── Main ──────────────────────────────────────────────────────────

def main():
    if not FEAT_CSV.exists():
        print(f"ERROR: feature CSV not found at {FEAT_CSV}", flush=True)
        print("Run meteor_02_extract_features.py first.", flush=True)
        return

    print("Loading features...", flush=True)
    df = pd.read_csv(FEAT_CSV)
    print(f"  {len(df):,} frames, {df['clip'].nunique()} clips\n", flush=True)

    # Filter to frames with at least 1 GT object
    df = df[df["num_gt"] >= 1].reset_index(drop=True)
    print(f"  After filtering (num_gt >= 1): {len(df):,} frames\n", flush=True)

    # Only use features that exist in CSV
    feature_cols = [f for f in FEATURE_COLS if f in df.columns]
    print(f"  Features available: {len(feature_cols)}/{len(FEATURE_COLS)}\n", flush=True)

    # ── recoverable_fn distribution ─────────────────────────────
    print("=" * 60)
    print("RECOVERABLE_FN DISTRIBUTION")
    print("=" * 60)
    rfn = df["recoverable_fn"]
    print(f"  mean={rfn.mean():.3f}, std={rfn.std():.3f}")
    print(f"  percentiles: 25th={rfn.quantile(0.25):.0f}, "
          f"50th={rfn.quantile(0.50):.0f}, "
          f"75th={rfn.quantile(0.75):.0f}, "
          f"90th={rfn.quantile(0.90):.0f}, "
          f"95th={rfn.quantile(0.95):.0f}")
    print(f"  zero fraction: {100*(rfn==0).mean():.1f}%\n")

    # ── Correlation vs recoverable_fn (primary target) ──────────
    print("=" * 60)
    print("WITHIN-SEQUENCE CORRELATION vs recoverable_fn (PRIMARY)")
    print("=" * 60)
    corr_recov_df = compute_correlations(df, feature_cols, target="recoverable_fn")
    print(corr_recov_df.to_string(index=False))

    n_above_03 = (corr_recov_df["within_seq_rho_mean"].abs() >= 0.3).sum()
    max_within = corr_recov_df["within_seq_rho_mean"].abs().max()
    print(f"\nFeatures with |within_seq_rho| >= 0.3: {n_above_03}")
    print(f"Max |within_seq_rho|: {max_within:.4f}")

    if n_above_03 >= 5:
        print("\n>>> SUCCESS: METEOR works! Proceed with full pipeline.")
    elif n_above_03 > 0:
        print("\n>>> MARGINAL: Some features pass threshold. Try delta features.")
    else:
        print("\n>>> WEAK: Moving camera didn't help enough.")

    # ── Correlation vs fn_coco_nano (secondary, for comparison) ─
    print(f"\n{'=' * 60}")
    print("WITHIN-SEQUENCE CORRELATION vs fn_coco_nano (COMPARISON)")
    print("=" * 60)
    corr_fn_df = compute_correlations(df, feature_cols, target="fn_coco_nano")
    print(corr_fn_df.to_string(index=False))

    # ── Side-by-side comparison ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("TARGET COMPARISON: recoverable_fn vs fn_coco_nano")
    print("=" * 60)
    compare = corr_recov_df[["feature", "category", "within_seq_rho_mean"]].copy()
    compare = compare.rename(columns={"within_seq_rho_mean": "rho_recoverable"})
    fn_rhos = corr_fn_df.set_index("feature")["within_seq_rho_mean"]
    compare["rho_fn_coco"] = compare["feature"].map(fn_rhos)
    compare["rho_diff"] = compare["rho_recoverable"] - compare["rho_fn_coco"].fillna(0)
    compare = compare.sort_values("rho_recoverable", ascending=False, key=abs)
    print(compare.to_string(index=False))

    # ── Delta features ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("DELTA FEATURES vs delta_recoverable_fn")
    print("=" * 60)
    delta_df = compute_delta_correlations(df, feature_cols, target="recoverable_fn")
    print(delta_df.to_string(index=False))

    # ── Elastic Net ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("ELASTIC NET FEATURE SELECTION")
    print("=" * 60)
    enet_result = run_elastic_net(df, feature_cols, target="recoverable_fn")
    if enet_result is not None:
        print(f"  Best alpha:    {enet_result['alpha']:.6f}")
        print(f"  Best l1_ratio: {enet_result['l1_ratio']:.2f}")
        print(f"  R-squared:     {enet_result['r2']:.4f}")
        print(f"  Non-zero features: {enet_result['n_nonzero']}")
        print()
        nonzero = enet_result["coef_df"][enet_result["coef_df"]["coefficient"] != 0]
        print(nonzero.to_string(index=False))

    # ── Save correlation results ──────────────────────────────────
    corr_recov_df.to_csv(CORR_CSV, index=False)
    print(f"\nCorrelation results saved -> {CORR_CSV}")

    # ── Generate PDF ──────────────────────────────────────────────
    print("\nGenerating PDF report...", flush=True)
    generate_pdf(df, corr_recov_df, corr_fn_df, delta_df, enet_result)

    print("\nDone.")


if __name__ == "__main__":
    main()
