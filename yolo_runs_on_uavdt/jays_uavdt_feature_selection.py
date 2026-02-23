"""
UAVDT Feature Selection Analysis
Builds analysis dataset, runs Spearman correlation, mutual information,
and redundancy analysis. Outputs summary CSV and 7-page PDF.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
FEATURES_CSV = ROOT / "jays_uavdt_frame_features.csv"
RESULTS_CSV = ROOT / "uavdt_results.csv"
ANALYSIS_CSV = ROOT / "jays_uavdt_analysis_dataset.csv"
SUMMARY_CSV = ROOT / "jays_uavdt_correlation_summary.csv"
OUTPUT_PDF = ROOT / "jays_uavdt_correlation_results.pdf"

FEATURE_NAMES = [
    "mean_brightness", "brightness_std", "brightness_skewness",
    "brightness_kurtosis", "overexposed_ratio", "underexposed_ratio",
    "dynamic_range", "backlight_ratio",
    "weber_contrast", "michelson_contrast", "rms_contrast", "local_contrast_mean",
    "dark_channel_mean", "dark_channel_std", "transmission_estimate",
    "atmospheric_light",
    "laplacian_variance", "tenengrad", "brenner_gradient",
    "high_freq_energy_ratio",
    "edge_density", "gradient_magnitude_mean", "gradient_magnitude_std",
    "GLCM_contrast", "GLCM_homogeneity", "GLCM_energy", "GLCM_entropy",
    "GLCM_correlation", "LBP_entropy", "LBP_uniformity",
    "spatial_entropy", "color_entropy", "local_entropy_mean", "local_entropy_std",
    "colorfulness", "mean_saturation", "saturation_std", "hue_std",
    "dominant_color_ratio", "color_channel_corr",
    "spectral_centroid", "spectral_rolloff", "wavelet_detail_energy",
    "FAST_keypoint_count", "connected_components_count", "foreground_ratio",
    "optical_flow_mag_mean", "optical_flow_mag_std", "optical_flow_angle_var",
    "frame_difference_mean", "SSIM_prev_frame", "histogram_intersection",
    "brightness_delta", "edge_density_delta", "flow_coherence", "scene_cut_score",
]


# ── Step 1: Build Analysis Dataset ──────────────────────────────

def build_analysis_dataset():
    """Merge features with YOLO results, compute recall_gap."""
    if ANALYSIS_CSV.exists():
        print(f"Loading cached {ANALYSIS_CSV}")
        return pd.read_csv(ANALYSIS_CSV)

    print("Building analysis dataset...")
    feat = pd.read_csv(FEATURES_CSV)
    res = pd.read_csv(RESULTS_CSV)

    # Get nano recall
    nano = res[res["model"] == "yolo11n.pt"][
        ["sequence", "frame_id", "recall", "weather"]
    ].copy()
    nano = nano.rename(columns={"recall": "recall_nano"})

    # Get medium recall
    medium = res[res["model"] == "yolo11m.pt"][
        ["sequence", "frame_id", "recall"]
    ].copy()
    medium = medium.rename(columns={"recall": "recall_medium"})

    # Merge nano + medium
    merged = nano.merge(medium, on=["sequence", "frame_id"])
    merged["recall_gap"] = merged["recall_medium"] - merged["recall_nano"]

    # Merge with features
    df = feat.merge(merged, on=["sequence", "frame_id"])

    df.to_csv(ANALYSIS_CSV, index=False)
    print(f"Saved {ANALYSIS_CSV} ({len(df)} rows)")
    return df


# ── Step 2: Spearman Correlation ─────────────────────────────────

def compute_spearman(df, features, target):
    """Compute Spearman rho and p-value for each feature vs target."""
    results = []
    for feat in features:
        valid = df[[feat, target]].dropna()
        if len(valid) < 30:
            results.append((feat, np.nan, np.nan))
            continue
        rho, pval = spearmanr(valid[feat], valid[target])
        results.append((feat, rho, pval))
    return results


def spearman_by_weather(df, features, target="recall_gap"):
    """Compute Spearman broken down by weather condition."""
    weather_results = {}
    for weather in ["daylight", "night", "fog"]:
        sub = df[df["weather"] == weather]
        if len(sub) < 30:
            weather_results[weather] = {f: np.nan for f in features}
            continue
        rhos = {}
        for feat in features:
            valid = sub[[feat, target]].dropna()
            if len(valid) < 30:
                rhos[feat] = np.nan
            else:
                rho, _ = spearmanr(valid[feat], valid[target])
                rhos[feat] = rho
        weather_results[weather] = rhos
    return weather_results


# ── Step 3: Mutual Information ───────────────────────────────────

def compute_mutual_info(df, features, target="recall_gap"):
    """Compute mutual information regression scores."""
    valid = df[features + [target]].dropna()
    X = valid[features].values
    y = valid[target].values
    mi = mutual_info_regression(X, y, n_neighbors=5, random_state=42)
    return dict(zip(features, mi))


# ── Step 4: Redundancy Analysis ──────────────────────────────────

def compute_redundancy(df, features, rho_gap_dict, threshold=0.85):
    """
    Pairwise |Spearman rho| between features.
    For pairs > threshold, flag the one with lower |rho(gap)| as redundant.
    """
    valid = df[features].dropna()
    n = len(features)
    corr_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i, n):
            if i == j:
                corr_matrix[i, j] = 1.0
                continue
            pair = valid[[features[i], features[j]]].dropna()
            if len(pair) < 30:
                continue
            rho, _ = spearmanr(pair[features[i]], pair[features[j]])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    # Find redundant pairs
    redundant = {}
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) > threshold:
                fi, fj = features[i], features[j]
                ri = abs(rho_gap_dict.get(fi, 0) or 0)
                rj = abs(rho_gap_dict.get(fj, 0) or 0)
                if ri < rj:
                    redundant[fi] = fj
                else:
                    redundant[fj] = fi

    return corr_matrix, redundant


# ── Visualization (7-page PDF) ───────────────────────────────────

def create_pdf(summary_df, corr_matrix, features, weather_rhos, df_analysis):
    """Generate 7-page PDF with all analysis results."""
    with PdfPages(str(OUTPUT_PDF)) as pdf:
        # ── Page 1: Bar chart — features ranked by |Spearman rho| with recall_gap
        fig, ax = plt.subplots(figsize=(14, 12))
        sorted_df = summary_df.sort_values("abs_rho", ascending=True)
        colors = ["#d32f2f" if r < 0 else "#1976d2"
                  for r in sorted_df["spearman_rho"]]
        ax.barh(range(len(sorted_df)), sorted_df["spearman_rho"], color=colors)
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df["feature"], fontsize=7)
        ax.set_xlabel("Spearman rho with recall_gap")
        ax.set_title("All 56 Features Ranked by Spearman Correlation with Recall Gap")
        ax.axvline(x=0, color="black", linewidth=0.5)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # ── Page 2: Bar chart — features ranked by mutual information
        fig, ax = plt.subplots(figsize=(14, 12))
        sorted_mi = summary_df.sort_values("mutual_info", ascending=True)
        ax.barh(range(len(sorted_mi)), sorted_mi["mutual_info"], color="#7b1fa2")
        ax.set_yticks(range(len(sorted_mi)))
        ax.set_yticklabels(sorted_mi["feature"], fontsize=7)
        ax.set_xlabel("Mutual Information Score")
        ax.set_title("All 56 Features Ranked by Mutual Information with Recall Gap")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # ── Page 3: Heatmap — 56x56 pairwise feature correlations (clustered)
        fig, ax = plt.subplots(figsize=(16, 14))
        corr_df = pd.DataFrame(corr_matrix, index=features, columns=features)
        # Cluster with seaborn
        g = sns.clustermap(corr_df, cmap="RdBu_r", vmin=-1, vmax=1,
                          figsize=(16, 14), dendrogram_ratio=0.1,
                          xticklabels=True, yticklabels=True)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=5, rotation=90)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=5)
        g.fig.suptitle("Pairwise Feature Correlations (Clustered)", y=1.01, fontsize=14)
        pdf.savefig(g.fig)
        plt.close("all")

        # ── Page 4: Scatter plots — top 6 features vs recall_gap (2x3)
        top6 = summary_df.nlargest(6, "abs_rho")["feature"].values
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for idx, feat_name in enumerate(top6):
            ax = axes[idx // 3, idx % 3]
            valid = df_analysis[[feat_name, "recall_gap"]].dropna()
            # Subsample if too many points
            if len(valid) > 3000:
                valid = valid.sample(3000, random_state=42)
            ax.scatter(valid[feat_name], valid["recall_gap"],
                      alpha=0.15, s=3, color="#1565c0")
            rho = summary_df[summary_df["feature"] == feat_name]["spearman_rho"].values[0]
            ax.set_title(f"{feat_name}\nrho={rho:.3f}", fontsize=9)
            ax.set_xlabel(feat_name, fontsize=7)
            ax.set_ylabel("recall_gap", fontsize=7)
        plt.suptitle("Top 6 Features vs Recall Gap", fontsize=14)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # ── Page 5: Grouped bar chart — Spearman rho by weather for top 15
        top15 = summary_df.nlargest(15, "abs_rho")["feature"].values
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(top15))
        width = 0.25
        for i, (weather, color) in enumerate([
            ("daylight", "#ffa726"), ("night", "#5c6bc0"), ("fog", "#78909c")
        ]):
            vals = [weather_rhos.get(weather, {}).get(f, 0) or 0 for f in top15]
            ax.bar(x + i * width, vals, width, label=weather, color=color)
        ax.set_xticks(x + width)
        ax.set_xticklabels(top15, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Spearman rho")
        ax.set_title("Spearman Correlation by Weather Condition (Top 15 Features)")
        ax.legend()
        ax.axhline(y=0, color="black", linewidth=0.5)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # ── Page 6: Table — recommended feature subset
        kept = summary_df[summary_df["keep"] == True].copy()
        fig, ax = plt.subplots(figsize=(14, max(6, len(kept) * 0.35 + 2)))
        ax.axis("off")
        if len(kept) > 0:
            table_data = kept[["feature", "spearman_rho", "mutual_info",
                              "redundant_with"]].copy()
            table_data["spearman_rho"] = table_data["spearman_rho"].apply(lambda x: f"{x:.4f}")
            table_data["mutual_info"] = table_data["mutual_info"].apply(lambda x: f"{x:.4f}")
            table_data = table_data.fillna("")
            table = ax.table(
                cellText=table_data.values,
                colLabels=["Feature", "Spearman rho", "Mutual Info", "Redundant With"],
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.auto_set_column_width(col=list(range(4)))
            table.scale(1, 1.4)
        ax.set_title(
            f"Recommended Feature Subset ({len(kept)} features)\n"
            "Filters: |rho| > 0.10, MI > median, not redundant",
            fontsize=12, pad=20,
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # ── Page 7: Side-by-side bars — rho_nano vs rho_medium vs rho_gap for top 15
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(top15))
        width = 0.25

        rho_nano_vals = [
            summary_df[summary_df["feature"] == f]["rho_nano"].values[0] or 0
            for f in top15
        ]
        rho_medium_vals = [
            summary_df[summary_df["feature"] == f]["rho_medium"].values[0] or 0
            for f in top15
        ]
        rho_gap_vals = [
            summary_df[summary_df["feature"] == f]["spearman_rho"].values[0] or 0
            for f in top15
        ]

        ax.bar(x - width, rho_nano_vals, width, label="rho with recall_nano",
               color="#ef5350")
        ax.bar(x, rho_medium_vals, width, label="rho with recall_medium",
               color="#42a5f5")
        ax.bar(x + width, rho_gap_vals, width, label="rho with recall_gap",
               color="#66bb6a")

        # Highlight features where |rho_nano| >> |rho_medium|
        for i, f in enumerate(top15):
            rn = abs(rho_nano_vals[i])
            rm = abs(rho_medium_vals[i])
            if rn > rm * 1.5 and rn > 0.05:
                ax.annotate("*", (x[i] - width, rho_nano_vals[i]),
                           fontsize=14, ha="center", va="bottom", color="red",
                           fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(top15, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Spearman rho")
        ax.set_title(
            "Independent Model Correlations — Top 15 Features\n"
            "* = |rho_nano| >> |rho_medium| (most valuable for switching)"
        )
        ax.legend()
        ax.axhline(y=0, color="black", linewidth=0.5)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"Saved {OUTPUT_PDF}")


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("UAVDT Feature Selection Analysis")
    print("=" * 60)

    # Step 1: Build analysis dataset
    df = build_analysis_dataset()
    print(f"Analysis dataset: {len(df)} rows, weather distribution:")
    print(df["weather"].value_counts().to_string())
    print()

    # Determine which features are available (non-NaN in >50% of rows)
    available_features = []
    for f in FEATURE_NAMES:
        if f in df.columns and df[f].notna().mean() > 0.5:
            available_features.append(f)
    print(f"Available features: {len(available_features)} / {len(FEATURE_NAMES)}")

    # Step 2: Spearman correlation
    print("\nStep 2: Spearman correlations...")
    spearman_gap = compute_spearman(df, available_features, "recall_gap")
    spearman_nano = compute_spearman(df, available_features, "recall_nano")
    spearman_medium = compute_spearman(df, available_features, "recall_medium")

    rho_gap_dict = {f: r for f, r, _ in spearman_gap}
    rho_nano_dict = {f: r for f, r, _ in spearman_nano}
    rho_medium_dict = {f: r for f, r, _ in spearman_medium}

    weather_rhos = spearman_by_weather(df, available_features)

    # Step 3: Mutual information
    print("Step 3: Mutual information...")
    mi_dict = compute_mutual_info(df, available_features)
    mi_median = np.median(list(mi_dict.values()))

    # Step 4: Redundancy analysis
    print("Step 4: Redundancy analysis...")
    corr_matrix, redundant_dict = compute_redundancy(
        df, available_features, rho_gap_dict
    )

    # Build summary DataFrame
    print("\nBuilding summary...")
    rows = []
    for feat in FEATURE_NAMES:
        rho = rho_gap_dict.get(feat, np.nan)
        pval = dict([(f, p) for f, _, p in spearman_gap]).get(feat, np.nan)
        rho_n = rho_nano_dict.get(feat, np.nan)
        rho_m = rho_medium_dict.get(feat, np.nan)
        mi = mi_dict.get(feat, np.nan)
        redundant_with = redundant_dict.get(feat, "")

        # Keep if: |rho| > 0.10, MI > median, not redundant
        abs_rho = abs(rho) if not np.isnan(rho) else 0
        keep = (
            abs_rho > 0.10
            and (mi if not np.isnan(mi) else 0) > mi_median
            and feat not in redundant_dict
            and feat in available_features
        )

        rows.append({
            "feature": feat,
            "spearman_rho": rho,
            "spearman_pval": pval,
            "rho_nano": rho_n,
            "rho_medium": rho_m,
            "mutual_info": mi,
            "redundant_with": redundant_with,
            "keep": keep,
        })

    summary_df = pd.DataFrame(rows)
    summary_df["abs_rho"] = summary_df["spearman_rho"].abs()
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved {SUMMARY_CSV}")

    # Print top 15
    print("\nTop 15 features by |Spearman rho| with recall_gap:")
    print("-" * 70)
    top15_df = summary_df.nlargest(15, "abs_rho")
    for _, row in top15_df.iterrows():
        marker = " *KEEP*" if row["keep"] else ""
        print(f"  {row['feature']:35s}  rho={row['spearman_rho']:+.4f}  "
              f"MI={row['mutual_info']:.4f}  rho_n={row['rho_nano']:+.4f}  "
              f"rho_m={row['rho_medium']:+.4f}{marker}")

    kept_count = summary_df["keep"].sum()
    print(f"\nRecommended feature subset: {kept_count} features")
    print(f"MI median threshold: {mi_median:.4f}")

    # Generate 7-page PDF
    print("\nGenerating PDF...")
    create_pdf(summary_df, corr_matrix, available_features, weather_rhos, df)

    print("\nDone!")
