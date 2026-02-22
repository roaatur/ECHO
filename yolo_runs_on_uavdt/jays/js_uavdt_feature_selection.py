"""
UAVDT Feature Selection Analysis (v2 — RF + LASSO)
Builds analysis dataset, runs Spearman correlation, Random Forest
permutation importance (primary), LASSO validation (secondary),
and redundancy analysis. Outputs summary CSV and 8-page PDF.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # echo/
UAVDT_DIR = ROOT / "yolo_runs_on_uavdt"
FEATURES_CSV = UAVDT_DIR / "jays_uavdt_frame_features.csv"
RESULTS_CSV = UAVDT_DIR / "uavdt_results.csv"

OUT_DIR = Path(__file__).resolve().parent               # echo/jays/
ANALYSIS_CSV = OUT_DIR / "js_uavdt_analysis_dataset.csv"
SUMMARY_CSV = OUT_DIR / "js_uavdt_correlation_summary.csv"
OUTPUT_PDF = OUT_DIR / "js_uavdt_correlation_results.pdf"

FEATURE_NAMES = [
    # A. Illumination (1-8)
    "mean_brightness", "brightness_std", "brightness_skewness",
    "brightness_kurtosis", "overexposed_ratio", "underexposed_ratio",
    "dynamic_range", "backlight_ratio",
    # B. Contrast (9-12)
    "weber_contrast", "michelson_contrast", "rms_contrast", "local_contrast_mean",
    # C. Haze/Fog (13-16)
    "dark_channel_mean", "dark_channel_std", "transmission_estimate",
    "atmospheric_light",
    # D. Sharpness/Blur (17-20)
    "laplacian_variance", "tenengrad", "brenner_gradient",
    "high_freq_energy_ratio",
    # E. Texture/Complexity (21-30)
    "edge_density", "gradient_magnitude_mean", "gradient_magnitude_std",
    "GLCM_contrast", "GLCM_homogeneity", "GLCM_energy", "GLCM_entropy",
    "GLCM_correlation", "LBP_entropy", "LBP_uniformity",
    # F. Entropy (31-34)
    "spatial_entropy", "color_entropy", "local_entropy_mean", "local_entropy_std",
    # G. Color (35-40)
    "colorfulness", "mean_saturation", "saturation_std", "hue_std",
    "dominant_color_ratio", "color_channel_corr",
    # H. Frequency Domain (41-43)
    "spectral_centroid", "spectral_rolloff", "wavelet_detail_energy",
    # I. Spatial Structure (44-46)
    "FAST_keypoint_count", "connected_components_count", "foreground_ratio",
    # J. Motion/Temporal (47-56)
    "optical_flow_mag_mean", "optical_flow_mag_std", "optical_flow_angle_var",
    "frame_difference_mean", "SSIM_prev_frame", "histogram_intersection",
    "brightness_delta", "edge_density_delta", "flow_coherence", "scene_cut_score",
]

assert len(FEATURE_NAMES) == 56

# Feature categories
FEATURE_CATEGORIES = {}
_cat_ranges = [
    ("illumination", FEATURE_NAMES[0:8]),
    ("contrast",     FEATURE_NAMES[8:12]),
    ("haze",         FEATURE_NAMES[12:16]),
    ("sharpness",    FEATURE_NAMES[16:20]),
    ("texture",      FEATURE_NAMES[20:30]),
    ("entropy",      FEATURE_NAMES[30:34]),
    ("color",        FEATURE_NAMES[34:40]),
    ("frequency",    FEATURE_NAMES[40:43]),
    ("spatial",      FEATURE_NAMES[43:46]),
    ("temporal",     FEATURE_NAMES[46:56]),
]
for _cat, _feats in _cat_ranges:
    for _f in _feats:
        FEATURE_CATEGORIES[_f] = _cat

STATIC_FEATURES = FEATURE_NAMES[:46]
TEMPORAL_FEATURES = FEATURE_NAMES[46:]


# ── Step 1: Build Analysis Dataset ──────────────────────────────

def build_analysis_dataset():
    """Merge features with YOLO results, compute recall_gap."""
    if ANALYSIS_CSV.exists():
        print(f"Loading cached {ANALYSIS_CSV}", flush=True)
        return pd.read_csv(ANALYSIS_CSV)

    print("Building analysis dataset...", flush=True)
    feat = pd.read_csv(FEATURES_CSV)
    res = pd.read_csv(RESULTS_CSV)

    nano = res[res["model"] == "yolo11n.pt"][
        ["sequence", "frame_id", "recall", "weather"]
    ].copy()
    nano = nano.rename(columns={"recall": "recall_nano"})

    medium = res[res["model"] == "yolo11m.pt"][
        ["sequence", "frame_id", "recall"]
    ].copy()
    medium = medium.rename(columns={"recall": "recall_medium"})

    merged = nano.merge(medium, on=["sequence", "frame_id"])
    merged["recall_gap"] = merged["recall_medium"] - merged["recall_nano"]

    df = feat.merge(merged, on=["sequence", "frame_id"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ANALYSIS_CSV, index=False)
    print(f"Saved {ANALYSIS_CSV} ({len(df)} rows)", flush=True)
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


# ── Step 3: Random Forest Permutation Importance ─────────────────

def compute_rf_importance(df, features, target="recall_gap"):
    """RF permutation importance for static features predicting recall_gap."""
    valid = df[features + [target]].dropna()
    X = valid[features].values
    y = valid[target].values

    print(f"  RF: fitting on {len(valid)} rows, {len(features)} features...", flush=True)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    print(f"  RF: computing permutation importance (10 repeats)...", flush=True)
    perm = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)

    return dict(zip(features, perm.importances_mean))


# ── Step 4: LASSO Validation ─────────────────────────────────────

def compute_lasso(df, features, target="recall_gap"):
    """LassoCV on static features predicting recall_gap."""
    valid = df[features + [target]].dropna()
    X = valid[features].values
    y = valid[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  LASSO: fitting LassoCV on {len(valid)} rows...", flush=True)
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Objective did not converge",
                                category=UserWarning)
        lasso.fit(X_scaled, y)

    print(f"  LASSO: alpha={lasso.alpha_:.6f}, "
          f"non-zero coefficients: {np.sum(lasso.coef_ != 0)}/{len(features)}",
          flush=True)

    return dict(zip(features, lasso.coef_))


# ── Step 5: Redundancy Analysis ──────────────────────────────────

def compute_redundancy(df, features, rf_importance_dict, rho_gap_dict, threshold=0.85):
    """
    Pairwise |Spearman rho| between features.
    For pairs > threshold: flag the one with lower RF importance as redundant.
    Temporal-temporal pairs fall back to lower |rho_gap|.
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

    redundant = {}
    for i in range(n):
        for j in range(i + 1, n):
            if np.isnan(corr_matrix[i, j]):
                continue
            if abs(corr_matrix[i, j]) > threshold:
                fi, fj = features[i], features[j]
                ri = rf_importance_dict.get(fi, None)
                rj = rf_importance_dict.get(fj, None)

                if ri is not None and rj is not None:
                    if ri < rj:
                        redundant[fi] = fj
                    else:
                        redundant[fj] = fi
                else:
                    ri_rho = abs(rho_gap_dict.get(fi, 0) or 0)
                    rj_rho = abs(rho_gap_dict.get(fj, 0) or 0)
                    if ri_rho < rj_rho:
                        redundant[fi] = fj
                    else:
                        redundant[fj] = fi

    return corr_matrix, redundant


# ── Visualization (8-page PDF) ───────────────────────────────────

def create_pdf(summary_df, corr_matrix, features, weather_rhos, df_analysis,
               static_features):
    """Generate 8-page PDF with all analysis results."""
    with PdfPages(str(OUTPUT_PDF)) as pdf:

        # ── Page 1: All 56 features by |Spearman rho| with recall_gap
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

        # ── Page 2: 46 static features by RF permutation importance
        fig, ax = plt.subplots(figsize=(14, 12))
        static_df = summary_df[summary_df["feature"].isin(static_features)].copy()
        sorted_rf = static_df.sort_values("rf_importance", ascending=True)
        ax.barh(range(len(sorted_rf)), sorted_rf["rf_importance"], color="#00796b")
        ax.set_yticks(range(len(sorted_rf)))
        ax.set_yticklabels(sorted_rf["feature"], fontsize=7)
        ax.set_xlabel("Permutation Importance")
        ax.set_title("46 Static Features Ranked by RF Permutation Importance")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # ── Page 3: 56x56 pairwise feature correlations (clustered heatmap)
        corr_df = pd.DataFrame(corr_matrix, index=features, columns=features)
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

        # ── Page 5: Grouped bars — Spearman rho by weather for top 15
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

        # ── Page 6: Side-by-side bars — rho_nano / rho_medium / rho_gap for top 15
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

        # ── Page 7: RF importance vs |LASSO coef| scatter for 46 static features
        fig, ax = plt.subplots(figsize=(10, 8))
        static_sum = summary_df[summary_df["feature"].isin(static_features)].copy()
        x_vals = static_sum["rf_importance"].values
        y_vals = static_sum["lasso_coef"].abs().values
        names = static_sum["feature"].values

        ax.scatter(x_vals, y_vals, color="#1565c0", s=40, alpha=0.7)
        for i, name in enumerate(names):
            ax.annotate(name, (x_vals[i], y_vals[i]),
                        fontsize=5, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")

        ax.set_xlabel("RF Permutation Importance")
        ax.set_ylabel("|LASSO Coefficient|")
        ax.set_title("RF Importance vs |LASSO Coefficient| — 46 Static Features\n"
                      "Top-right = strong agreement, axes-only = method disagreement")
        ax.axvline(x=np.median(x_vals), color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=np.median(y_vals), color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # ── Page 8: Final recommended feature subset table
        kept = summary_df[summary_df["keep"]].copy()
        kept_static = kept[~kept["feature"].isin(TEMPORAL_FEATURES)]
        kept_temporal = kept[kept["feature"].isin(TEMPORAL_FEATURES)]

        fig, ax = plt.subplots(figsize=(16, max(6, len(kept) * 0.3 + 4)))
        ax.axis("off")

        if len(kept) > 0:
            display_cols = ["feature", "category", "spearman_rho", "rf_importance",
                            "lasso_coef", "keep_reason"]
            table_data = kept[display_cols].copy()
            table_data["spearman_rho"] = table_data["spearman_rho"].apply(
                lambda x: f"{x:.4f}")
            table_data["rf_importance"] = table_data["rf_importance"].apply(
                lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
            table_data["lasso_coef"] = table_data["lasso_coef"].apply(
                lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")

            table = ax.table(
                cellText=table_data.values,
                colLabels=["Feature", "Category", "Spearman rho", "RF Importance",
                            "LASSO Coef", "Keep Reason"],
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.auto_set_column_width(col=list(range(6)))
            table.scale(1, 1.3)

        ax.set_title(
            f"Recommended Feature Subset ({len(kept)} features: "
            f"{len(kept_static)} static + {len(kept_temporal)} temporal)\n"
            "Static: RF importance > median AND not redundant  |  "
            "Temporal: kept unconditionally",
            fontsize=12, pad=20,
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"Saved {OUTPUT_PDF}", flush=True)


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("UAVDT Feature Selection Analysis (v2 — RF + LASSO)", flush=True)
    print("=" * 60, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1
    print("\nStep 1: Build analysis dataset...", flush=True)
    df = build_analysis_dataset()
    print(f"  Analysis dataset: {len(df)} rows, weather distribution:", flush=True)
    print(df["weather"].value_counts().to_string(), flush=True)

    available_features = [
        f for f in FEATURE_NAMES
        if f in df.columns and df[f].notna().mean() > 0.5
    ]
    available_static = [f for f in available_features if f in STATIC_FEATURES]
    available_temporal = [f for f in available_features if f in TEMPORAL_FEATURES]
    print(f"  Available: {len(available_features)}/56 "
          f"({len(available_static)} static + {len(available_temporal)} temporal)",
          flush=True)

    # Step 2
    print("\nStep 2: Spearman correlations...", flush=True)
    spearman_gap = compute_spearman(df, available_features, "recall_gap")
    spearman_nano = compute_spearman(df, available_features, "recall_nano")
    spearman_medium = compute_spearman(df, available_features, "recall_medium")

    rho_gap_dict = {f: r for f, r, _ in spearman_gap}
    pval_gap_dict = {f: p for f, _, p in spearman_gap}
    rho_nano_dict = {f: r for f, r, _ in spearman_nano}
    rho_medium_dict = {f: r for f, r, _ in spearman_medium}

    weather_rhos = spearman_by_weather(df, available_features)
    print("  Done.", flush=True)

    # Step 3
    print("\nStep 3: Random Forest permutation importance (static only)...", flush=True)
    rf_importance_dict = compute_rf_importance(df, available_static)

    # Step 4
    print("\nStep 4: LASSO validation (static only)...", flush=True)
    lasso_coef_dict = compute_lasso(df, available_static)

    # Log disagreements
    rf_median = np.median(list(rf_importance_dict.values()))
    lasso_nonzero = {f for f, c in lasso_coef_dict.items() if c != 0}
    rf_above_median = {f for f, v in rf_importance_dict.items() if v > rf_median}
    rf_only = rf_above_median - lasso_nonzero
    lasso_only = lasso_nonzero - rf_above_median
    if rf_only:
        print(f"  Disagreement — RF selects, LASSO drops: {sorted(rf_only)}", flush=True)
    if lasso_only:
        print(f"  Disagreement — LASSO selects, RF drops: {sorted(lasso_only)}", flush=True)

    # Step 5
    print("\nStep 5: Redundancy analysis...", flush=True)
    corr_matrix, redundant_dict = compute_redundancy(
        df, available_features, rf_importance_dict, rho_gap_dict
    )
    print(f"  Redundant features: {len(redundant_dict)}", flush=True)

    # Step 6: Build summary
    print("\nStep 6: Building summary with final selection...", flush=True)
    rows = []
    for feat in FEATURE_NAMES:
        rho = rho_gap_dict.get(feat, np.nan)
        pval = pval_gap_dict.get(feat, np.nan)
        rho_n = rho_nano_dict.get(feat, np.nan)
        rho_m = rho_medium_dict.get(feat, np.nan)
        rf_imp = rf_importance_dict.get(feat, np.nan)
        lasso_c = lasso_coef_dict.get(feat, np.nan)
        redundant_with = redundant_dict.get(feat, "")
        category = FEATURE_CATEGORIES.get(feat, "unknown")

        is_temporal = feat in TEMPORAL_FEATURES
        is_available = feat in available_features

        if not is_available:
            keep = False
            keep_reason = "unavailable"
        elif is_temporal:
            keep = True
            keep_reason = "temporal (kept unconditionally)"
        else:
            above_rf = (rf_imp if not np.isnan(rf_imp) else 0) > rf_median
            not_redundant = feat not in redundant_dict

            if above_rf and not_redundant:
                lasso_agrees = lasso_c != 0 if not np.isnan(lasso_c) else False
                if lasso_agrees:
                    keep = True
                    keep_reason = "RF > median, not redundant, LASSO agrees"
                else:
                    keep = True
                    keep_reason = "RF > median, not redundant (LASSO disagrees)"
            elif above_rf and not not_redundant:
                keep = False
                keep_reason = f"RF > median but redundant with {redundant_with}"
            elif not above_rf and not_redundant:
                keep = False
                keep_reason = "RF below median"
            else:
                keep = False
                keep_reason = f"RF below median and redundant with {redundant_with}"

        rows.append({
            "feature": feat,
            "category": category,
            "spearman_rho": rho,
            "spearman_pval": pval,
            "rho_nano": rho_n,
            "rho_medium": rho_m,
            "rf_importance": rf_imp,
            "lasso_coef": lasso_c,
            "redundant_with": redundant_with,
            "keep": keep,
            "keep_reason": keep_reason,
        })

    summary_df = pd.DataFrame(rows)
    summary_df["abs_rho"] = summary_df["spearman_rho"].abs()
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"  Saved {SUMMARY_CSV}", flush=True)

    # Print top 15
    print("\nTop 15 features by |Spearman rho| with recall_gap:", flush=True)
    print("-" * 85, flush=True)
    top15_df = summary_df.nlargest(15, "abs_rho")
    for _, row in top15_df.iterrows():
        marker = " *KEEP*" if row["keep"] else ""
        rf_str = (f"RF={row['rf_importance']:.4f}"
                  if not pd.isna(row['rf_importance']) else "RF=N/A")
        lasso_str = (f"L={row['lasso_coef']:.4f}"
                     if not pd.isna(row['lasso_coef']) else "L=N/A")
        print(f"  {row['feature']:35s}  rho={row['spearman_rho']:+.4f}  "
              f"{rf_str}  {lasso_str}  [{row['category']}]{marker}", flush=True)

    kept_count = summary_df["keep"].sum()
    static_kept = summary_df[
        (summary_df["keep"]) & (~summary_df["feature"].isin(TEMPORAL_FEATURES))
    ].shape[0]
    temporal_kept = summary_df[
        (summary_df["keep"]) & (summary_df["feature"].isin(TEMPORAL_FEATURES))
    ].shape[0]
    print(f"\nRecommended subset: {kept_count} features "
          f"({static_kept} static + {temporal_kept} temporal)", flush=True)
    print(f"RF importance median threshold: {rf_median:.6f}", flush=True)

    # Generate PDF
    print("\nGenerating 8-page PDF...", flush=True)
    create_pdf(summary_df, corr_matrix, available_features, weather_rhos, df,
               available_static)

    print("\nDone!", flush=True)
