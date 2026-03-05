#!/usr/bin/env python3
"""Comprehensive within-sequence correlation analysis for new features.

Methods:
  1. Within-sequence Spearman (per-frame + windowed)
  2. Within-sequence Elastic Net (per-sequence normalized)
  3. Within-sequence Ridge Regression
  4. Within-sequence Lasso
  5. Feature-feature correlation matrix
  6. Method comparison and ranking

All analysis is WITHIN-SEQUENCE to avoid Simpson's paradox.
Reports both signed mean and mean(|rho|) to avoid cancellation bug.

Output:
  new_features/new_analysis_results.pdf
  new_features/new_analysis_summary.csv
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "new_all_features.csv"
OUT_PDF = SCRIPT_DIR / "new_analysis_results.pdf"
OUT_SUMMARY = SCRIPT_DIR / "new_analysis_summary.csv"

META_COLS = ["sequence", "frame_id", "weather", "fn_nano", "num_gt",
             "tp_nano", "tp_medium", "recoverable_fn"]

# Primary target
PRIMARY_TARGET = "fn_nano"
SECONDARY_TARGETS = ["recoverable_fn", "num_gt"]

# Minimum frames per sequence for valid statistics
MIN_FRAMES = 30


# ═══════════════════════════════════════════════════════════════════
# 1. Within-Sequence Spearman Correlation
# ═══════════════════════════════════════════════════════════════════

def within_seq_spearman(df, feature_cols, target, min_frames=MIN_FRAMES):
    """Compute within-sequence Spearman correlation.

    Returns DataFrame with columns:
      feature, signed_mean_rho, mean_abs_rho, median_rho, std_rho, n_valid_seqs,
      pct_positive, pct_significant
    """
    results = []
    grouped = df.groupby("sequence")

    for feat in feature_cols:
        rhos = []
        pvals = []

        for seq_name, seq_df in grouped:
            if len(seq_df) < min_frames:
                continue
            x = seq_df[feat].values
            y = seq_df[target].values

            # Skip if either is constant
            if x.std() < 1e-10 or y.std() < 1e-10:
                continue

            rho, p = spearmanr(x, y)
            if np.isnan(rho):
                continue
            rhos.append(rho)
            pvals.append(p)

        if len(rhos) == 0:
            results.append({
                "feature": feat, "signed_mean_rho": 0.0, "mean_abs_rho": 0.0,
                "median_rho": 0.0, "std_rho": 0.0, "n_valid_seqs": 0,
                "pct_positive": 0.0, "pct_significant": 0.0,
            })
            continue

        rhos = np.array(rhos)
        pvals = np.array(pvals)
        results.append({
            "feature": feat,
            "signed_mean_rho": float(rhos.mean()),
            "mean_abs_rho": float(np.abs(rhos).mean()),
            "median_rho": float(np.median(rhos)),
            "std_rho": float(rhos.std()),
            "n_valid_seqs": len(rhos),
            "pct_positive": float((rhos > 0).mean() * 100),
            "pct_significant": float((pvals < 0.05).mean() * 100),
        })

    return pd.DataFrame(results).sort_values("mean_abs_rho", ascending=False)


def windowed_spearman(df, feature_cols, target, window_size, min_frames=MIN_FRAMES):
    """Within-sequence Spearman on non-overlapping window averages."""
    records = []
    grouped = df.groupby("sequence")

    for seq_name, seq_df in grouped:
        seq_df = seq_df.sort_values("frame_id")
        n = len(seq_df)
        if n < min_frames:
            continue

        n_windows = n // window_size
        if n_windows < 5:  # need enough windows for meaningful rho
            continue

        for col in feature_cols + [target]:
            vals = seq_df[col].values[:n_windows * window_size]
            windowed = vals.reshape(n_windows, window_size).mean(axis=1)
            for w_idx in range(n_windows):
                if col == feature_cols[0]:
                    # Initialize record
                    pass
                # Store windowed value
                if w_idx >= len(records) or records[-n_windows + w_idx].get("sequence") != seq_name:
                    pass

    # Simpler approach: build windowed DataFrame
    windowed_records = []
    for seq_name, seq_df in grouped:
        seq_df = seq_df.sort_values("frame_id").reset_index(drop=True)
        n = len(seq_df)
        n_windows = n // window_size
        if n_windows < 5:
            continue

        for w_idx in range(n_windows):
            start = w_idx * window_size
            end = start + window_size
            chunk = seq_df.iloc[start:end]
            rec = {"sequence": seq_name, "window_idx": w_idx}
            for col in feature_cols + [target]:
                rec[col] = chunk[col].mean()
            windowed_records.append(rec)

    if not windowed_records:
        return pd.DataFrame()

    w_df = pd.DataFrame(windowed_records)
    return within_seq_spearman(w_df, feature_cols, target, min_frames=5)


# ═══════════════════════════════════════════════════════════════════
# 2. Within-Sequence Linear Models (Elastic Net / Ridge / Lasso)
# ═══════════════════════════════════════════════════════════════════

def within_seq_linear_models(df, feature_cols, target, min_frames=MIN_FRAMES):
    """Fit linear models with per-sequence z-score normalization.

    Uses GroupKFold (groups = sequences) to prevent data leakage.
    Returns dict with model results and coefficient rankings.
    """
    # Per-sequence z-score normalization
    df_norm = df.copy()
    grouped = df_norm.groupby("sequence")

    # Filter sequences with enough frames
    valid_seqs = [name for name, g in grouped if len(g) >= min_frames]
    df_norm = df_norm[df_norm["sequence"].isin(valid_seqs)]

    if len(df_norm) < 100:
        return None

    # Z-score normalize within each sequence
    for col in feature_cols + [target]:
        df_norm[col] = df_norm.groupby("sequence")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        )

    X = df_norm[feature_cols].values
    y = df_norm[target].values
    groups = df_norm["sequence"].values

    # Remove any rows with NaN/inf
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid_mask]
    y = y[valid_mask]
    groups = groups[valid_mask]

    if len(X) < 100:
        return None

    # Scale features (on top of per-sequence z-score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # GroupKFold: 5-fold with sequences as groups
    n_unique_groups = len(np.unique(groups))
    n_splits = min(5, n_unique_groups)
    if n_splits < 2:
        return None

    gkf = GroupKFold(n_splits=n_splits)

    results = {}

    # ── Elastic Net ──
    try:
        enet = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
            n_alphas=50,
            cv=gkf.split(X_scaled, y, groups),
            max_iter=5000,
            random_state=42,
        )
        enet.fit(X_scaled, y)
        # Cross-validated R²
        cv_scores = []
        for train_idx, test_idx in gkf.split(X_scaled, y, groups):
            enet_fold = ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
                n_alphas=50, cv=3, max_iter=5000, random_state=42,
            )
            enet_fold.fit(X_scaled[train_idx], y[train_idx])
            cv_scores.append(enet_fold.score(X_scaled[test_idx], y[test_idx]))

        coefs = pd.Series(enet.coef_, index=feature_cols)
        results["elastic_net"] = {
            "r2_cv": float(np.mean(cv_scores)),
            "r2_cv_std": float(np.std(cv_scores)),
            "l1_ratio": float(enet.l1_ratio_),
            "alpha": float(enet.alpha_),
            "n_nonzero": int((np.abs(enet.coef_) > 1e-6).sum()),
            "top_features": coefs.abs().sort_values(ascending=False).head(20),
            "all_coefs": coefs,
        }
    except Exception as e:
        print(f"  Elastic Net failed: {e}")
        results["elastic_net"] = None

    # ── Ridge ──
    try:
        alphas = np.logspace(-3, 3, 50)
        cv_scores = []
        best_alpha = 1.0
        best_score = -np.inf

        for train_idx, test_idx in gkf.split(X_scaled, y, groups):
            ridge = RidgeCV(alphas=alphas, cv=3)
            ridge.fit(X_scaled[train_idx], y[train_idx])
            score = ridge.score(X_scaled[test_idx], y[test_idx])
            cv_scores.append(score)

        # Fit final model on all data for coefficients
        ridge_final = RidgeCV(alphas=alphas, cv=3)
        ridge_final.fit(X_scaled, y)
        coefs = pd.Series(ridge_final.coef_, index=feature_cols)
        results["ridge"] = {
            "r2_cv": float(np.mean(cv_scores)),
            "r2_cv_std": float(np.std(cv_scores)),
            "alpha": float(ridge_final.alpha_),
            "top_features": coefs.abs().sort_values(ascending=False).head(20),
            "all_coefs": coefs,
        }
    except Exception as e:
        print(f"  Ridge failed: {e}")
        results["ridge"] = None

    # ── Lasso ──
    try:
        cv_scores = []
        for train_idx, test_idx in gkf.split(X_scaled, y, groups):
            lasso_fold = LassoCV(
                n_alphas=50, cv=3, max_iter=5000, random_state=42,
            )
            lasso_fold.fit(X_scaled[train_idx], y[train_idx])
            cv_scores.append(lasso_fold.score(X_scaled[test_idx], y[test_idx]))

        lasso_final = LassoCV(n_alphas=50, cv=3, max_iter=5000, random_state=42)
        lasso_final.fit(X_scaled, y)
        coefs = pd.Series(lasso_final.coef_, index=feature_cols)
        results["lasso"] = {
            "r2_cv": float(np.mean(cv_scores)),
            "r2_cv_std": float(np.std(cv_scores)),
            "alpha": float(lasso_final.alpha_),
            "n_nonzero": int((np.abs(lasso_final.coef_) > 1e-6).sum()),
            "top_features": coefs.abs().sort_values(ascending=False).head(20),
            "all_coefs": coefs,
        }
    except Exception as e:
        print(f"  Lasso failed: {e}")
        results["lasso"] = None

    return results


# ═══════════════════════════════════════════════════════════════════
# 3. Feature-Feature Correlation Matrix
# ═══════════════════════════════════════════════════════════════════

def feature_correlation_matrix(df, feature_cols, target):
    """Compute within-sequence Spearman correlation between all feature pairs.

    Returns matrix of mean(|rho|) across sequences.
    """
    n_feats = len(feature_cols)
    all_cols = feature_cols + [target]
    n_all = len(all_cols)
    corr_matrix = np.zeros((n_all, n_all))

    grouped = df.groupby("sequence")
    n_seqs = 0

    for seq_name, seq_df in grouped:
        if len(seq_df) < MIN_FRAMES:
            continue
        n_seqs += 1

        # Compute Spearman for this sequence
        data = seq_df[all_cols].values
        # Skip columns with zero variance
        stds = data.std(axis=0)
        valid = stds > 1e-10

        if valid.sum() < 2:
            continue

        seq_corr = np.zeros((n_all, n_all))
        for i in range(n_all):
            if not valid[i]:
                continue
            for j in range(i, n_all):
                if not valid[j]:
                    continue
                rho, _ = spearmanr(data[:, i], data[:, j])
                if np.isnan(rho):
                    continue
                seq_corr[i, j] = abs(rho)
                seq_corr[j, i] = abs(rho)

        corr_matrix += seq_corr

    if n_seqs > 0:
        corr_matrix /= n_seqs

    return pd.DataFrame(corr_matrix, index=all_cols, columns=all_cols)


# ═══════════════════════════════════════════════════════════════════
# PDF Report Generation
# ═══════════════════════════════════════════════════════════════════

def generate_pdf(df, feature_cols, spearman_results, windowed_results,
                 linear_results, corr_matrix):
    """Generate comprehensive PDF report."""

    with PdfPages(OUT_PDF) as pdf:
        # ── Page 1: Spearman rankings (per-frame) ──
        fig, axes = plt.subplots(1, 2, figsize=(18, 12))

        # Left: mean(|rho|) bar chart
        ax = axes[0]
        sp = spearman_results[PRIMARY_TARGET].head(30)
        colors = []
        for feat in sp["feature"]:
            if any(feat.startswith(p) for p in ["fft_", "dct_", "downsample_",
                   "keypoint_", "gabor_", "laplacian_", "edge_fine_",
                   "spatial_frequency", "fast_keypoints"]):
                colors.append("#2196F3")  # Blue = resolution
            elif any(feat.startswith(p) for p in ["foreground_", "shadow_",
                     "motion_", "temporal_"]):
                colors.append("#4CAF50")  # Green = density
            elif any(feat.startswith(p) for p in ["mscn_", "brisque"]):
                colors.append("#FF9800")  # Orange = BRISQUE/quality
            else:
                colors.append("#9E9E9E")  # Gray = other

        y_pos = range(len(sp))
        ax.barh(y_pos, sp["mean_abs_rho"], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sp["feature"], fontsize=7)
        ax.set_xlabel("Mean |ρ| (within-sequence)")
        ax.set_title(f"Top 30 Features vs {PRIMARY_TARGET}\n(Within-Seq Spearman)")
        ax.axvline(x=0.3, color="red", linestyle="--", alpha=0.5, label="ρ=0.3")
        ax.axvline(x=0.2, color="orange", linestyle="--", alpha=0.5, label="ρ=0.2")
        ax.legend(fontsize=7)
        ax.invert_yaxis()

        # Right: signed mean rho (to show direction)
        ax = axes[1]
        ax.barh(y_pos, sp["signed_mean_rho"], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sp["feature"], fontsize=7)
        ax.set_xlabel("Signed Mean ρ (within-sequence)")
        ax.set_title(f"Direction of Correlation\n(positive = more feature → more fn)")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.invert_yaxis()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: Consistency analysis ──
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))

        # Left: % sequences where correlation is significant
        ax = axes[0]
        sp_top = spearman_results[PRIMARY_TARGET].head(20)
        ax.barh(range(len(sp_top)), sp_top["pct_significant"], color="#5C6BC0")
        ax.set_yticks(range(len(sp_top)))
        ax.set_yticklabels(sp_top["feature"], fontsize=7)
        ax.set_xlabel("% Sequences with p < 0.05")
        ax.set_title("Consistency: % Sequences with Significant Correlation")
        ax.axvline(x=50, color="red", linestyle="--", alpha=0.5)
        ax.invert_yaxis()

        # Right: std of rho across sequences (lower = more consistent)
        ax = axes[1]
        ax.barh(range(len(sp_top)), sp_top["std_rho"], color="#AB47BC")
        ax.set_yticks(range(len(sp_top)))
        ax.set_yticklabels(sp_top["feature"], fontsize=7)
        ax.set_xlabel("Std(ρ) across sequences")
        ax.set_title("Stability: Lower = More Consistent Across Sequences")
        ax.invert_yaxis()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 3: Windowed Spearman comparison ──
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))

        # Get top 10 features from per-frame analysis
        top10 = spearman_results[PRIMARY_TARGET].head(10)["feature"].tolist()

        # Left: rho vs window size
        ax = axes[0]
        window_sizes = sorted(windowed_results.keys())
        for feat in top10:
            rhos = []
            for ws in window_sizes:
                w_df = windowed_results[ws]
                if w_df is not None and len(w_df) > 0:
                    row = w_df[w_df["feature"] == feat]
                    if len(row) > 0:
                        rhos.append(row["mean_abs_rho"].values[0])
                    else:
                        rhos.append(0.0)
                else:
                    rhos.append(0.0)
            ax.plot(window_sizes, rhos, "o-", label=feat[:25], markersize=4)

        ax.set_xlabel("Window Size (frames)")
        ax.set_ylabel("Mean |ρ| (within-sequence)")
        ax.set_title(f"Effect of Temporal Averaging on Correlation\n(Top 10 features)")
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)

        # Right: best windowed (w=30) bar chart
        ax = axes[1]
        if 30 in windowed_results and windowed_results[30] is not None:
            w30 = windowed_results[30].head(20)
            ax.barh(range(len(w30)), w30["mean_abs_rho"])
            ax.set_yticks(range(len(w30)))
            ax.set_yticklabels(w30["feature"], fontsize=7)
            ax.set_xlabel("Mean |ρ| (within-seq, w=30)")
            ax.set_title("Top 20 at Window=30 Frames")
            ax.axvline(x=0.3, color="red", linestyle="--", alpha=0.5)
            ax.invert_yaxis()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 4: Linear model coefficients ──
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))

        for idx, (model_name, display_name) in enumerate([
            ("elastic_net", "Elastic Net"),
            ("ridge", "Ridge"),
            ("lasso", "Lasso"),
        ]):
            ax = axes[idx]
            if linear_results and model_name in linear_results and linear_results[model_name]:
                res = linear_results[model_name]
                top = res["top_features"]
                if len(top) > 0:
                    ax.barh(range(len(top)), top.values)
                    ax.set_yticks(range(len(top)))
                    ax.set_yticklabels(top.index, fontsize=6)

                    r2 = res["r2_cv"]
                    r2_std = res["r2_cv_std"]
                    title = f"{display_name}\nCV R² = {r2:.4f} ± {r2_std:.4f}"
                    if "n_nonzero" in res:
                        title += f"\n{res['n_nonzero']} non-zero features"
                    ax.set_title(title, fontsize=9)
                    ax.set_xlabel("|Coefficient|")
                    ax.invert_yaxis()
            else:
                ax.text(0.5, 0.5, f"{display_name}\nFailed", ha="center",
                        va="center", transform=ax.transAxes)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 5: Feature-feature correlation heatmap ──
        fig, ax = plt.subplots(figsize=(20, 18))

        # Show top 25 features + target
        top25 = spearman_results[PRIMARY_TARGET].head(25)["feature"].tolist()
        show_cols = top25 + [PRIMARY_TARGET]
        sub_matrix = corr_matrix.loc[show_cols, show_cols]

        im = ax.imshow(sub_matrix.values, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(show_cols)))
        ax.set_xticklabels(show_cols, rotation=90, fontsize=6)
        ax.set_yticks(range(len(show_cols)))
        ax.set_yticklabels(show_cols, fontsize=6)
        ax.set_title("Within-Sequence Feature Correlation (Mean |ρ|)\nTop 25 Features + Target")
        plt.colorbar(im, ax=ax, shrink=0.6)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 6: Multi-target comparison ──
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))

        for idx, target in enumerate([PRIMARY_TARGET] + SECONDARY_TARGETS):
            ax = axes[idx]
            if target in spearman_results:
                sp_t = spearman_results[target].head(20)
                ax.barh(range(len(sp_t)), sp_t["mean_abs_rho"])
                ax.set_yticks(range(len(sp_t)))
                ax.set_yticklabels(sp_t["feature"], fontsize=7)
                ax.set_xlabel("Mean |ρ|")
                ax.set_title(f"Top 20 vs {target}")
                ax.axvline(x=0.3, color="red", linestyle="--", alpha=0.5)
                ax.invert_yaxis()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 7: Summary table ──
        fig, ax = plt.subplots(figsize=(18, 14))
        ax.axis("off")

        # Build summary table
        summary_lines = []
        summary_lines.append("=" * 90)
        summary_lines.append("COMPREHENSIVE FEATURE ANALYSIS SUMMARY")
        summary_lines.append("=" * 90)
        summary_lines.append(f"\nDataset: DETRAC — {df['sequence'].nunique()} sequences, "
                             f"{len(df):,} frames")
        summary_lines.append(f"Features: {len(feature_cols)}")
        summary_lines.append(f"Primary target: {PRIMARY_TARGET}")
        summary_lines.append(f"Analysis: class-agnostic, within-sequence")

        # Spearman top 5
        sp_main = spearman_results[PRIMARY_TARGET]
        summary_lines.append(f"\n{'─' * 60}")
        summary_lines.append("SPEARMAN TOP 5 (per-frame, mean |ρ|):")
        for _, row in sp_main.head(5).iterrows():
            summary_lines.append(
                f"  {row['feature']:40s}  |ρ|={row['mean_abs_rho']:.4f}  "
                f"signed={row['signed_mean_rho']:+.4f}  "
                f"sig={row['pct_significant']:.0f}%"
            )

        # Windowed top 5
        if 30 in windowed_results and windowed_results[30] is not None:
            w30 = windowed_results[30]
            summary_lines.append(f"\nSPEARMAN TOP 5 (w=30 avg, mean |ρ|):")
            for _, row in w30.head(5).iterrows():
                summary_lines.append(
                    f"  {row['feature']:40s}  |ρ|={row['mean_abs_rho']:.4f}"
                )

        # Linear model R²
        summary_lines.append(f"\n{'─' * 60}")
        summary_lines.append("LINEAR MODEL COMPARISON (within-seq normalized, GroupKFold CV):")
        if linear_results:
            for model_name in ["elastic_net", "ridge", "lasso"]:
                if linear_results.get(model_name):
                    res = linear_results[model_name]
                    line = f"  {model_name:15s}  R²={res['r2_cv']:.4f} ± {res['r2_cv_std']:.4f}"
                    if "n_nonzero" in res:
                        line += f"  ({res['n_nonzero']} features)"
                    summary_lines.append(line)

        # Feature category breakdown
        summary_lines.append(f"\n{'─' * 60}")
        summary_lines.append("FEATURE CATEGORY PERFORMANCE:")
        categories = {
            "A: Resolution": ["fft_", "dct_", "downsample_", "keypoint_",
                              "gabor_", "laplacian_", "edge_fine_", "edge_density_",
                              "spatial_frequency", "fast_keypoints", "gradient_magnitude"],
            "B: Density": ["foreground_", "shadow_", "motion_", "temporal_"],
            "C: Complexity": ["glcm_", "colorfulness", "image_entropy",
                              "mean_brightness", "brightness_std", "dark_channel",
                              "rms_contrast", "saturation_", "small_edge_",
                              "total_edge_", "overall_edge_"],
            "D: Strip": ["top_", "mid_", "bot_", "ratio_"],
            "E: BRISQUE": ["mscn_"],
        }

        for cat_name, prefixes in categories.items():
            cat_feats = [f for f in feature_cols
                         if any(f.startswith(p) or f == p for p in prefixes)]
            if not cat_feats:
                continue
            cat_rhos = sp_main[sp_main["feature"].isin(cat_feats)]["mean_abs_rho"]
            if len(cat_rhos) > 0:
                summary_lines.append(
                    f"  {cat_name:20s}  n={len(cat_feats):2d}  "
                    f"best |ρ|={cat_rhos.max():.4f}  "
                    f"mean |ρ|={cat_rhos.mean():.4f}"
                )

        # Redundancy warning
        summary_lines.append(f"\n{'─' * 60}")
        summary_lines.append("REDUNDANCY CHECK (feature pairs with |ρ| > 0.9):")
        n_redundant = 0
        checked = set()
        for i, f1 in enumerate(feature_cols):
            for j, f2 in enumerate(feature_cols):
                if i >= j:
                    continue
                key = (f1, f2)
                if key in checked:
                    continue
                checked.add(key)
                if f1 in corr_matrix.index and f2 in corr_matrix.columns:
                    r = corr_matrix.loc[f1, f2]
                    if r > 0.9:
                        summary_lines.append(f"  {f1:35s} ↔ {f2:35s}  |ρ|={r:.3f}")
                        n_redundant += 1
        if n_redundant == 0:
            summary_lines.append("  No highly redundant pairs found.")

        text = "\n".join(summary_lines)
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                fontfamily="monospace", fontsize=7, verticalalignment="top")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\n  PDF saved to {OUT_PDF}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("COMPREHENSIVE WITHIN-SEQUENCE CORRELATION ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading features ...", flush=True)
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  {len(df):,} frames, {df['sequence'].nunique()} sequences, "
          f"{len(feature_cols)} features")

    # Handle NaN: fill temporal features for first frames with 0
    temporal_feats = [c for c in feature_cols
                      if c.startswith("temporal_") or c == "motion_pixel_ratio"]
    for tf in temporal_feats:
        df[tf] = df[tf].fillna(0.0)

    # Drop any remaining NaN rows (should be none)
    n_before = len(df)
    df = df.dropna(subset=feature_cols)
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} rows with NaN")

    # ── Step 1: Per-frame Spearman ──
    print("\n[1/6] Within-sequence Spearman (per-frame) ...", flush=True)
    spearman_results = {}
    for target in [PRIMARY_TARGET] + SECONDARY_TARGETS:
        print(f"  Target: {target}", flush=True)
        spearman_results[target] = within_seq_spearman(df, feature_cols, target)

    # Print top 15 for primary target
    sp_main = spearman_results[PRIMARY_TARGET]
    print(f"\n  {'Feature':<40s} {'|ρ|':>8s} {'signed':>8s} {'%sig':>6s} {'%pos':>6s}")
    print(f"  {'-'*68}")
    for _, row in sp_main.head(15).iterrows():
        print(f"  {row['feature']:<40s} {row['mean_abs_rho']:8.4f} "
              f"{row['signed_mean_rho']:+8.4f} {row['pct_significant']:5.0f}% "
              f"{row['pct_positive']:5.0f}%")

    # ── Step 2: Windowed Spearman ──
    print("\n[2/6] Windowed Spearman ...", flush=True)
    windowed_results = {}
    for ws in [1, 10, 30, 60, 90]:
        print(f"  Window = {ws} frames ...", flush=True)
        if ws == 1:
            windowed_results[ws] = spearman_results[PRIMARY_TARGET]
        else:
            result = windowed_spearman(df, feature_cols, PRIMARY_TARGET, ws)
            windowed_results[ws] = result

    # Print windowed improvement
    if 30 in windowed_results and windowed_results[30] is not None:
        w30 = windowed_results[30]
        print(f"\n  Top 5 at window=30:")
        for _, row in w30.head(5).iterrows():
            print(f"    {row['feature']:<40s} |ρ|={row['mean_abs_rho']:.4f}")

    # ── Step 3: Linear models ──
    print("\n[3/6] Linear models (per-seq normalized, GroupKFold CV) ...", flush=True)
    linear_results = within_seq_linear_models(df, feature_cols, PRIMARY_TARGET)

    if linear_results:
        for model_name in ["elastic_net", "ridge", "lasso"]:
            res = linear_results.get(model_name)
            if res:
                line = f"  {model_name:15s}  R²={res['r2_cv']:.4f} ± {res['r2_cv_std']:.4f}"
                if "n_nonzero" in res:
                    line += f"  ({res['n_nonzero']} non-zero)"
                print(line)

    # ── Step 4: Feature-feature correlation ──
    print("\n[4/6] Feature-feature correlation matrix ...", flush=True)
    # Use top 30 features + target for matrix (full 65×65 is too slow)
    top30_feats = sp_main.head(30)["feature"].tolist()
    corr_matrix = feature_correlation_matrix(df, top30_feats, PRIMARY_TARGET)
    print(f"  Computed {len(corr_matrix)}×{len(corr_matrix)} matrix")

    # ── Step 5: Generate PDF ──
    print("\n[5/6] Generating PDF report ...", flush=True)
    generate_pdf(df, feature_cols, spearman_results, windowed_results,
                 linear_results, corr_matrix)

    # ── Step 6: Save summary CSV ──
    print("\n[6/6] Saving summary CSV ...", flush=True)
    summary = sp_main.copy()

    # Add linear model coefficients
    for model_name in ["elastic_net", "ridge", "lasso"]:
        if linear_results and linear_results.get(model_name):
            coefs = linear_results[model_name]["all_coefs"]
            summary[f"{model_name}_coef"] = summary["feature"].map(
                lambda f: coefs.get(f, 0.0)
            )
            summary[f"{model_name}_abs_coef"] = summary[f"{model_name}_coef"].abs()

    # Add windowed rho (w=30)
    if 30 in windowed_results and windowed_results[30] is not None:
        w30_map = windowed_results[30].set_index("feature")["mean_abs_rho"].to_dict()
        summary["windowed_30_abs_rho"] = summary["feature"].map(
            lambda f: w30_map.get(f, 0.0)
        )

    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"  Summary saved to {OUT_SUMMARY}")

    # ── Final verdict ──
    print(f"\n{'=' * 70}")
    print("FINAL RANKING — BEST 3 METHODS")
    print(f"{'=' * 70}")

    methods = []
    methods.append(("Spearman (per-frame)", sp_main["mean_abs_rho"].max(),
                     sp_main.iloc[0]["feature"]))
    if 30 in windowed_results and windowed_results[30] is not None:
        w30 = windowed_results[30]
        methods.append(("Spearman (w=30)", w30["mean_abs_rho"].max(),
                         w30.iloc[0]["feature"]))
    if 90 in windowed_results and windowed_results[90] is not None:
        w90 = windowed_results[90]
        methods.append(("Spearman (w=90)", w90["mean_abs_rho"].max(),
                         w90.iloc[0]["feature"]))
    if linear_results:
        for model_name in ["elastic_net", "ridge", "lasso"]:
            res = linear_results.get(model_name)
            if res:
                methods.append((model_name.replace("_", " ").title(),
                                res["r2_cv"], f"R² ({res.get('n_nonzero', '?')} feats)"))

    methods.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, score, detail) in enumerate(methods, 1):
        print(f"  {rank}. {name:25s}  score={score:.4f}  ({detail})")

    print(f"\n  PDF:     {OUT_PDF}")
    print(f"  Summary: {OUT_SUMMARY}")
    print()


if __name__ == "__main__":
    main()
