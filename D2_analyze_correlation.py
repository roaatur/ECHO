#!/usr/bin/env python3
"""D²-City Step 4 — Comprehensive within-clip correlation analysis.

Ports new_features/new_analyze_correlation.py to D²-City.

Methods:
  1. Within-clip Spearman (per-frame + windowed at w = 1, 10, 30, 60, 90)
  2. Linear models: ElasticNetCV, RidgeCV, LassoCV (GroupKFold, per-clip normalised)
  3. Feature-feature correlation matrix (top 25 + target)
  4. Multi-target comparison: fn_nano, recoverable_fn, num_gt_coco
  5. Comparison banner vs DETRAC and METEOR reference values

All analysis is WITHIN-CLIP to avoid Simpson's paradox.
Both signed mean and mean(|ρ|) are reported to avoid sign-cancellation bugs.

Outputs:
  D2_analysis_results.pdf   — 7-page report
  D2_analysis_summary.csv   — ranked feature table (input for D2_select_features.py)
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

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D2_common import SCRIPT_DIR

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_CSV = SCRIPT_DIR / "D2_all_features.csv"
OUT_PDF      = SCRIPT_DIR / "D2_analysis_results.pdf"
OUT_SUMMARY  = SCRIPT_DIR / "D2_analysis_summary.csv"

META_COLS       = ["clip_id", "frame_idx", "fn_nano", "num_gt_coco",
                   "tp_nano", "tp_medium", "recoverable_fn"]
PRIMARY_TARGET   = "fn_nano"
SECONDARY_TARGETS = ["recoverable_fn", "num_gt_coco"]
MIN_FRAMES       = 30

# Reference values from prior experiments (for comparison banner)
DETRAC_BEST_ABS_RHO  = 0.413   # fast_keypoints_half, Exp 7
DETRAC_BEST_W30_RHO  = 0.495
METEOR_BEST_ABS_RHO  = 0.082   # colorfulness, Exp 6


# ── 1. Within-clip Spearman ───────────────────────────────────────────────────

def within_clip_spearman(df, feature_cols, target, min_frames=MIN_FRAMES):
    """Per-frame within-clip Spearman ρ for all features vs target.

    Returns DataFrame: feature, signed_mean_rho, mean_abs_rho, median_rho,
                       std_rho, n_valid_clips, pct_positive, pct_significant
    """
    results = []
    grouped = df.groupby("clip_id")

    for feat in feature_cols:
        rhos, pvals = [], []
        for clip_id, cdf in grouped:
            if len(cdf) < min_frames:
                continue
            x, y = cdf[feat].values, cdf[target].values
            if x.std() < 1e-10 or y.std() < 1e-10:
                continue
            rho, p = spearmanr(x, y)
            if not np.isnan(rho):
                rhos.append(rho); pvals.append(p)

        if not rhos:
            results.append({"feature": feat, "signed_mean_rho": 0.0,
                             "mean_abs_rho": 0.0, "median_rho": 0.0,
                             "std_rho": 0.0, "n_valid_clips": 0,
                             "pct_positive": 0.0, "pct_significant": 0.0})
            continue

        ra = np.array(rhos); pa = np.array(pvals)
        results.append({
            "feature":        feat,
            "signed_mean_rho":float(ra.mean()),
            "mean_abs_rho":   float(np.abs(ra).mean()),
            "median_rho":     float(np.median(ra)),
            "std_rho":        float(ra.std()),
            "n_valid_clips":  len(ra),
            "pct_positive":   float((ra > 0).mean() * 100),
            "pct_significant":float((pa < 0.05).mean() * 100),
        })

    return (pd.DataFrame(results)
            .sort_values("mean_abs_rho", ascending=False)
            .reset_index(drop=True))


# ── 2. Windowed within-clip Spearman ─────────────────────────────────────────

def windowed_spearman(df, feature_cols, target, windows=(1, 10, 30, 60, 90)):
    """Non-overlapping window Spearman for each window size."""
    records = {feat: {} for feat in feature_cols}

    for w in windows:
        for feat in feature_cols:
            rhos = []
            for clip_id, cdf in df.groupby("clip_id"):
                cdf = cdf.sort_values("frame_idx").reset_index(drop=True)
                n   = len(cdf)
                if n < 2 * w:
                    continue
                feat_means = [cdf[feat].iloc[t:t+w].mean()
                              for t in range(0, n - w + 1, w)]
                tgt_means  = [cdf[target].iloc[t:t+w].mean()
                              for t in range(0, n - w + 1, w)]
                if len(feat_means) < 3:
                    continue
                fa, ta = np.array(feat_means), np.array(tgt_means)
                if fa.std() < 1e-10 or ta.std() < 1e-10:
                    continue
                rho, _ = spearmanr(fa, ta)
                if not np.isnan(rho):
                    rhos.append(rho)
            records[feat][f"w{w}_abs_rho"] = (float(np.abs(rhos).mean())
                                               if rhos else 0.0)
            records[feat][f"w{w}_signed"]  = (float(np.mean(rhos))
                                               if rhos else 0.0)

    return pd.DataFrame(records).T.reset_index().rename(columns={"index": "feature"})


# ── 3. Linear models (per-clip normalised) ────────────────────────────────────

def fit_linear_models(df, feature_cols, target, n_splits=5):
    """GroupKFold CV with ElasticNet, Ridge, Lasso.
    Per-clip standardisation: subtract clip mean, divide by clip std.
    Groups = clip_id (no train/val leakage across clips).
    """
    rows = []
    for clip_id, cdf in df.groupby("clip_id"):
        cdf = cdf.sort_values("frame_idx")
        for feat in feature_cols:
            mu, sd = cdf[feat].mean(), cdf[feat].std()
            if sd < 1e-10:
                continue
            rows.append({"clip_id": clip_id, "feat_norm": (cdf[feat].values - mu) / sd,
                         "tgt_norm": (cdf[target].values - cdf[target].mean()) /
                                     (cdf[target].std() + 1e-10),
                         "feat": feat, "n": len(cdf)})

    # Build normalised arrays
    X_list, y_list, groups = [], [], []
    for clip_id, cdf in df.groupby("clip_id"):
        cdf = cdf.sort_values("frame_idx")
        xc  = cdf[feature_cols].values
        mu  = xc.mean(axis=0); sd = xc.std(axis=0)
        sd[sd < 1e-10] = 1.0
        x_norm = (xc - mu) / sd

        yc  = cdf[target].values
        y_mu, y_sd = yc.mean(), yc.std()
        if y_sd < 1e-10:
            continue
        y_norm = (yc - y_mu) / y_sd

        X_list.append(x_norm)
        y_list.append(y_norm)
        groups.extend([clip_id] * len(cdf))

    if not X_list:
        return pd.DataFrame({"feature": feature_cols,
                             "elastic_coef": 0.0, "ridge_coef": 0.0, "lasso_coef": 0.0})

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.float32)
    g = np.array(groups)

    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))

    coefs = {}
    for name, model in [
        ("elastic", ElasticNetCV(cv=gkf, l1_ratio=[0.1, 0.5, 0.9], max_iter=5000)),
        ("ridge",   RidgeCV(cv=gkf)),
        ("lasso",   LassoCV(cv=gkf, max_iter=5000)),
    ]:
        try:
            model.fit(X, y, groups=g)
            coefs[name] = model.coef_
        except Exception:
            coefs[name] = np.zeros(len(feature_cols))

    return pd.DataFrame({
        "feature":      feature_cols,
        "elastic_coef": coefs.get("elastic", np.zeros(len(feature_cols))),
        "ridge_coef":   coefs.get("ridge",   np.zeros(len(feature_cols))),
        "lasso_coef":   coefs.get("lasso",   np.zeros(len(feature_cols))),
    })


# ── 4. PDF report ─────────────────────────────────────────────────────────────

def generate_pdf(summary_df, windowed_df, linear_df, df, feature_cols):
    n_clips  = df["clip_id"].nunique()
    n_frames = len(df)
    best_rho = summary_df["mean_abs_rho"].max()
    best_w30 = (windowed_df.set_index("feature")["w30_abs_rho"].max()
                if "w30_abs_rho" in windowed_df.columns else 0.0)
    best_feat = summary_df.iloc[0]["feature"]

    with PdfPages(OUT_PDF) as pdf:

        # ── Page 1: Top features bar chart (per-frame within-clip |ρ|) ──
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))

        top20 = summary_df.head(20)
        y_pos = range(len(top20))

        ax = axes[0]
        colors = ["#4CAF50" if r > 0.3 else "#FF9800" if r > 0.2 else "#F44336"
                  for r in top20["mean_abs_rho"]]
        ax.barh(y_pos, top20["mean_abs_rho"], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top20["feature"], fontsize=8)
        ax.axvline(0.3, color="green",  ls="--", lw=1, label="|ρ|=0.3 (target)")
        ax.axvline(DETRAC_BEST_ABS_RHO, color="blue", ls=":", lw=1.5,
                   label=f"DETRAC best={DETRAC_BEST_ABS_RHO}")
        ax.axvline(METEOR_BEST_ABS_RHO, color="red",  ls=":", lw=1.5,
                   label=f"METEOR best={METEOR_BEST_ABS_RHO}")
        ax.set_xlabel("Mean within-clip |ρ| vs fn_nano")
        ax.set_title("Top 20 Features — D²-City Within-Clip Spearman")
        ax.legend(fontsize=8)
        ax.invert_yaxis()

        ax = axes[1]
        ax.barh(y_pos, top20["signed_mean_rho"],
                color=["#2196F3" if v > 0 else "#E91E63" for v in top20["signed_mean_rho"]])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top20["feature"], fontsize=8)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("Signed mean ρ (direction of correlation)")
        ax.set_title("Signed Mean ρ — Positive = higher feature → more misses")
        ax.invert_yaxis()

        # Headline verdict
        n_above_03 = (summary_df["mean_abs_rho"] > 0.3).sum()
        verdict    = ("D²-City CONFIRMS dashcam hypothesis" if n_above_03 >= 5
                      else "MARGINAL — features show some within-clip signal"
                      if n_above_03 >= 2 else "WEAK — within-clip signal similar to DETRAC")
        fig.suptitle(f"D²-City: {n_clips} clips, {n_frames:,} sampled frames\n"
                     f"Best |ρ|={best_rho:.3f} ({best_feat}) — "
                     f"{n_above_03} features above 0.3 threshold\n"
                     f"→ {verdict}", fontsize=11, fontweight="bold", y=1.01)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 2: Windowed Spearman (how correlation changes with window size) ──
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        top10 = summary_df.head(10)["feature"].tolist()
        windows = [1, 10, 30, 60, 90]
        wd      = windowed_df.set_index("feature")

        ax = axes[0]
        for feat in top10:
            if feat in wd.index:
                vals = [wd.loc[feat, f"w{w}_abs_rho"] for w in windows]
                ax.plot(windows, vals, "o-", label=feat, alpha=0.8)
        ax.set_xlabel("Window size (sampled frames)")
        ax.set_ylabel("Mean |ρ|")
        ax.axhline(0.3, color="green", ls="--", lw=0.8, label="Target ρ=0.3")
        ax.axhline(DETRAC_BEST_W30_RHO, color="blue", ls=":", lw=0.8,
                   label=f"DETRAC w30={DETRAC_BEST_W30_RHO}")
        ax.set_title("Windowed Spearman — D²-City Top 10 Features")
        ax.legend(fontsize=7, loc="lower right")

        ax = axes[1]
        # Per-clip consistency: distribution of per-clip ρ for top feature
        clip_rhos = []
        for clip_id, cdf in df.groupby("clip_id"):
            if len(cdf) < MIN_FRAMES:
                continue
            x, y = cdf[best_feat].values, cdf[PRIMARY_TARGET].values
            if x.std() < 1e-10 or y.std() < 1e-10:
                continue
            rho, _ = spearmanr(x, y)
            if not np.isnan(rho):
                clip_rhos.append(rho)
        ax.hist(clip_rhos, bins=20, color="#2196F3", alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(clip_rhos), color="red", lw=2,
                   label=f"Mean={np.mean(clip_rhos):.3f}")
        ax.axvline(0.3, color="green", ls="--", lw=1, label="|ρ|=0.3")
        ax.set_xlabel(f"Per-clip ρ ({best_feat} vs fn_nano)")
        ax.set_ylabel("Clip count")
        ax.set_title(f"Per-Clip ρ Distribution — {best_feat}\n"
                     f"({sum(abs(r) > 0.3 for r in clip_rhos)}/{len(clip_rhos)} "
                     f"clips above |0.3|)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # ── Page 3: Linear model coefficients ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 10))
        top30 = summary_df.head(30)["feature"].tolist()
        lin   = linear_df.set_index("feature")

        for ax, col, title in zip(
            axes,
            ["elastic_coef", "ridge_coef", "lasso_coef"],
            ["ElasticNet", "Ridge", "Lasso"],
        ):
            vals = [lin.loc[f, col] if f in lin.index else 0.0 for f in top30]
            y_p  = range(len(top30))
            colors_lin = ["#4CAF50" if v > 0 else "#F44336" for v in vals]
            ax.barh(y_p, vals, color=colors_lin)
            ax.set_yticks(y_p)
            ax.set_yticklabels(top30, fontsize=7)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_xlabel("Normalised Coefficient")
            ax.set_title(f"{title} (per-clip normalised, GroupKFold)")
            ax.invert_yaxis()
        plt.suptitle("D²-City Linear Model Coefficients (Top 30 Features)",
                     fontsize=12, y=1.01)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 4: Feature-feature correlation heatmap (top 25 + target) ──
        fig, ax = plt.subplots(figsize=(14, 12))
        top25 = summary_df.head(25)["feature"].tolist()
        corr_data = []
        for clip_id, cdf in df.groupby("clip_id"):
            if len(cdf) >= MIN_FRAMES:
                corr_data.append(cdf[top25 + [PRIMARY_TARGET]].values)
        if corr_data:
            all_data = np.vstack(corr_data)
            rho_mat, _ = spearmanr(all_data)
            labels_hm = top25 + [PRIMARY_TARGET]
            im = ax.imshow(np.abs(rho_mat), cmap="YlOrRd", vmin=0, vmax=1)
            ax.set_xticks(range(len(labels_hm)))
            ax.set_xticklabels(labels_hm, rotation=90, fontsize=6)
            ax.set_yticks(range(len(labels_hm)))
            ax.set_yticklabels(labels_hm, fontsize=6)
            plt.colorbar(im, ax=ax, shrink=0.6)
            ax.set_title("Within-Clip Feature Correlation Matrix |ρ|\n"
                         "(Top 25 + target; pooled across clips)")
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # ── Page 5: Multi-target comparison ──
        fig, axes = plt.subplots(1, len(SECONDARY_TARGETS) + 1, figsize=(18, 10))
        targets_all = [PRIMARY_TARGET] + SECONDARY_TARGETS
        top15 = summary_df.head(15)["feature"].tolist()

        for ax, tgt in zip(axes, targets_all):
            rho_t = []
            for feat in top15:
                rhos_t = []
                for clip_id, cdf in df.groupby("clip_id"):
                    if len(cdf) < MIN_FRAMES:
                        continue
                    x, y = cdf[feat].values, cdf[tgt].values
                    if x.std() < 1e-10 or y.std() < 1e-10:
                        continue
                    rho, _ = spearmanr(x, y)
                    if not np.isnan(rho):
                        rhos_t.append(abs(rho))
                rho_t.append(np.mean(rhos_t) if rhos_t else 0.0)
            y_p = range(len(top15))
            ax.barh(y_p, rho_t, color="#607D8B")
            ax.set_yticks(y_p)
            ax.set_yticklabels(top15, fontsize=7)
            ax.set_xlabel("Mean |ρ|")
            ax.set_title(f"Target: {tgt}")
            ax.axvline(0.3, color="green", ls="--", lw=0.8)
            ax.invert_yaxis()
        plt.suptitle("D²-City: Multi-Target Comparison (Top 15 Features)",
                     fontsize=12, y=1.01)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 6: Global vs within-clip scatter (Simpson check) ──
        fig, ax = plt.subplots(figsize=(10, 10))
        global_rhos = []
        within_rhos = []
        labels_scatter = []
        for feat in feature_cols[:40]:
            x_all = df[feat].values; y_all = df[PRIMARY_TARGET].values
            if x_all.std() < 1e-10:
                continue
            g_rho, _ = spearmanr(x_all, y_all)
            row       = summary_df[summary_df["feature"] == feat]
            if row.empty:
                continue
            w_rho = float(row["signed_mean_rho"].values[0])
            global_rhos.append(g_rho); within_rhos.append(w_rho)
            labels_scatter.append(feat)
        ax.scatter(global_rhos, within_rhos, alpha=0.7, c="#2196F3", s=60)
        for i, lab in enumerate(labels_scatter):
            ax.annotate(lab, (global_rhos[i], within_rhos[i]),
                        fontsize=6, alpha=0.7)
        lim = max(abs(v) for v in global_rhos + within_rhos) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y = x")
        ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
        ax.set_xlabel("Global Spearman ρ (pooled)")
        ax.set_ylabel("Within-clip mean signed ρ")
        ax.set_title("Global vs Within-Clip ρ (D²-City)\n"
                     "Points below diagonal: Simpson's paradox — "
                     "global overestimates within-seq signal")
        ax.legend(fontsize=8)
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # ── Page 7: Summary table + cross-dataset comparison ──
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.axis("off")

        top15_feat = summary_df.head(15)["feature"].tolist()
        wd_idx     = windowed_df.set_index("feature")

        header = ["Feature", "Mean|ρ|", "Signed ρ", "%Sig", "W10|ρ|",
                  "W30|ρ|", "W60|ρ|", "ElNet"]
        rows_t = []
        for feat in top15_feat:
            row = summary_df[summary_df["feature"] == feat].iloc[0]
            lin_row = linear_df[linear_df["feature"] == feat]
            el = float(lin_row["elastic_coef"].values[0]) if len(lin_row) else 0.0
            w  = wd_idx.loc[feat] if feat in wd_idx.index else pd.Series()
            rows_t.append([
                feat,
                f"{row['mean_abs_rho']:.4f}",
                f"{row['signed_mean_rho']:+.4f}",
                f"{row['pct_significant']:.0f}%",
                f"{w.get('w10_abs_rho', 0):.4f}",
                f"{w.get('w30_abs_rho', 0):.4f}",
                f"{w.get('w60_abs_rho', 0):.4f}",
                f"{el:+.4f}",
            ])

        tbl = ax.table(cellText=rows_t, colLabels=header,
                       loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.8)
        for j in range(len(header)):
            tbl[0, j].set_facecolor("#37474F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # Cross-dataset comparison
        comp_text = (
            f"\n\nCross-Dataset Comparison (best within-clip |ρ| vs fn_nano / fn_coco)\n"
            f"{'─'*70}\n"
            f"  DETRAC (fixed camera, 100 seqs, new_features 65-feat)  "
            f"best |ρ|={DETRAC_BEST_ABS_RHO:.3f}  w30={DETRAC_BEST_W30_RHO:.3f}\n"
            f"  METEOR (dashcam, 50 clips, 41-feat)                    "
            f"best |ρ|={METEOR_BEST_ABS_RHO:.3f}\n"
            f"  D²-City (dashcam, {n_clips} clips, 65-feat)           "
            f"best |ρ|={best_rho:.3f}  w30={best_w30:.3f}\n"
            f"{'─'*70}\n"
            f"  {'D²-City > DETRAC: dashcam hypothesis CONFIRMED' if best_rho > DETRAC_BEST_ABS_RHO else 'D²-City ≤ DETRAC: within-clip signal not stronger than fixed cam'}\n"
            f"  {'D²-City > METEOR: Chinese traffic has more predictable patterns than METEOR' if best_rho > METEOR_BEST_ABS_RHO else 'D²-City ≈ METEOR: within-clip signal comparable across dashcam datasets'}\n"
            f"\n  Success criterion (from D2City_validation_prompt.md):\n"
            f"    ≥5 features with within-clip |ρ| > 0.3 → proceed to LSTM training\n"
            f"    Current: {(summary_df['mean_abs_rho'] > 0.3).sum()} features above 0.3\n"
        )
        ax.text(0.01, 0.20, comp_text, transform=ax.transAxes,
                fontfamily="monospace", fontsize=8, verticalalignment="top")
        ax.set_title("D²-City Analysis Summary", fontsize=14,
                     fontweight="bold", pad=40)
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

    print(f"  PDF: {OUT_PDF}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("D²-CITY CORRELATION ANALYSIS")
    print("=" * 70)

    if not FEATURES_CSV.exists():
        print(f"\n  ERROR: {FEATURES_CSV} not found. Run D2_extract_features.py first.")
        return

    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    for c in feature_cols:
        if "temporal" in c or c == "motion_pixel_ratio":
            df[c] = df[c].fillna(0.0)

    print(f"\n  {len(df):,} frames, {df['clip_id'].nunique()} clips, "
          f"{len(feature_cols)} features")
    print(f"  fn_nano: mean={df['fn_nano'].mean():.2f}, "
          f"std={df['fn_nano'].std():.2f}, max={df['fn_nano'].max()}")

    # Step 1: Within-clip Spearman
    print("\n[1/4] Within-clip Spearman (per-frame) ...", flush=True)
    summary_df = within_clip_spearman(df, feature_cols, PRIMARY_TARGET)

    print(f"\n  Top 15 features by mean |ρ|:")
    print(f"  {'Feature':<35s} {'mean|ρ|':>8s} {'signed ρ':>9s} "
          f"{'%sig':>6s} {'n_clips':>8s}")
    print(f"  {'-'*70}")
    for _, row in summary_df.head(15).iterrows():
        print(f"  {row['feature']:<35s} {row['mean_abs_rho']:8.4f} "
              f"{row['signed_mean_rho']:+9.4f} {row['pct_significant']:6.0f}% "
              f"{int(row['n_valid_clips']):>8d}")

    n_above_03 = (summary_df["mean_abs_rho"] > 0.3).sum()
    print(f"\n  Features with |ρ| > 0.3: {n_above_03}")
    if n_above_03 >= 5:
        print("  ✓ SUCCESS — dashcam hypothesis confirmed. Proceed to LSTM training.")
    elif n_above_03 >= 2:
        print("  ⚠ MARGINAL — some signal but below 5-feature threshold.")
    else:
        print("  ✗ WEAK — within-clip signal not significantly better than DETRAC.")

    # Step 2: Windowed Spearman
    print("\n[2/4] Windowed Spearman (w=1,10,30,60,90) ...", flush=True)
    windowed_df = windowed_spearman(df, feature_cols, PRIMARY_TARGET)
    best_feat   = summary_df.iloc[0]["feature"]
    if "w30_abs_rho" in windowed_df.columns:
        wd      = windowed_df.set_index("feature")
        w30_top = wd.loc[best_feat, "w30_abs_rho"] if best_feat in wd.index else 0.0
        print(f"  Best feature at w=30: {best_feat} |ρ|={w30_top:.4f}")

    # Step 3: Linear models
    print("\n[3/4] Linear models (ElasticNet, Ridge, Lasso, GroupKFold) ...",
          flush=True)
    linear_df = fit_linear_models(df, feature_cols, PRIMARY_TARGET)
    top5_el = (linear_df.assign(abs_coef=linear_df["elastic_coef"].abs())
               .sort_values("abs_coef", ascending=False).head(5))
    print("  Top 5 ElasticNet features:")
    for _, row in top5_el.iterrows():
        print(f"    {row['feature']:<40s} {row['elastic_coef']:+.4f}")

    # Step 4: Save summary CSV + PDF
    print("\n[4/4] Saving CSV + PDF ...", flush=True)
    # Merge windowed ρ into summary
    merged = summary_df.merge(windowed_df, on="feature", how="left")
    merged = merged.merge(linear_df, on="feature", how="left")
    merged.to_csv(OUT_SUMMARY, index=False)
    print(f"  CSV: {OUT_SUMMARY}")

    generate_pdf(summary_df, windowed_df, linear_df, df, feature_cols)

    print(f"\n{'=' * 70}")
    print("D²-CITY ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Best feature      : {best_feat} (mean |ρ|={summary_df.iloc[0]['mean_abs_rho']:.4f})")
    print(f"  Features > 0.3    : {n_above_03}")
    print(f"  Proceed to LSTM   : {'YES' if n_above_03 >= 5 else 'MARGINAL' if n_above_03 >= 2 else 'NO'}")


if __name__ == "__main__":
    main()
