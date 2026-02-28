#!/usr/bin/env python3
"""Feature comparison: DETRAC-65 vs Jay's-48 on VisDrone.

Analyses:
  1. Spearman ρ of each feature vs future fn_nano (train-only)
  2. Lasso (L1-logistic) to identify surviving features from each set
  3. Overlap check — features with the same/similar name across sets

Output: viss_feature_comparison.pdf
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ECHO_DIR = SCRIPT_DIR.parent
EVAL_CSV = ECHO_DIR / "yolo_runs_on_uavdt" / "visdrone_classagnostic_results.csv"
DETRAC_FEAT_CSV = SCRIPT_DIR / "viss_detrac65_features.csv"
JAYS_FEAT_CSV = ECHO_DIR / "yolo_runs_on_uavdt" / "jays" / "jays_visdrone_features.csv"
OUT_PDF = SCRIPT_DIR / "viss_feature_comparison.pdf"

VAL_SEQS = [
    "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
    "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v",
]

# Jay's features: columns that are all-NaN
DROP_COLS = [
    "GLCM_contrast", "GLCM_homogeneity", "GLCM_energy", "GLCM_entropy",
    "GLCM_correlation", "LBP_entropy", "LBP_uniformity", "wavelet_detail_energy",
]

WINDOW = 15
HORIZON = 15
META_COLS = ["sequence", "frame_id", "tp_nano", "fp_nano", "fn_nano",
             "num_gt", "tp_medium"]


# ── Data ────────────────────────────────────────────────────────────────────


def load_eval():
    """Load eval results (shared by both feature sets)."""
    ev = pd.read_csv(EVAL_CSV)
    nano = ev[ev["model"] == "yolo11n.pt"][
        ["sequence", "frame_id", "tp", "fp", "fn", "num_gt"]
    ].copy()
    nano.columns = ["sequence", "frame_id", "tp_nano", "fp_nano", "fn_nano", "num_gt"]
    med = ev[ev["model"] == "yolo11m.pt"][["sequence", "frame_id", "tp"]].copy()
    med.columns = ["sequence", "frame_id", "tp_medium"]
    return nano.merge(med, on=["sequence", "frame_id"])


def load_features(eval_merged):
    """Load and merge both feature sets, return unified dataframe + column lists."""
    # DETRAC-65
    detrac = pd.read_csv(DETRAC_FEAT_CSV).fillna(0.0)
    detrac_feat_cols = [c for c in detrac.columns if c not in ("sequence", "frame_id")]
    # Prefix to avoid name collisions
    detrac_renamed = detrac.rename(
        columns={c: f"D_{c}" for c in detrac_feat_cols}
    )
    detrac_feat_cols_prefixed = [f"D_{c}" for c in detrac_feat_cols]

    # Jay's-48
    jays = pd.read_csv(JAYS_FEAT_CSV)
    jays = jays.drop(columns=[c for c in DROP_COLS if c in jays.columns], errors="ignore")
    jays = jays.fillna(0.0)
    jays_feat_cols = [c for c in jays.columns if c not in ("sequence", "frame_id")]
    jays_renamed = jays.rename(
        columns={c: f"J_{c}" for c in jays_feat_cols}
    )
    jays_feat_cols_prefixed = [f"J_{c}" for c in jays_feat_cols]

    # Merge both with eval on (sequence, frame_id)
    df = eval_merged.merge(detrac_renamed, on=["sequence", "frame_id"], how="inner")
    df = df.merge(jays_renamed, on=["sequence", "frame_id"], how="inner")

    print(f"  Merged: {len(df)} frames, {df['sequence'].nunique()} sequences")
    print(f"  DETRAC features: {len(detrac_feat_cols_prefixed)}")
    print(f"  Jay's features:  {len(jays_feat_cols_prefixed)}")

    return df, detrac_feat_cols_prefixed, jays_feat_cols_prefixed


def compute_threshold(df):
    """Median threshold over all sequences."""
    all_targets = []
    for seq, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        fn = sdf["fn_nano"].values.astype(np.float32)
        n = len(sdf)
        total_need = WINDOW + HORIZON
        for t in range(0, n - total_need + 1, WINDOW):
            all_targets.append(fn[t + WINDOW:t + WINDOW + HORIZON].mean())
    return float(np.median(all_targets))


def build_windows(df, feature_cols, threshold, train_only=True):
    """Build windowed feature matrix + binary targets (train sequences only)."""
    if train_only:
        df = df[~df["sequence"].isin(VAL_SEQS)]
    feats, targets = [], []
    for seq, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        fn = sdf["fn_nano"].values.astype(np.float32)
        n = len(sdf)
        total_need = WINDOW + HORIZON
        for t in range(0, n - total_need + 1, WINDOW):
            x = sdf[feature_cols].iloc[t:t + WINDOW].mean().values
            future_fn = fn[t + WINDOW:t + WINDOW + HORIZON].mean()
            feats.append(x)
            targets.append(1 if future_fn >= threshold else 0)
    return np.array(feats), np.array(targets)


# ── Analysis ────────────────────────────────────────────────────────────────


def spearman_analysis(X, y, col_names):
    """Spearman ρ for each feature vs target."""
    results = {}
    for j, col in enumerate(col_names):
        rho, pval = spearmanr(X[:, j], y)
        results[col] = (rho, pval)
    return results


def lasso_analysis(X, y):
    """L1-penalized logistic regression with CV to select alpha."""
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    clf = LogisticRegressionCV(
        penalty="l1", solver="saga", Cs=20, cv=5,
        max_iter=5000, random_state=42, scoring="accuracy",
    )
    clf.fit(X_sc, y)
    return clf.coef_.ravel(), clf.C_[0]


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    print("Loading data ...", flush=True)
    eval_merged = load_eval()
    df, detrac_cols, jays_cols = load_features(eval_merged)
    all_cols = detrac_cols + jays_cols

    threshold = compute_threshold(df)
    print(f"Threshold: {threshold:.3f}")

    # Build windows (train only)
    X_all, y = build_windows(df, all_cols, threshold, train_only=True)
    n_detrac = len(detrac_cols)
    n_jays = len(jays_cols)
    X_detrac = X_all[:, :n_detrac]
    X_jays = X_all[:, n_detrac:]
    print(f"  {len(y)} train windows  (hard={y.mean():.1%})")

    # ── 1. Spearman ──────────────────────────────────────────────────────

    print("\n=== Spearman Correlation ===", flush=True)
    detrac_corrs = spearman_analysis(X_detrac, y, detrac_cols)
    jays_corrs = spearman_analysis(X_jays, y, jays_cols)

    # Combine and rank
    all_corrs = {**detrac_corrs, **jays_corrs}
    ranked = sorted(all_corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)

    print(f"\n  Top 30 features by |ρ| (D_=DETRAC, J_=Jay's):")
    for feat, (rho, pval) in ranked[:30]:
        src = "DETRAC" if feat.startswith("D_") else "Jay's"
        print(f"    {feat:45s}  ρ={rho:+.4f}  ({src})")

    # Stats
    detrac_rhos = [abs(v[0]) for v in detrac_corrs.values()]
    jays_rhos = [abs(v[0]) for v in jays_corrs.values()]
    print(f"\n  DETRAC-65: mean|ρ|={np.mean(detrac_rhos):.4f}  "
          f"max|ρ|={np.max(detrac_rhos):.4f}  "
          f"features with |ρ|>0.1: {sum(1 for r in detrac_rhos if r > 0.1)}")
    print(f"  Jay's-48:  mean|ρ|={np.mean(jays_rhos):.4f}  "
          f"max|ρ|={np.max(jays_rhos):.4f}  "
          f"features with |ρ|>0.1: {sum(1 for r in jays_rhos if r > 0.1)}")

    # ── 2. Lasso ─────────────────────────────────────────────────────────

    print("\n=== Lasso (L1-Logistic) ===", flush=True)

    # Combined Lasso
    print("  Fitting combined Lasso (all features) ...", flush=True)
    coefs_all, C_all = lasso_analysis(X_all, y)
    print(f"  Best C={C_all:.4f}")

    # Separate by source
    detrac_coefs = coefs_all[:n_detrac]
    jays_coefs = coefs_all[n_detrac:]

    detrac_surviving = [(col, coef) for col, coef in zip(detrac_cols, detrac_coefs) if abs(coef) > 1e-6]
    jays_surviving = [(col, coef) for col, coef in zip(jays_cols, jays_coefs) if abs(coef) > 1e-6]

    print(f"\n  Surviving features (non-zero coefs):")
    print(f"    DETRAC: {len(detrac_surviving)}/{n_detrac}")
    print(f"    Jay's:  {len(jays_surviving)}/{n_jays}")

    all_surviving = sorted(
        [(c, v, "DETRAC") for c, v in detrac_surviving] +
        [(c, v, "Jay's") for c, v in jays_surviving],
        key=lambda x: abs(x[1]), reverse=True
    )
    print(f"\n  Top surviving features by |coef|:")
    for col, coef, src in all_surviving[:25]:
        print(f"    {col:45s}  coef={coef:+.4f}  ({src})")

    # ── 3. Overlap check ─────────────────────────────────────────────────

    print("\n=== Feature Name Overlap ===", flush=True)
    detrac_base = {c.replace("D_", ""): c for c in detrac_cols}
    jays_base = {c.replace("J_", ""): c for c in jays_cols}
    overlap_names = set(detrac_base.keys()) & set(jays_base.keys())
    print(f"  {len(overlap_names)} features with identical names in both sets:")

    overlap_data = []
    for name in sorted(overlap_names):
        d_col = detrac_base[name]
        j_col = jays_base[name]
        d_rho = detrac_corrs[d_col][0]
        j_rho = jays_corrs[j_col][0]
        # Pearson between the two versions
        pearson = np.corrcoef(X_all[:, all_cols.index(d_col)],
                              X_all[:, all_cols.index(j_col)])[0, 1]
        print(f"    {name:35s}  DETRAC ρ={d_rho:+.4f}  Jay's ρ={j_rho:+.4f}  "
              f"Pearson(D,J)={pearson:+.4f}")
        overlap_data.append((name, d_rho, j_rho, pearson))

    # ── PDF ──────────────────────────────────────────────────────────────

    print(f"\nGenerating PDF ...", flush=True)
    with PdfPages(str(OUT_PDF)) as pdf:

        # Page 1: Spearman ρ — top 25 from each set side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))

        # DETRAC top 25
        ax = axes[0]
        detrac_sorted = sorted(detrac_corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)
        top_d = detrac_sorted[:25]
        names_d = [f[0].replace("D_", "") for f, _ in top_d]
        rhos_d = [v[0] for _, v in top_d]
        colors_d = ["#4C72B0" if r > 0 else "#C44E52" for r in rhos_d]
        ax.barh(range(len(top_d)), rhos_d, color=colors_d)
        ax.set_yticks(range(len(top_d)))
        ax.set_yticklabels(names_d, fontsize=7)
        ax.set_xlabel("Spearman ρ")
        ax.set_title("DETRAC-65 Features (top 25 by |ρ|)")
        ax.axvline(0, color="black", lw=0.5)
        ax.invert_yaxis()

        # Jay's top 25
        ax = axes[1]
        jays_sorted = sorted(jays_corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)
        top_j = jays_sorted[:25]
        names_j = [f[0].replace("J_", "") for f, _ in top_j]
        rhos_j = [v[0] for _, v in top_j]
        colors_j = ["#55A868" if r > 0 else "#C44E52" for r in rhos_j]
        ax.barh(range(len(top_j)), rhos_j, color=colors_j)
        ax.set_yticks(range(len(top_j)))
        ax.set_yticklabels(names_j, fontsize=7)
        ax.set_xlabel("Spearman ρ")
        ax.set_title("Jay's-48 Features (top 25 by |ρ|)")
        ax.axvline(0, color="black", lw=0.5)
        ax.invert_yaxis()

        plt.suptitle("Spearman ρ vs Future fn_nano (train windows only)", fontsize=13, y=1.01)
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 2: Combined ranking — top 40, color by source
        fig, ax = plt.subplots(figsize=(12, 10))
        top40 = ranked[:40]
        names40 = [f[0].replace("D_", "").replace("J_", "") for f in top40]
        rhos40 = [v[0] for _, v in top40]
        src40 = ["DETRAC" if f[0].startswith("D_") else "Jay's" for f in top40]
        colors40 = ["#4C72B0" if s == "DETRAC" else "#55A868" for s in src40]
        bars = ax.barh(range(len(top40)), rhos40, color=colors40)
        ax.set_yticks(range(len(top40)))
        labels40 = [f"{n}  [{s[0]}]" for n, s in zip(names40, src40)]
        ax.set_yticklabels(labels40, fontsize=7)
        ax.set_xlabel("Spearman ρ")
        ax.set_title("Combined Ranking: Top 40 Features by |ρ|\n(Blue=DETRAC, Green=Jay's)")
        ax.axvline(0, color="black", lw=0.5)
        ax.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 3: Lasso coefficients — non-zero features
        if all_surviving:
            fig, ax = plt.subplots(figsize=(12, max(6, len(all_surviving) * 0.35)))
            show_n = min(40, len(all_surviving))
            show = all_surviving[:show_n]
            names_l = [c.replace("D_", "").replace("J_", "") for c, _, _ in show]
            coefs_l = [v for _, v, _ in show]
            src_l = [s for _, _, s in show]
            colors_l = ["#4C72B0" if s == "DETRAC" else "#55A868" for s in src_l]
            labels_l = [f"{n}  [{s[0]}]" for n, s in zip(names_l, src_l)]
            ax.barh(range(show_n), coefs_l, color=colors_l)
            ax.set_yticks(range(show_n))
            ax.set_yticklabels(labels_l, fontsize=7)
            ax.set_xlabel("Lasso Coefficient")
            ax.set_title(f"L1-Logistic Surviving Features ({len(detrac_surviving)}D + "
                         f"{len(jays_surviving)}J out of {n_detrac}+{n_jays})\n"
                         f"(Blue=DETRAC, Green=Jay's)")
            ax.axvline(0, color="black", lw=0.5)
            ax.invert_yaxis()
            plt.tight_layout()
            pdf.savefig(fig); plt.close()

        # Page 4: Overlap heatmap
        if overlap_data:
            fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(overlap_data) * 0.5)))

            ol_names = [d[0] for d in overlap_data]
            ol_d_rho = [d[1] for d in overlap_data]
            ol_j_rho = [d[2] for d in overlap_data]
            ol_pearson = [d[3] for d in overlap_data]

            # Rho comparison
            ax = axes[0]
            y_pos = np.arange(len(ol_names))
            ax.barh(y_pos - 0.2, ol_d_rho, 0.4, label="DETRAC", color="#4C72B0")
            ax.barh(y_pos + 0.2, ol_j_rho, 0.4, label="Jay's", color="#55A868")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(ol_names, fontsize=7)
            ax.set_xlabel("Spearman ρ")
            ax.set_title("Same-Name Features: ρ Comparison")
            ax.legend(fontsize=8)
            ax.axvline(0, color="black", lw=0.5)
            ax.invert_yaxis()

            # Pearson between versions
            ax = axes[1]
            colors_p = ["#DD8452" if abs(p) > 0.9 else "#CCCCCC" for p in ol_pearson]
            ax.barh(y_pos, ol_pearson, color=colors_p)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(ol_names, fontsize=7)
            ax.set_xlabel("Pearson r (DETRAC vs Jay's extraction)")
            ax.set_title("Are They Measuring the Same Thing?\n(Orange = high correlation)")
            ax.axvline(0.9, color="red", ls="--", lw=0.8, label="r=0.9")
            ax.axvline(-0.9, color="red", ls="--", lw=0.8)
            ax.legend(fontsize=8)
            ax.invert_yaxis()

            plt.tight_layout()
            pdf.savefig(fig); plt.close()

    print(f"\nSaved PDF: {OUT_PDF}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("SUMMARY — Feature Comparison")
    print(f"{'='*60}")
    print(f"DETRAC-65:  mean|ρ|={np.mean(detrac_rhos):.4f}  max|ρ|={np.max(detrac_rhos):.4f}  "
          f"|ρ|>0.1: {sum(1 for r in detrac_rhos if r > 0.1)}/{n_detrac}")
    print(f"Jay's-48:   mean|ρ|={np.mean(jays_rhos):.4f}  max|ρ|={np.max(jays_rhos):.4f}  "
          f"|ρ|>0.1: {sum(1 for r in jays_rhos if r > 0.1)}/{n_jays}")
    print(f"\nLasso surviving: {len(detrac_surviving)}/{n_detrac} DETRAC  "
          f"{len(jays_surviving)}/{n_jays} Jay's")
    print(f"Overlapping names: {len(overlap_names)}")
    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
