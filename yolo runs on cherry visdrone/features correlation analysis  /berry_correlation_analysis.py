"""Correlation analysis: 37 per-frame features vs YOLO recall."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
FEAT_CSV = ROOT / "berry_frame_features.csv"
RES_CSV = ROOT / "cherry_results.csv"
OUT_PDF = ROOT / "berry_correlation_results.pdf"
OUT_CSV = ROOT / "berry_correlation_summary.csv"

MODELS = {"yolo11n.pt": "nano", "yolo11s.pt": "small", "yolo11m.pt": "medium"}
TARGETS = ["recall_nano", "recall_small", "recall_medium", "recall_gap"]


# ── Preprocessing ──────────────────────────────────────────────────
def load_and_merge():
    feat = pd.read_csv(FEAT_CSV)
    res = pd.read_csv(RES_CSV)

    # Pivot recall by model
    recall_pivot = res.pivot_table(
        index=["sequence", "frame_id"], columns="model", values="recall"
    ).reset_index()
    recall_pivot.columns.name = None
    recall_pivot = recall_pivot.rename(columns={
        "yolo11n.pt": "recall_nano",
        "yolo11s.pt": "recall_small",
        "yolo11m.pt": "recall_medium",
    })
    recall_pivot["recall_gap"] = recall_pivot["recall_medium"] - recall_pivot["recall_nano"]

    # gt_count from nano row (tp + fn)
    nano = res[res["model"] == "yolo11n.pt"][["sequence", "frame_id", "tp", "fn"]].copy()
    nano["gt_count"] = nano["tp"] + nano["fn"]
    recall_pivot = recall_pivot.merge(nano[["sequence", "frame_id", "gt_count"]],
                                      on=["sequence", "frame_id"])

    # Merge with features
    df = feat.merge(recall_pivot, on=["sequence", "frame_id"])

    # Filter sparse frames
    df = df[df["gt_count"] >= 3].reset_index(drop=True)

    feature_cols = [c for c in feat.columns if c not in ("sequence", "frame_id")]
    print(f"Merged: {len(df)} frames, {len(feature_cols)} features (dropped gt_count<3)")
    return df, feature_cols


# ── GT-weighted Spearman ───────────────────────────────────────────
def weighted_spearman(x, y, weights):
    """Spearman via repeating samples by integer weights."""
    w = np.round(weights).astype(int)
    w = np.maximum(w, 1)
    x_rep = np.repeat(x, w)
    y_rep = np.repeat(y, w)
    rho, p = stats.spearmanr(x_rep, y_rep)
    return rho, p


def compute_correlations(df, feature_cols):
    """Returns DataFrame: feature_name, rho_nano, ..., rho_gap, p_value_gap."""
    rows = []
    w = df["gt_count"].values
    for feat in feature_cols:
        x = df[feat].values
        valid = ~np.isnan(x)
        if valid.sum() < 30:
            rows.append({"feature_name": feat,
                         **{f"rho_{t.split('_',1)[1]}": np.nan for t in TARGETS},
                         "p_value_gap": np.nan})
            continue
        xv, wv = x[valid], w[valid]
        rec = {"feature_name": feat}
        for target in TARGETS:
            yv = df[target].values[valid]
            rho, p = weighted_spearman(xv, yv, wv)
            short = target.split("_", 1)[1]
            rec[f"rho_{short}"] = rho
            if target == "recall_gap":
                rec["p_value_gap"] = p
        rows.append(rec)
    return pd.DataFrame(rows)


# ── Random Forest importance ───────────────────────────────────────
def rf_importance(df, feature_cols):
    """Permutation importance of 37 features → recall_gap."""
    sub = df[feature_cols + ["recall_gap"]].dropna()
    X = sub[feature_cols].values
    y = sub["recall_gap"].values
    print(f"RF training on {len(sub)} rows (after dropping NaN)")

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    result = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    return dict(zip(feature_cols, result.importances_mean))


# ── Plotting ───────────────────────────────────────────────────────
def plot_heatmap(corr_df, pdf):
    """Page 1: Spearman rho heatmap (37 × 4)."""
    fig, ax = plt.subplots(figsize=(8, 14))
    rho_cols = ["rho_nano", "rho_small", "rho_medium", "rho_gap"]
    mat = corr_df.set_index("feature_name")[rho_cols].values.astype(float)
    labels_y = corr_df["feature_name"].tolist()

    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Nano", "Small", "Medium", "Gap"], fontsize=9)
    ax.set_yticks(range(len(labels_y)))
    ax.set_yticklabels(labels_y, fontsize=7)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.3 else "black")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Spearman rho")
    ax.set_title("GT-Weighted Spearman Correlation: Features vs Recall", fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def plot_top15(corr_df, pdf):
    """Page 2: Top 15 features by |rho| with recall_gap."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sub = corr_df.dropna(subset=["rho_gap"]).copy()
    sub["abs_rho"] = sub["rho_gap"].abs()
    sub = sub.nlargest(15, "abs_rho").sort_values("abs_rho")

    colors = []
    for v in sub["abs_rho"]:
        if v > 0.3:
            colors.append("#e74c3c")
        elif v > 0.15:
            colors.append("#e67e22")
        else:
            colors.append("#95a5a6")

    ax.barh(range(len(sub)), sub["abs_rho"], color=colors)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["feature_name"], fontsize=8)
    ax.set_xlabel("|Spearman rho| with Recall Gap")
    ax.set_title("Top 15 Features Correlated with Recall Gap (Medium-Nano)")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e74c3c", label="Strong (>0.3)"),
                       Patch(color="#e67e22", label="Moderate (>0.15)"),
                       Patch(color="#95a5a6", label="Weak")],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def plot_temporal_horizon(corr_df, pdf):
    """Page 3: |rho| vs window size for temporal features."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Optical flow windows
    flow_feats = ["flow_mean_5f", "flow_mean_10f", "flow_mean_15f", "flow_mean_30f"]
    flow_wins = [5, 10, 15, 30]
    for target, label, color in [("rho_nano", "Nano", "#2ecc71"),
                                  ("rho_medium", "Medium", "#3498db")]:
        vals = []
        for feat in flow_feats:
            row = corr_df[corr_df["feature_name"] == feat]
            vals.append(abs(row[target].values[0]) if len(row) else np.nan)
        ax1.plot(flow_wins, vals, "o-", label=label, color=color, linewidth=2)
    ax1.set_xlabel("Window Size (frames)")
    ax1.set_ylabel("|Spearman rho|")
    ax1.set_title("Optical Flow Rolling Mean")
    ax1.legend()
    ax1.set_xticks(flow_wins)

    # Brightness variance windows
    bv_feats = ["brightness_var_5f", "brightness_var_10f", "brightness_var_30f"]
    bv_wins = [5, 10, 30]
    for target, label, color in [("rho_nano", "Nano", "#2ecc71"),
                                  ("rho_medium", "Medium", "#3498db")]:
        vals = []
        for feat in bv_feats:
            row = corr_df[corr_df["feature_name"] == feat]
            vals.append(abs(row[target].values[0]) if len(row) else np.nan)
        ax2.plot(bv_wins, vals, "o-", label=label, color=color, linewidth=2)
    ax2.set_xlabel("Window Size (frames)")
    ax2.set_ylabel("|Spearman rho|")
    ax2.set_title("Brightness Variance")
    ax2.legend()
    ax2.set_xticks(bv_wins)

    fig.suptitle("Temporal Horizon Analysis: Correlation Strength vs Window Size", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def plot_redundancy(df, feature_cols, pdf):
    """Page 4: 37×37 pairwise Spearman heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))
    feat_data = df[feature_cols].dropna()
    corr_mat = feat_data.corr(method="spearman").values
    n = len(feature_cols)

    im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(feature_cols, fontsize=5, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(feature_cols, fontsize=5)

    # Mark redundant pairs
    redundant = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_mat[i, j]) > 0.85:
                ax.plot(j, i, "kx", markersize=6, markeredgewidth=1.5)
                redundant.append((feature_cols[i], feature_cols[j], corr_mat[i, j]))

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("Feature Pairwise Spearman Correlation (x = |rho|>0.85)", fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    print(f"\nRedundant pairs (|rho| > 0.85): {len(redundant)}")
    for a, b, r in sorted(redundant, key=lambda x: -abs(x[2])):
        print(f"  {a:<35s} <-> {b:<35s}  rho={r:+.3f}")


def plot_rf_vs_spearman(corr_df, rf_imp, pdf):
    """Page 5: Scatter — Spearman |rho| vs RF importance."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sub = corr_df.dropna(subset=["rho_gap"]).copy()
    sub["abs_rho"] = sub["rho_gap"].abs()
    sub["rf_imp"] = sub["feature_name"].map(rf_imp).fillna(0)

    ax.scatter(sub["abs_rho"], sub["rf_imp"], s=40, alpha=0.7, c="#3498db")
    for _, row in sub.iterrows():
        ax.annotate(row["feature_name"], (row["abs_rho"], row["rf_imp"]),
                    fontsize=6, alpha=0.8, textcoords="offset points", xytext=(4, 4))

    ax.set_xlabel("|Spearman rho| with Recall Gap")
    ax.set_ylabel("Random Forest Permutation Importance")
    ax.set_title("Spearman vs RF Feature Importance for Recall Gap")
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────
def main():
    df, feature_cols = load_and_merge()

    # Analysis 1 & 2: Spearman correlations
    print("Computing GT-weighted Spearman correlations...")
    corr_df = compute_correlations(df, feature_cols)

    # Print top features ranking
    ranked = corr_df.dropna(subset=["rho_gap"]).copy()
    ranked["abs_rho"] = ranked["rho_gap"].abs()
    ranked = ranked.sort_values("abs_rho", ascending=False)
    print("\n=== Top Features by |rho| with Recall Gap ===")
    for _, row in ranked.iterrows():
        strength = "STRONG" if row["abs_rho"] > 0.3 else "moderate" if row["abs_rho"] > 0.15 else "weak"
        print(f"  {row['feature_name']:<35s} rho={row['rho_gap']:+.4f}  {strength}")

    # Analysis 5: Random Forest
    print("\nTraining Random Forest...")
    rf_imp = rf_importance(df, feature_cols)

    # Add RF importance to summary
    corr_df["rf_importance"] = corr_df["feature_name"].map(rf_imp)

    # Save summary CSV
    corr_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    # Generate PDF
    print("Generating PDF...")
    with PdfPages(str(OUT_PDF)) as pdf:
        plot_heatmap(corr_df, pdf)
        plot_top15(corr_df, pdf)
        plot_temporal_horizon(corr_df, pdf)
        plot_redundancy(df, feature_cols, pdf)
        plot_rf_vs_spearman(corr_df, rf_imp, pdf)

    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
