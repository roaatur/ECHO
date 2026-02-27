#!/usr/bin/env python3
"""Feature selection: redundancy pruning + shifted-label validation.

Steps:
  1. Compute within-sequence feature-feature Spearman matrix (top 30)
  2. Hierarchical clustering (threshold |ρ| > 0.85) to merge redundant groups
  3. Keep the best representative per cluster (highest |ρ| vs fn_nano)
  4. Validate with shifted-label Spearman: predict mean(fn[t+30:t+60])
     from mean(feature[t:t+30]) — the actual forecasting task

Output:
  new_features/new_select_features_results.pdf
"""

import warnings
from pathlib import Path

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

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "new_all_features.csv"
SUMMARY_CSV = SCRIPT_DIR / "new_analysis_summary.csv"
OUT_PDF = SCRIPT_DIR / "new_select_features_results.pdf"

META_COLS = ["sequence", "frame_id", "weather", "fn_nano", "num_gt",
             "tp_nano", "tp_medium", "recoverable_fn"]
TARGET = "fn_nano"
WINDOW = 30     # 30 frames = 1 second at 30 fps
MIN_FRAMES = 60  # need input window + target window


# ═══════════════════════════════════════════════════════════════════
# 1. Within-Sequence Feature-Feature Correlation
# ═══════════════════════════════════════════════════════════════════

def within_seq_feature_corr(df, feature_cols):
    """Compute within-sequence Spearman |ρ| between all feature pairs.

    Uses vectorized spearmanr per sequence.
    Returns symmetric matrix of mean |ρ| across sequences.
    """
    n = len(feature_cols)
    corr_sum = np.zeros((n, n))
    n_valid = 0

    for seq_name, sdf in df.groupby("sequence"):
        if len(sdf) < MIN_FRAMES:
            continue

        data = sdf[feature_cols].values
        stds = data.std(axis=0)
        if (stds < 1e-10).sum() > n // 2:
            continue

        # scipy spearmanr on matrix returns (n×n) correlation matrix
        rho_matrix, _ = spearmanr(data)
        if rho_matrix.ndim == 0:
            continue
        rho_matrix = np.nan_to_num(rho_matrix, nan=0.0)
        corr_sum += np.abs(rho_matrix)
        n_valid += 1

    if n_valid > 0:
        corr_sum /= n_valid
    np.fill_diagonal(corr_sum, 1.0)
    return pd.DataFrame(corr_sum, index=feature_cols, columns=feature_cols)


# ═══════════════════════════════════════════════════════════════════
# 2. Shifted-Label Spearman Validation
# ═══════════════════════════════════════════════════════════════════

def shifted_label_spearman(df, feature_cols, target, window):
    """Shifted-label correlation: does mean(feature[t:t+w]) predict
    mean(target[t+w:t+2w])?

    Uses non-overlapping windows (stride=w) for unbiased estimates.
    """
    results = []

    for feat in feature_cols:
        rhos = []
        for seq_name, sdf in df.groupby("sequence"):
            sdf = sdf.sort_values("frame_id").reset_index(drop=True)
            n = len(sdf)
            total_need = 2 * window

            feat_means, target_means = [], []
            for t in range(0, n - total_need + 1, window):
                feat_means.append(sdf[feat].iloc[t : t + window].mean())
                target_means.append(sdf[target].iloc[t + window : t + total_need].mean())

            if len(feat_means) < 5:
                continue

            feat_arr = np.array(feat_means)
            target_arr = np.array(target_means)

            if feat_arr.std() < 1e-10 or target_arr.std() < 1e-10:
                continue

            rho, _ = spearmanr(feat_arr, target_arr)
            if not np.isnan(rho):
                rhos.append(rho)

        if rhos:
            rhos_arr = np.array(rhos)
            results.append({
                "feature": feat,
                "shifted_abs_rho": float(np.abs(rhos_arr).mean()),
                "shifted_signed_rho": float(rhos_arr.mean()),
                "shifted_std": float(rhos_arr.std()),
                "n_seqs": len(rhos),
            })
        else:
            results.append({
                "feature": feat, "shifted_abs_rho": 0.0,
                "shifted_signed_rho": 0.0, "shifted_std": 0.0, "n_seqs": 0,
            })

    return pd.DataFrame(results).sort_values("shifted_abs_rho", ascending=False)


# ═══════════════════════════════════════════════════════════════════
# 3. Hierarchical Clustering for Redundancy Pruning
# ═══════════════════════════════════════════════════════════════════

def select_features(corr_matrix, summary_df, threshold=0.85):
    """Cluster features by correlation; keep the best per cluster."""
    features = list(corr_matrix.index)
    n = len(features)

    # Distance = 1 - |correlation|
    dist_matrix = 1.0 - corr_matrix.values
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    dist_matrix = np.clip(dist_matrix, 0.0, 2.0)

    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="average")

    # Cut tree: merge clusters if distance < (1 - threshold)
    clusters = fcluster(Z, t=1.0 - threshold, criterion="distance")

    rho_map = summary_df.set_index("feature")["mean_abs_rho"].to_dict()

    selected = []
    cluster_info = {}
    for cid in sorted(set(clusters)):
        members = [features[i] for i in range(n) if clusters[i] == cid]
        members.sort(key=lambda f: rho_map.get(f, 0.0), reverse=True)
        best = members[0]
        selected.append(best)
        cluster_info[cid] = {
            "selected": best,
            "rho": rho_map.get(best, 0.0),
            "members": members,
            "dropped": members[1:],
        }

    # Sort selected by rho descending
    selected.sort(key=lambda f: rho_map.get(f, 0.0), reverse=True)
    return selected, clusters, Z, cluster_info


# ═══════════════════════════════════════════════════════════════════
# 4. PDF Report
# ═══════════════════════════════════════════════════════════════════

def generate_pdf(top_features, corr_matrix, Z, clusters, selected,
                 cluster_info, shifted_df, rho_map):
    with PdfPages(OUT_PDF) as pdf:

        # Page 1: Dendrogram
        fig, ax = plt.subplots(figsize=(16, 10))
        dendrogram(Z, labels=top_features, ax=ax,
                   leaf_rotation=90, leaf_font_size=7,
                   color_threshold=1.0 - 0.85)
        ax.axhline(y=1.0 - 0.85, color="red", linestyle="--", alpha=0.7,
                   label="Merge threshold (|ρ| > 0.85)")
        ax.set_ylabel("Distance (1 − |ρ|)")
        ax.set_title("Feature Redundancy Clustering\n"
                     "(Within-Sequence Spearman, Top 30 Features)")
        ax.legend(fontsize=9)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

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
                ax.get_xticklabels()[i].set_fontweight("bold")
                ax.get_xticklabels()[i].set_color("blue")
        plt.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title("Within-Seq Feature Correlation (Mean |ρ|)\n"
                     "Blue/Bold = Selected features")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 3: Shifted-label validation
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        ax = axes[0]
        sdf = shifted_df.sort_values("shifted_abs_rho", ascending=True)
        y_pos = range(len(sdf))
        ax.barh(y_pos, sdf["shifted_abs_rho"], color="#2196F3")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sdf["feature"], fontsize=8)
        ax.set_xlabel("Mean |ρ| (shifted-label, w=30)")
        ax.set_title("Shifted-Label Spearman\n"
                     "predict mean(fn[t+30:t+60]) from mean(feat[t:t+30])")

        ax = axes[1]
        feats = sdf["feature"].tolist()
        same_rho = [rho_map.get(f, 0) for f in feats]
        shift_rho = sdf["shifted_abs_rho"].values
        x = np.arange(len(feats))
        ax.barh(x - 0.2, same_rho, 0.35, label="Same-time |ρ|", color="#9E9E9E")
        ax.barh(x + 0.2, shift_rho, 0.35, label="Shifted |ρ|", color="#2196F3")
        ax.set_yticks(x)
        ax.set_yticklabels(feats, fontsize=8)
        ax.legend(fontsize=9)
        ax.set_xlabel("Mean |ρ|")
        ax.set_title("Same-Time vs Shifted-Label")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 4: Summary text
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis("off")
        lines = [
            "=" * 70,
            "FEATURE SELECTION SUMMARY",
            "=" * 70,
            f"",
            f"Input:     65 features, top 30 analyzed for redundancy",
            f"Method:    Hierarchical clustering (merge if within-seq |ρ| > 0.85)",
            f"Selected:  {len(selected)} non-redundant features",
            f"",
            f"Validation: Shifted-label Spearman (window = {WINDOW} frames = 1 sec)",
            f"  Task: predict mean(fn_nano[t+30:t+60]) from mean(feature[t:t+30])",
            f"",
            f"{'Feature':<35s} {'Same-time':>10s} {'Shifted':>10s} {'Dropped from cluster':<35s}",
            "-" * 90,
        ]
        for f in selected:
            same = rho_map.get(f, 0)
            row = shifted_df[shifted_df["feature"] == f]
            shift = float(row["shifted_abs_rho"].values[0]) if len(row) > 0 else 0.0
            # Find what was dropped
            for cid, info in cluster_info.items():
                if info["selected"] == f:
                    dropped = ", ".join(info["dropped"][:3]) or "(singleton)"
                    break
            else:
                dropped = ""
            lines.append(f"{f:<35s} {same:10.4f} {shift:10.4f} {dropped:<35s}")

        lines.extend([
            "",
            "=" * 70,
            "COPY-PASTE FOR TRAINING:",
            "=" * 70,
            "SELECTED_FEATURES = [",
        ])
        for f in selected:
            lines.append(f'    "{f}",')
        lines.append("]")

        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontfamily="monospace", fontsize=7, verticalalignment="top")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"  PDF: {OUT_PDF}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("FEATURE SELECTION: REDUNDANCY PRUNING + SHIFTED-LABEL VALIDATION")
    print("=" * 70)

    # Load data
    print("\nLoading data ...", flush=True)
    df = pd.read_csv(FEATURES_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    feature_cols = [c for c in df.columns if c not in META_COLS]

    # Fill NaN in temporal features (first frame of each sequence)
    for c in feature_cols:
        if "temporal" in c or c == "motion_pixel_ratio":
            df[c] = df[c].fillna(0.0)

    print(f"  {len(df):,} frames, {df['sequence'].nunique()} seqs, "
          f"{len(feature_cols)} features")

    # Step 1: Feature-feature correlation (top 30 by same-time |ρ|)
    print("\n[1/4] Within-seq feature-feature correlation (top 30) ...", flush=True)
    top30 = summary.head(30)["feature"].tolist()
    corr_matrix = within_seq_feature_corr(df, top30)

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
    print(f"  Total: {n_pairs} pairs")

    # Step 2: Cluster and select
    print("\n[2/4] Clustering ...", flush=True)
    selected, clusters, Z, cluster_info = select_features(
        corr_matrix, summary, threshold=0.85)

    print(f"\n  {len(selected)} features selected from {max(clusters)} clusters:")
    rho_map = summary.set_index("feature")["mean_abs_rho"].to_dict()
    for cid in sorted(cluster_info, key=lambda c: cluster_info[c]["rho"],
                       reverse=True):
        info = cluster_info[cid]
        dropped = ", ".join(info["dropped"][:3]) if info["dropped"] else "(sole member)"
        print(f"    {info['selected']:35s}  |ρ|={info['rho']:.4f}  "
              f"dropped: {dropped}")

    # Step 3: Shifted-label validation on selected features
    print("\n[3/4] Shifted-label Spearman (w=30) on selected features ...", flush=True)
    shifted_df = shifted_label_spearman(df, selected, TARGET, WINDOW)

    print(f"\n  {'Feature':<35s} {'Same-time |ρ|':>14s} {'Shifted |ρ|':>12s} {'Δ':>6s}")
    print(f"  {'-'*67}")
    for _, row in shifted_df.iterrows():
        feat = row["feature"]
        same = rho_map.get(feat, 0)
        shift = row["shifted_abs_rho"]
        delta = shift - same
        print(f"  {feat:<35s} {same:14.4f} {shift:12.4f} {delta:+6.3f}")

    # Step 4: PDF
    print("\n[4/4] Generating PDF ...", flush=True)
    generate_pdf(top30, corr_matrix, Z, clusters, selected,
                 cluster_info, shifted_df, rho_map)

    # Final output
    print(f"\n{'=' * 70}")
    print("SELECTED FEATURES = [")
    for f in selected:
        print(f'    "{f}",')
    print("]")
    print(f"{'=' * 70}")

    return selected


if __name__ == "__main__":
    main()
