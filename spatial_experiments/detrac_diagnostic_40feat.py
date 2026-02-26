#!/usr/bin/env python3
"""Quick diagnostic: all 40 DETRAC candidate features.

Tests whether the 25 features dropped during selection carry signal
that the final 15 missed. Uses existing 14.5K-frame candidate data.

Runs: windowed Spearman (w=30,60,90) + Elastic Net on all 40 features.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────
ECHO_ROOT = Path(__file__).resolve().parent.parent
DET_CSV = ECHO_ROOT / "res_experiment" / "detrac_results.csv"
CAND_CSV = ECHO_ROOT / "spatial_experiments" / "spatial_experiments_candidate_features.csv"
PDF_PATH = ECHO_ROOT / "spatial_experiments" / "detrac_diagnostic_40feat.pdf"

# ── The 15 features that were kept ────────────────────────────────
KEPT_15 = {
    "top_laplacian_variance", "top_small_component_count", "top_gradient_magnitude_std",
    "mid_laplacian_variance", "mid_small_component_count", "mid_gradient_magnitude_std",
    "bot_laplacian_variance", "bot_small_component_count", "bot_gradient_magnitude_std",
    "global_total_small_components", "global_overall_edge_density",
    "global_FAST_keypoint_count", "global_mean_brightness",
    "global_dark_channel_mean", "global_temporal_diff_mean",
}

WINDOW_SIZES = [30, 60, 90]


def within_seq_spearman(df, feature_cols, target, min_frames=5):
    results = {}
    for feature in feature_cols:
        per_seq_rhos = []
        for _, group in df.groupby("sequence"):
            if len(group) < min_frames:
                continue
            if group[target].std() > 0 and group[feature].std() > 0:
                rho, _ = spearmanr(group[feature], group[target])
                if not np.isnan(rho):
                    per_seq_rhos.append(rho)
        results[feature] = {
            "rho": np.mean(per_seq_rhos) if per_seq_rhos else 0.0,
            "n": len(per_seq_rhos),
        }
    return results


def build_windowed_df(df, feature_cols, window_size):
    rows = []
    for seq, group in df.groupby("sequence"):
        group = group.sort_values("frame_id").reset_index(drop=True)
        n_windows = len(group) // window_size
        for w in range(n_windows):
            chunk = group.iloc[w * window_size:(w + 1) * window_size]
            row = {"sequence": seq, "window_idx": w}
            row["fn_nano"] = chunk["fn_nano"].mean()
            row["num_gt"] = chunk["num_gt"].mean()
            for feat in feature_cols:
                row[feat] = chunk[feat].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    # ── Load & merge ──────────────────────────────────────────────
    print("Loading data...", flush=True)
    det_df = pd.read_csv(DET_CSV)
    cand_df = pd.read_csv(CAND_CSV)

    # Get fn_nano from detection results
    nano = det_df[det_df["model"] == "nano_320"][["sequence", "frame_id", "tp", "fn"]].copy()
    nano = nano.rename(columns={"tp": "tp_nano", "fn": "fn_nano"})

    # cand_df has num_misses — merge with nano for fn_nano and num_gt
    medium = det_df[det_df["model"] == "medium_640"][["sequence", "frame_id", "fn"]].copy()
    medium = medium.rename(columns={"fn": "fn_medium"})
    det_info = nano.merge(medium, on=["sequence", "frame_id"])

    # Also need num_gt — get from det_df
    num_gt = det_df[det_df["model"] == "nano_320"][["sequence", "frame_id", "num_gt"]].copy()
    det_info = det_info.merge(num_gt, on=["sequence", "frame_id"])

    df = cand_df.merge(det_info, on=["sequence", "frame_id"])
    print(f"  {len(df):,} frames, {df['sequence'].nunique()} sequences", flush=True)

    # Feature columns = all 40
    meta_cols = {"sequence", "frame_id", "weather", "num_misses",
                 "tp_nano", "fn_nano", "fn_medium", "num_gt"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    dropped_25 = [f for f in feature_cols if f not in KEPT_15]

    print(f"  {len(feature_cols)} features total, {len(dropped_25)} were dropped\n", flush=True)

    # ── Per-frame Spearman (all 40) ──────────────────────────────
    print("=" * 65)
    print("PER-FRAME WITHIN-SEQUENCE SPEARMAN (all 40 features vs fn_nano)")
    print("=" * 65)

    pf_rhos = within_seq_spearman(df, feature_cols, "fn_nano", min_frames=30)
    pf_sorted = sorted(pf_rhos.items(), key=lambda x: abs(x[1]["rho"]), reverse=True)

    print(f"\n{'Feature':<42s} {'rho':>8s}  {'kept?':>5s}")
    print("-" * 60)
    for feat, vals in pf_sorted:
        kept = "YES" if feat in KEPT_15 else "no"
        print(f"{feat:<42s} {vals['rho']:>+8.4f}  {kept:>5s}")

    # ── Windowed Spearman ─────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("WINDOWED SPEARMAN (all 40 features vs fn_nano)")
    print("=" * 65)

    windowed_results = {}
    for ws in WINDOW_SIZES:
        wdf = build_windowed_df(df, feature_cols, ws)
        rhos = within_seq_spearman(wdf, feature_cols, "fn_nano", min_frames=5)
        windowed_results[ws] = rhos

        sorted_rhos = sorted(rhos.items(), key=lambda x: abs(x[1]["rho"]), reverse=True)
        print(f"\n--- Window = {ws} ({len(wdf)} windows) ---")
        print(f"{'Feature':<42s} {'rho':>8s}  {'kept?':>5s}")
        print("-" * 60)
        for feat, vals in sorted_rhos[:15]:  # top 15
            kept = "YES" if feat in KEPT_15 else "no"
            print(f"{feat:<42s} {vals['rho']:>+8.4f}  {kept:>5s}")

    # ── Elastic Net (windowed w=30) ──────────────────────────────
    print(f"\n{'=' * 65}")
    print("ELASTIC NET (window=30, within-seq normalized, all 40 features)")
    print("=" * 65)

    wdf30 = build_windowed_df(df, feature_cols, 30)

    # Per-sequence normalization
    for feat in feature_cols:
        wdf30[feat] = wdf30.groupby("sequence")[feat].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    wdf30["fn_nano"] = wdf30.groupby("sequence")["fn_nano"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    # Drop rows with NaN from normalization
    wdf30 = wdf30.dropna(subset=feature_cols + ["fn_nano"]).reset_index(drop=True)

    X = wdf30[feature_cols].values
    y = wdf30["fn_nano"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
                        cv=5, random_state=42, max_iter=5000)
    enet.fit(X_scaled, y)
    r2 = enet.score(X_scaled, y)

    print(f"\n  R² = {r2:.4f}  (l1_ratio={enet.l1_ratio_:.2f}, alpha={enet.alpha_:.6f})")

    # Feature importances
    coefs = pd.Series(enet.coef_, index=feature_cols)
    nonzero = coefs[coefs != 0].abs().sort_values(ascending=False)
    print(f"  Non-zero features: {len(nonzero)} / {len(feature_cols)}")
    print(f"\n  {'Feature':<42s} {'coef':>8s}  {'kept?':>5s}")
    print(f"  {'-'*58}")
    for feat, coef in nonzero.head(20).items():
        kept = "YES" if feat in KEPT_15 else "no"
        print(f"  {feat:<42s} {coefs[feat]:>+8.4f}  {kept:>5s}")

    # ── Compare kept vs dropped ──────────────────────────────────
    print(f"\n{'=' * 65}")
    print("KEPT 15 vs DROPPED 25: HEAD-TO-HEAD")
    print("=" * 65)

    for ws in WINDOW_SIZES:
        rhos = windowed_results[ws]
        kept_best = max((abs(rhos[f]["rho"]) for f in feature_cols if f in KEPT_15), default=0)
        dropped_best = max((abs(rhos[f]["rho"]) for f in feature_cols if f not in KEPT_15), default=0)
        kept_mean = np.mean([abs(rhos[f]["rho"]) for f in feature_cols if f in KEPT_15])
        dropped_mean = np.mean([abs(rhos[f]["rho"]) for f in feature_cols if f not in KEPT_15])
        print(f"  w={ws:>3d}:  kept best={kept_best:.4f}  dropped best={dropped_best:.4f}  "
              f"| kept mean={kept_mean:.4f}  dropped mean={dropped_mean:.4f}")

    # Also check: Elastic Net with ONLY dropped features
    print(f"\n  Elastic Net R² (all 40): {r2:.4f}")

    X_kept = wdf30[[f for f in feature_cols if f in KEPT_15]].values
    enet_kept = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
                             cv=5, random_state=42, max_iter=5000)
    enet_kept.fit(scaler.fit_transform(X_kept), y)
    print(f"  Elastic Net R² (kept 15 only): {enet_kept.score(scaler.transform(X_kept), y):.4f}")

    X_dropped = wdf30[[f for f in feature_cols if f not in KEPT_15]].values
    enet_drop = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
                             cv=5, random_state=42, max_iter=5000)
    enet_drop.fit(scaler.fit_transform(X_dropped), y)
    print(f"  Elastic Net R² (dropped 25 only): {enet_drop.score(scaler.transform(X_dropped), y):.4f}")

    # ── PDF ──────────────────────────────────────────────────────
    print(f"\nGenerating PDF...", flush=True)
    _generate_pdf(feature_cols, pf_rhos, windowed_results, coefs, r2)
    print(f"Done. -> {PDF_PATH}")


def _generate_pdf(feature_cols, pf_rhos, windowed_results, enet_coefs, enet_r2):
    with PdfPages(str(PDF_PATH)) as pdf:

        # ── Page 1: Per-frame + windowed rho, kept vs dropped ────
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))

        # Left: per-frame rho bar chart (all 40, colored by kept/dropped)
        ax = axes[0]
        sorted_feats = sorted(feature_cols, key=lambda f: abs(pf_rhos[f]["rho"]))
        y = np.arange(len(sorted_feats))
        colors = ["#e74c3c" if f in KEPT_15 else "#3498db" for f in sorted_feats]
        vals = [pf_rhos[f]["rho"] for f in sorted_feats]
        ax.barh(y, vals, color=colors, alpha=0.8, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([f.replace("global_", "g_") for f in sorted_feats], fontsize=5)
        ax.set_xlabel("Within-seq Spearman rho (per-frame)")
        ax.set_title("Per-Frame rho: All 40 Features\n(red=kept 15, blue=dropped 25)")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.axvline(0.3, color="green", ls="--", lw=0.8)
        ax.axvline(-0.3, color="green", ls="--", lw=0.8)

        # Right: windowed rho at w=90
        ax = axes[1]
        rhos_90 = windowed_results[90]
        sorted_feats_w = sorted(feature_cols, key=lambda f: abs(rhos_90[f]["rho"]))
        y = np.arange(len(sorted_feats_w))
        colors = ["#e74c3c" if f in KEPT_15 else "#3498db" for f in sorted_feats_w]
        vals = [rhos_90[f]["rho"] for f in sorted_feats_w]
        ax.barh(y, vals, color=colors, alpha=0.8, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([f.replace("global_", "g_") for f in sorted_feats_w], fontsize=5)
        ax.set_xlabel("Within-seq Spearman rho (window=90)")
        ax.set_title("Windowed rho (w=90): All 40 Features\n(red=kept 15, blue=dropped 25)")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.axvline(0.3, color="green", ls="--", lw=0.8)
        ax.axvline(-0.3, color="green", ls="--", lw=0.8)

        fig.suptitle("DETRAC 40-Feature Diagnostic: Kept vs Dropped",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 2: Elastic Net coefficients ─────────────────────
        fig, ax = plt.subplots(figsize=(14, 10))
        sorted_by_coef = sorted(feature_cols, key=lambda f: abs(enet_coefs[f]))
        y = np.arange(len(sorted_by_coef))
        colors = ["#e74c3c" if f in KEPT_15 else "#3498db" for f in sorted_by_coef]
        ax.barh(y, [enet_coefs[f] for f in sorted_by_coef],
                color=colors, alpha=0.8, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([f.replace("global_", "g_") for f in sorted_by_coef], fontsize=5)
        ax.set_xlabel("Elastic Net Coefficient")
        ax.set_title(f"Elastic Net (w=30, within-seq normalized): R²={enet_r2:.4f}\n"
                     f"(red=kept 15, blue=dropped 25)")
        ax.axvline(0, color="k", lw=0.5, ls="--")

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
