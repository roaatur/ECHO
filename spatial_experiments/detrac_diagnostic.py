#!/usr/bin/env python3
"""DETRAC Diagnostic: recoverable_fn + Windowed Spearman Analysis.

Tests whether the target metric and prediction granularity are the issue
behind near-zero within-sequence Spearman correlations.

Three experiments on existing DETRAC data (no new YOLO runs or feature extraction):
  1. recoverable_fn distribution (compare to fn_nano)
  2. Per-frame Spearman: fn_nano vs recoverable_fn side-by-side
  3. Windowed Spearman: average over 10/30/60/90-frame windows
  4. High-GT filter: only frames with num_gt >= 5

Output:
  - Console: full results tables
  - spatial_experiments/detrac_diagnostic.pdf (4 pages)
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
ECHO_ROOT = Path(__file__).resolve().parent.parent
DET_CSV = ECHO_ROOT / "res_experiment" / "detrac_results.csv"
FEAT_CSV = ECHO_ROOT / "spatial_experiments" / "spatial_experiments_all_features.csv"
PDF_PATH = ECHO_ROOT / "spatial_experiments" / "detrac_diagnostic.pdf"

# ── Feature columns ───────────────────────────────────────────────
FEATURE_COLS = [
    "top_laplacian_variance", "top_small_component_count", "top_gradient_magnitude_std",
    "mid_laplacian_variance", "mid_small_component_count", "mid_gradient_magnitude_std",
    "bot_laplacian_variance", "bot_small_component_count", "bot_gradient_magnitude_std",
    "global_total_small_components", "global_overall_edge_density",
    "global_FAST_keypoint_count", "global_mean_brightness",
    "global_dark_channel_mean", "global_temporal_diff_mean",
]

TARGETS = ["fn_nano", "recoverable_fn", "miss_rate"]
WINDOW_SIZES = [10, 30, 60, 90]


# ── Helpers ────────────────────────────────────────────────────────

def within_seq_spearman(df, feature_cols, target, min_frames=30):
    """Compute within-sequence Spearman rho for each feature."""
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
            "within_seq_rho": np.mean(per_seq_rhos) if per_seq_rhos else 0.0,
            "n_valid_seqs": len(per_seq_rhos),
        }
    return results


def build_windowed_df(df, feature_cols, window_size):
    """Average targets and features over non-overlapping windows."""
    rows = []
    for seq, group in df.groupby("sequence"):
        group = group.sort_values("frame_id").reset_index(drop=True)
        n_windows = len(group) // window_size
        for w in range(n_windows):
            chunk = group.iloc[w * window_size:(w + 1) * window_size]
            row = {"sequence": seq, "window_idx": w}
            row["fn_nano"] = chunk["fn_nano"].mean()
            row["recoverable_fn"] = chunk["recoverable_fn"].mean()
            row["miss_rate"] = chunk["fn_nano"].mean() / max(chunk["num_gt"].mean(), 1)
            for feat in feature_cols:
                row[feat] = chunk[feat].mean()
            row["num_gt"] = chunk["num_gt"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────

def main():
    # ── Load and merge ─────────────────────────────────────────────
    print("Loading data...", flush=True)
    det_df = pd.read_csv(DET_CSV)
    feat_df = pd.read_csv(FEAT_CSV)

    # Pivot detection results
    nano = det_df[det_df["model"] == "nano_320"][["sequence", "frame_id", "tp", "fn"]].copy()
    nano = nano.rename(columns={"tp": "tp_nano", "fn": "fn_nano"})
    medium = det_df[det_df["model"] == "medium_640"][["sequence", "frame_id", "tp", "fn"]].copy()
    medium = medium.rename(columns={"tp": "tp_medium", "fn": "fn_medium"})
    det = nano.merge(medium, on=["sequence", "frame_id"])
    det["recoverable_fn"] = (det["tp_medium"] - det["tp_nano"]).clip(lower=0)

    # Merge with features
    df = feat_df.merge(
        det[["sequence", "frame_id", "tp_nano", "tp_medium", "fn_nano", "fn_medium", "recoverable_fn"]],
        on=["sequence", "frame_id"],
    )
    # Per-frame miss rate (fn / num_gt, clamped for num_gt=0)
    df["miss_rate"] = df["fn_nano"] / df["num_gt"].clip(lower=1)
    print(f"  {len(df):,} frames, {df['sequence'].nunique()} sequences\n", flush=True)

    # ── Step 1: recoverable_fn distribution ────────────────────────
    print("=" * 65)
    print("STEP 1: TARGET DISTRIBUTIONS")
    print("=" * 65)
    print(f"\n{'Statistic':<25s} {'fn_nano':>12s} {'recoverable_fn':>15s} {'miss_rate':>12s}")
    print("-" * 67)
    for stat_name, fn in [
        ("mean", lambda s: f"{s.mean():.3f}"),
        ("std", lambda s: f"{s.std():.3f}"),
        ("min", lambda s: f"{s.min():.3f}"),
        ("max", lambda s: f"{s.max():.3f}"),
        ("25th pctile", lambda s: f"{s.quantile(0.25):.3f}"),
        ("50th pctile", lambda s: f"{s.quantile(0.50):.3f}"),
        ("75th pctile", lambda s: f"{s.quantile(0.75):.3f}"),
        ("90th pctile", lambda s: f"{s.quantile(0.90):.3f}"),
        ("95th pctile", lambda s: f"{s.quantile(0.95):.3f}"),
        ("% zeros", lambda s: f"{100 * (s == 0).mean():.1f}%"),
    ]:
        fn_val = fn(df["fn_nano"])
        rfn_val = fn(df["recoverable_fn"])
        mr_val = fn(df["miss_rate"])
        print(f"{stat_name:<25s} {fn_val:>12s} {rfn_val:>15s} {mr_val:>12s}")

    # ── Step 2: Per-frame Spearman ─────────────────────────────────
    print(f"\n{'=' * 65}")
    print("STEP 2: PER-FRAME WITHIN-SEQUENCE SPEARMAN")
    print("=" * 65)

    step2_results = []
    for target in TARGETS:
        rhos = within_seq_spearman(df, FEATURE_COLS, target, min_frames=30)
        for feat, vals in rhos.items():
            step2_results.append({
                "condition": "per_frame",
                "target": target,
                "feature": feat,
                "within_seq_rho": round(vals["within_seq_rho"], 4),
                "n_valid_seqs": vals["n_valid_seqs"],
            })

    step2_df = pd.DataFrame(step2_results)

    # Pivot for side-by-side
    pivot = step2_df.pivot(index="feature", columns="target", values="within_seq_rho")
    pivot = pivot[["fn_nano", "recoverable_fn", "miss_rate"]]
    pivot = pivot.sort_values("fn_nano", ascending=False, key=abs)
    print(pivot.round(4).to_string())

    # ── Step 3: Windowed Spearman ──────────────────────────────────
    print(f"\n{'=' * 65}")
    print("STEP 3: WINDOWED SPEARMAN (CRITICAL TEST)")
    print("=" * 65)

    all_windowed_results = []
    for ws in WINDOW_SIZES:
        wdf = build_windowed_df(df, FEATURE_COLS, ws)
        n_seqs = wdf["sequence"].nunique()
        print(f"\n--- Window size = {ws} frames ({n_seqs} sequences, "
              f"{len(wdf)} windows) ---")

        for target in TARGETS:
            rhos = within_seq_spearman(wdf, FEATURE_COLS, target, min_frames=5)
            for feat, vals in rhos.items():
                all_windowed_results.append({
                    "condition": f"window_{ws}",
                    "target": target,
                    "feature": feat,
                    "within_seq_rho": round(vals["within_seq_rho"], 4),
                    "n_valid_seqs": vals["n_valid_seqs"],
                    "window_size": ws,
                })

        # Print window=30 summary
        if ws == 30:
            w30 = [r for r in all_windowed_results
                   if r["condition"] == "window_30"]
            w30_df = pd.DataFrame(w30)
            pivot_w30 = w30_df.pivot(index="feature", columns="target",
                                      values="within_seq_rho")
            pivot_w30 = pivot_w30[["fn_nano", "recoverable_fn", "miss_rate"]]
            pivot_w30 = pivot_w30.sort_values("fn_nano", ascending=False, key=abs)
            print(pivot_w30.round(4).to_string())

    windowed_df = pd.DataFrame(all_windowed_results)

    # Print rho vs window size for top features
    print(f"\n--- Rho vs Window Size (top features by per-frame |rho|) ---")
    top_feats = pivot.index[:5].tolist()
    for target in TARGETS:
        print(f"\n  Target: {target}")
        print(f"  {'Feature':<40s}", end="")
        for ws in WINDOW_SIZES:
            print(f"  w={ws:>3d}", end="")
        print()
        for feat in top_feats:
            print(f"  {feat:<40s}", end="")
            for ws in WINDOW_SIZES:
                mask = ((windowed_df["feature"] == feat) &
                        (windowed_df["target"] == target) &
                        (windowed_df["window_size"] == ws))
                val = windowed_df.loc[mask, "within_seq_rho"].values
                if len(val) > 0:
                    print(f"  {val[0]:>6.3f}", end="")
                else:
                    print(f"  {'N/A':>6s}", end="")
            print()

    # ── Step 4: High-GT filter ─────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("STEP 4: HIGH-GT FILTER")
    print("=" * 65)

    # Find threshold giving ~50% of frames
    gt_thresholds = [3, 5, 7, 10]
    for t in gt_thresholds:
        pct = (df["num_gt"] >= t).mean()
        print(f"  num_gt >= {t}: {100*pct:.1f}% of frames ({(df['num_gt'] >= t).sum():,})")

    # Use num_gt >= 5
    gt_thresh = 5
    df_dense = df[df["num_gt"] >= gt_thresh].copy().reset_index(drop=True)
    print(f"\nUsing num_gt >= {gt_thresh}: {len(df_dense):,} frames "
          f"({100*len(df_dense)/len(df):.1f}%)\n")

    step4_results = []
    for target in TARGETS:
        rhos = within_seq_spearman(df_dense, FEATURE_COLS, target, min_frames=30)
        for feat, vals in rhos.items():
            step4_results.append({
                "condition": f"high_gt_{gt_thresh}",
                "target": target,
                "feature": feat,
                "within_seq_rho": round(vals["within_seq_rho"], 4),
                "n_valid_seqs": vals["n_valid_seqs"],
            })

    step4_df = pd.DataFrame(step4_results)
    pivot_hgt = step4_df.pivot(index="feature", columns="target", values="within_seq_rho")
    pivot_hgt = pivot_hgt[["fn_nano", "recoverable_fn", "miss_rate"]]
    pivot_hgt = pivot_hgt.sort_values("fn_nano", ascending=False, key=abs)
    print(pivot_hgt.round(4).to_string())

    # ── Combined summary table ─────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("COMBINED SUMMARY (best rho per condition)")
    print("=" * 65)

    all_results = step2_results + all_windowed_results + step4_results
    all_df = pd.DataFrame(all_results)

    for cond in all_df["condition"].unique():
        for target in TARGETS:
            subset = all_df[(all_df["condition"] == cond) & (all_df["target"] == target)]
            if len(subset) == 0:
                continue
            best = subset.loc[subset["within_seq_rho"].abs().idxmax()]
            print(f"  {cond:<15s} {target:<18s} "
                  f"best: {best['feature']:<40s} rho={best['within_seq_rho']:+.4f}")

    # ── Generate PDF ───────────────────────────────────────────────
    print(f"\nGenerating PDF report...", flush=True)
    generate_pdf(df, df_dense, gt_thresh, pivot, pivot_w30, pivot_hgt,
                 windowed_df, top_feats, all_df)
    print(f"\nDone.")


# ── PDF generation ─────────────────────────────────────────────────

def generate_pdf(df, df_dense, gt_thresh, pivot_perframe, pivot_w30,
                 pivot_hgt, windowed_df, top_feats, all_df):

    with PdfPages(str(PDF_PATH)) as pdf:

        # ── Page 1: Distribution + Per-frame Spearman ─────────────
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # (0,0) fn_nano histogram
        ax = axes[0, 0]
        ax.hist(df["fn_nano"], bins=50, color="#e74c3c", alpha=0.7, edgecolor="white")
        ax.axvline(df["fn_nano"].mean(), color="k", ls="--", lw=1,
                   label=f"mean={df['fn_nano'].mean():.2f}")
        ax.set_xlabel("fn_nano")
        ax.set_ylabel("Count")
        ax.set_title(f"fn_nano (zero_pct={100*(df['fn_nano']==0).mean():.1f}%)")
        ax.legend(fontsize=9)

        # (0,1) recoverable_fn histogram
        ax = axes[0, 1]
        ax.hist(df["recoverable_fn"], bins=50, color="#3498db", alpha=0.7, edgecolor="white")
        ax.axvline(df["recoverable_fn"].mean(), color="k", ls="--", lw=1,
                   label=f"mean={df['recoverable_fn'].mean():.2f}")
        ax.set_xlabel("recoverable_fn = tp_medium - tp_nano")
        ax.set_ylabel("Count")
        ax.set_title(f"recoverable_fn (zero_pct={100*(df['recoverable_fn']==0).mean():.1f}%)")
        ax.legend(fontsize=9)

        # (1,0) Per-frame Spearman side-by-side bar chart (3 targets)
        ax = axes[1, 0]
        feats_sorted = pivot_perframe.sort_values("fn_nano", key=abs).index.tolist()
        y = np.arange(len(feats_sorted))
        bw = 0.25
        fn_vals = [pivot_perframe.loc[f, "fn_nano"] for f in feats_sorted]
        rfn_vals = [pivot_perframe.loc[f, "recoverable_fn"] for f in feats_sorted]
        mr_vals = [pivot_perframe.loc[f, "miss_rate"] for f in feats_sorted]
        ax.barh(y - bw, fn_vals, height=bw, color="#e74c3c", alpha=0.8, label="fn_nano")
        ax.barh(y, rfn_vals, height=bw, color="#3498db", alpha=0.8, label="recoverable_fn")
        ax.barh(y + bw, mr_vals, height=bw, color="#2ecc71", alpha=0.8, label="miss_rate")
        ax.set_yticks(y)
        ax.set_yticklabels([f.replace("_", "\n", 1) for f in feats_sorted], fontsize=6)
        ax.set_xlabel("Within-seq Spearman rho")
        ax.set_title("Per-Frame: 3 Targets Compared")
        ax.axvline(0.3, color="green", ls="--", lw=0.8)
        ax.axvline(-0.3, color="green", ls="--", lw=0.8)
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.legend(fontsize=7)

        # (1,1) Stats text
        ax = axes[1, 1]
        ax.axis("off")
        stats = (
            f"Data Summary\n{'='*35}\n\n"
            f"Frames:     {len(df):,}\n"
            f"Sequences:  {df['sequence'].nunique()}\n\n"
            f"fn_nano:         mean={df['fn_nano'].mean():.2f}  std={df['fn_nano'].std():.2f}\n"
            f"recoverable_fn:  mean={df['recoverable_fn'].mean():.2f}  std={df['recoverable_fn'].std():.2f}\n"
            f"miss_rate:       mean={df['miss_rate'].mean():.2f}  std={df['miss_rate'].std():.2f}\n\n"
            f"Best per-frame |rho|:\n"
            f"  fn_nano:        {pivot_perframe['fn_nano'].abs().max():.4f}\n"
            f"  recoverable_fn: {pivot_perframe['recoverable_fn'].abs().max():.4f}\n"
            f"  miss_rate:      {pivot_perframe['miss_rate'].abs().max():.4f}\n"
        )
        ax.text(0.05, 0.95, stats, transform=ax.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace")

        fig.suptitle("Page 1: Target Distributions + Per-Frame Spearman",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 2: Windowed Spearman ─────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # (0,0) and (0,1): rho vs window size for top features
        # Show fn_nano and miss_rate line plots (recoverable_fn in bar/text)
        for idx, target in enumerate(["fn_nano", "miss_rate"]):
            ax = axes[0, idx]
            for feat in top_feats:
                rhos = []
                for ws in WINDOW_SIZES:
                    mask = ((windowed_df["feature"] == feat) &
                            (windowed_df["target"] == target) &
                            (windowed_df["window_size"] == ws))
                    vals = windowed_df.loc[mask, "within_seq_rho"].values
                    rhos.append(vals[0] if len(vals) > 0 else 0)
                ax.plot(WINDOW_SIZES, rhos, "o-", label=feat.replace("global_", "g_"),
                        markersize=4)
            ax.axhline(0.3, color="green", ls="--", lw=0.8)
            ax.axhline(-0.3, color="green", ls="--", lw=0.8)
            ax.axhline(0, color="k", lw=0.3, ls=":")
            ax.set_xlabel("Window size (frames)")
            ax.set_ylabel("Within-seq rho")
            ax.set_title(f"rho vs Window Size ({target})")
            ax.legend(fontsize=6, loc="best")

        # (1,0) Windowed (w=30) side-by-side bar chart (3 targets)
        ax = axes[1, 0]
        feats_w30 = pivot_w30.sort_values("fn_nano", key=abs).index.tolist()
        y = np.arange(len(feats_w30))
        bw2 = 0.25
        fn_vals = [pivot_w30.loc[f, "fn_nano"] for f in feats_w30]
        rfn_vals = [pivot_w30.loc[f, "recoverable_fn"] for f in feats_w30]
        mr_vals = [pivot_w30.loc[f, "miss_rate"] for f in feats_w30]
        ax.barh(y - bw2, fn_vals, height=bw2, color="#e74c3c", alpha=0.8, label="fn_nano")
        ax.barh(y, rfn_vals, height=bw2, color="#3498db", alpha=0.8, label="recoverable_fn")
        ax.barh(y + bw2, mr_vals, height=bw2, color="#2ecc71", alpha=0.8, label="miss_rate")
        ax.set_yticks(y)
        ax.set_yticklabels([f.replace("_", "\n", 1) for f in feats_w30], fontsize=6)
        ax.set_xlabel("Within-seq Spearman rho")
        ax.set_title("Window=30: 3 Targets Compared")
        ax.axvline(0.3, color="green", ls="--", lw=0.8)
        ax.axvline(-0.3, color="green", ls="--", lw=0.8)
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.legend(fontsize=7)

        # (1,1) Window improvement text
        ax = axes[1, 1]
        ax.axis("off")
        txt = f"Windowed Spearman Summary\n{'='*35}\n\n"
        for target in TARGETS:
            txt += f"Target: {target}\n"
            for ws in WINDOW_SIZES:
                mask = ((windowed_df["target"] == target) &
                        (windowed_df["window_size"] == ws))
                sub = windowed_df[mask]
                if len(sub) > 0:
                    best_rho = sub["within_seq_rho"].abs().max()
                    best_feat = sub.loc[sub["within_seq_rho"].abs().idxmax(), "feature"]
                    txt += f"  w={ws:>3d}: best |rho|={best_rho:.4f} ({best_feat})\n"
            txt += "\n"
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", fontfamily="monospace")

        fig.suptitle("Page 2: Windowed Spearman (Granularity Test)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 3: High-GT filter ────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

        # Left: high-GT bar chart (3 targets)
        feats_hgt = pivot_hgt.sort_values("fn_nano", key=abs).index.tolist()
        y = np.arange(len(feats_hgt))
        bw3 = 0.25
        fn_vals = [pivot_hgt.loc[f, "fn_nano"] for f in feats_hgt]
        rfn_vals = [pivot_hgt.loc[f, "recoverable_fn"] for f in feats_hgt]
        mr_vals = [pivot_hgt.loc[f, "miss_rate"] for f in feats_hgt]
        ax1.barh(y - bw3, fn_vals, height=bw3, color="#e74c3c", alpha=0.8, label="fn_nano")
        ax1.barh(y, rfn_vals, height=bw3, color="#3498db", alpha=0.8, label="recoverable_fn")
        ax1.barh(y + bw3, mr_vals, height=bw3, color="#2ecc71", alpha=0.8, label="miss_rate")
        ax1.set_yticks(y)
        ax1.set_yticklabels([f.replace("_", "\n", 1) for f in feats_hgt], fontsize=6)
        ax1.set_xlabel("Within-seq Spearman rho")
        ax1.set_title(f"High-GT (num_gt >= {gt_thresh}): 3 Targets")
        ax1.axvline(0.3, color="green", ls="--", lw=0.8)
        ax1.axvline(-0.3, color="green", ls="--", lw=0.8)
        ax1.axvline(0, color="k", lw=0.5, ls="--")
        ax1.legend(fontsize=7)

        # Right: all-frames vs high-GT comparison
        # Compare per-frame rho (all) vs per-frame rho (high-GT) for fn_nano
        ax2.set_title("All Frames vs High-GT Filter (fn_nano)")
        all_rhos = [pivot_perframe.loc[f, "fn_nano"] if f in pivot_perframe.index else 0
                    for f in FEATURE_COLS]
        hgt_rhos = [pivot_hgt.loc[f, "fn_nano"] if f in pivot_hgt.index else 0
                    for f in FEATURE_COLS]
        for i, feat in enumerate(FEATURE_COLS):
            ax2.scatter(all_rhos[i], hgt_rhos[i], color="#3498db", s=50, zorder=3)
            ax2.annotate(feat.replace("global_", "g_").replace("_", "\n", 1),
                         (all_rhos[i], hgt_rhos[i]), fontsize=5,
                         ha="center", va="bottom")
        lim = max(max(abs(v) for v in all_rhos), max(abs(v) for v in hgt_rhos)) + 0.05
        ax2.plot([-lim, lim], [-lim, lim], "k--", lw=0.5, label="y=x")
        ax2.set_xlabel("Per-frame rho (all frames)")
        ax2.set_ylabel(f"Per-frame rho (num_gt >= {gt_thresh})")
        ax2.legend(fontsize=8)

        fig.suptitle("Page 3: High-GT Filter Effect",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 4: Summary table + recommendation ────────────────
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.axis("off")

        # Build summary
        summary = f"DETRAC Diagnostic Summary\n{'='*60}\n\n"

        # Best per condition
        summary += f"{'Condition':<18s} {'Target':<18s} {'Best Feature':<42s} {'rho':>8s}\n"
        summary += "-" * 86 + "\n"
        for cond in sorted(all_df["condition"].unique()):
            for target in TARGETS:
                subset = all_df[(all_df["condition"] == cond) & (all_df["target"] == target)]
                if len(subset) == 0:
                    continue
                best = subset.loc[subset["within_seq_rho"].abs().idxmax()]
                summary += (f"{cond:<18s} {target:<18s} "
                           f"{best['feature']:<42s} {best['within_seq_rho']:>+8.4f}\n")

        summary += f"\n\n{'='*60}\n"
        summary += "KEY FINDINGS:\n\n"

        # Auto-generate findings
        best_overall = all_df.loc[all_df["within_seq_rho"].abs().idxmax()]
        summary += (f"1. Best rho overall: {best_overall['within_seq_rho']:+.4f} "
                   f"({best_overall['feature']}, {best_overall['condition']}, "
                   f"{best_overall['target']})\n\n")

        # Compare targets
        fn_best = all_df[all_df["target"] == "fn_nano"]["within_seq_rho"].abs().max()
        rfn_best = all_df[all_df["target"] == "recoverable_fn"]["within_seq_rho"].abs().max()
        mr_best = all_df[all_df["target"] == "miss_rate"]["within_seq_rho"].abs().max()
        target_ranking = sorted([("fn_nano", fn_best), ("recoverable_fn", rfn_best),
                                  ("miss_rate", mr_best)], key=lambda x: x[1], reverse=True)
        summary += (f"2. Target ranking: {target_ranking[0][0]} |rho|={target_ranking[0][1]:.4f} > "
                   f"{target_ranking[1][0]} {target_ranking[1][1]:.4f} > "
                   f"{target_ranking[2][0]} {target_ranking[2][1]:.4f}\n\n")

        # Windowing effect
        pf_best = all_df[all_df["condition"] == "per_frame"]["within_seq_rho"].abs().max()
        w90_best = all_df[all_df["condition"] == "window_90"]["within_seq_rho"].abs().max() if len(all_df[all_df["condition"] == "window_90"]) > 0 else 0
        if w90_best > pf_best * 1.5:
            summary += f"3. Windowing HELPS: per-frame best={pf_best:.4f}, window_90 best={w90_best:.4f}\n\n"
        elif w90_best > pf_best:
            summary += f"3. Windowing helps slightly: per-frame best={pf_best:.4f}, window_90 best={w90_best:.4f}\n\n"
        else:
            summary += f"3. Windowing does NOT help: per-frame best={pf_best:.4f}, window_90 best={w90_best:.4f}\n\n"

        # Threshold check
        if abs(best_overall["within_seq_rho"]) >= 0.3:
            summary += "RECOMMENDATION: Signal found! Proceed with identified condition/target.\n"
        elif abs(best_overall["within_seq_rho"]) >= 0.2:
            summary += "RECOMMENDATION: Marginal signal. Consider combining features or alternative approaches.\n"
        else:
            summary += "RECOMMENDATION: Weak signal across all conditions. Spatial features alone may be insufficient.\n"

        ax.text(0.02, 0.98, summary, transform=ax.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace")

        fig.suptitle("Page 4: Summary & Recommendation",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"PDF saved -> {PDF_PATH}", flush=True)


if __name__ == "__main__":
    main()
