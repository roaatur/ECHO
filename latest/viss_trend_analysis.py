#!/usr/bin/env python3
"""Trend analysis: do feature SLOPES predict future difficulty on VisDrone?

For each feature, compute a per-frame slope (linear regression over a short
lookback window). Then measure Spearman rho of that slope vs future fn_nano,
and compare it to the rho of the raw feature value.

This tells us which features' rate-of-change carries predictive signal for
intra-sequence difficulty transitions.

Output:
  - viss_trend_analysis_results.pdf
  - viss_trend_analysis_results.csv
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

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ECHO_DIR = SCRIPT_DIR.parent
EVAL_CSV = ECHO_DIR / "yolo_runs_on_uavdt" / "visdrone_classagnostic_results.csv"
DETRAC_FEAT_CSV = SCRIPT_DIR / "viss_detrac65_features.csv"
JAYS_FEAT_CSV = ECHO_DIR / "yolo_runs_on_uavdt" / "jays" / "jays_visdrone_features.csv"
OUT_PDF = SCRIPT_DIR / "viss_trend_analysis_results.pdf"
OUT_CSV = SCRIPT_DIR / "viss_trend_analysis_results.csv"

JAYS_DROP_COLS = [
    "GLCM_contrast", "GLCM_homogeneity", "GLCM_energy", "GLCM_entropy",
    "GLCM_correlation", "LBP_entropy", "LBP_uniformity", "wavelet_detail_energy",
]

# 23 val sequences (same as viss_full_evaluation.py)
VAL_SEQS = [
    "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
    "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v",
    "uav0000009_03358_v", "uav0000073_00600_v", "uav0000073_04464_v",
    "uav0000077_00720_v", "uav0000088_00290_v", "uav0000119_02301_v",
    "uav0000120_04775_v", "uav0000161_00000_v", "uav0000188_00000_v",
    "uav0000201_00000_v", "uav0000249_00001_v", "uav0000249_02688_v",
    "uav0000297_00000_v", "uav0000297_02761_v", "uav0000306_00230_v",
    "uav0000355_00001_v", "uav0000370_00001_v",
]

WINDOW = 15
HORIZON = 15
SLOPE_LOOKBACKS = [5, 10, 15]  # frames to compute slope over


def load_data():
    """Load eval results + both feature sets, merge."""
    print("Loading data ...", flush=True)
    ev = pd.read_csv(EVAL_CSV)
    nano = ev[ev["model"] == "yolo11n.pt"][
        ["sequence", "frame_id", "tp", "fp", "fn", "num_gt"]].copy()
    nano.columns = ["sequence", "frame_id", "tp_nano", "fp_nano", "fn_nano", "num_gt"]
    med = ev[ev["model"] == "yolo11m.pt"][["sequence", "frame_id", "tp"]].copy()
    med.columns = ["sequence", "frame_id", "tp_medium"]
    eval_merged = nano.merge(med, on=["sequence", "frame_id"])

    # DETRAC-65 features
    detrac = pd.read_csv(DETRAC_FEAT_CSV).fillna(0.0)
    detrac_feat_cols = [c for c in detrac.columns if c not in ("sequence", "frame_id")]
    detrac = detrac.rename(columns={c: f"D_{c}" for c in detrac_feat_cols})

    # Jay's features
    jays = pd.read_csv(JAYS_FEAT_CSV)
    jays = jays.drop(columns=[c for c in JAYS_DROP_COLS if c in jays.columns], errors="ignore")
    jays = jays.fillna(0.0)
    jays_feat_cols = [c for c in jays.columns if c not in ("sequence", "frame_id")]
    jays = jays.rename(columns={c: f"J_{c}" for c in jays_feat_cols})

    df = eval_merged.merge(detrac, on=["sequence", "frame_id"], how="inner")
    df = df.merge(jays, on=["sequence", "frame_id"], how="inner")

    all_cols = [f"D_{c}" for c in detrac_feat_cols] + [f"J_{c}" for c in jays_feat_cols]
    print(f"  {len(df)} frames, {df['sequence'].nunique()} sequences, "
          f"{len(all_cols)} features", flush=True)
    return df, all_cols


def compute_slopes(values, lookback):
    """Compute per-frame slopes via linear regression over lookback window.

    Returns array same length as values, with NaN for first (lookback-1) frames.
    """
    n = len(values)
    slopes = np.full(n, np.nan)
    # Precompute time vector centered at 0
    t = np.arange(lookback, dtype=np.float64) - (lookback - 1) / 2.0
    t_var = (t ** 2).sum()
    if t_var == 0:
        return slopes
    for i in range(lookback - 1, n):
        window = values[i - lookback + 1: i + 1].astype(np.float64)
        slopes[i] = np.dot(window - window.mean(), t) / t_var
    return slopes


def analyze_trends(df, all_cols):
    """For each feature and lookback, compute slope and measure rho vs future fn_nano."""
    # Use train sequences only for correlation analysis
    train_df = df[~df["sequence"].isin(VAL_SEQS)]

    results = []

    for feat in all_cols:
        src = "DETRAC" if feat.startswith("D_") else "Jays"
        feat_name = feat[2:]  # strip D_ or J_ prefix

        # Collect raw and slope values paired with future fn_nano targets
        raw_vals_all = []
        slope_vals_all = {lb: [] for lb in SLOPE_LOOKBACKS}
        target_all = []

        for seq, sdf in train_df.groupby("sequence"):
            sdf = sdf.sort_values("frame_id").reset_index(drop=True)
            n = len(sdf)
            if n < WINDOW + HORIZON:
                continue

            feat_values = sdf[feat].values.astype(np.float64)
            fn_values = sdf["fn_nano"].values.astype(np.float64)

            # Precompute slopes for each lookback
            slopes = {}
            for lb in SLOPE_LOOKBACKS:
                slopes[lb] = compute_slopes(feat_values, lb)

            # Build analysis windows
            for t in range(WINDOW - 1, n - HORIZON):
                future_fn = fn_values[t + 1: t + 1 + HORIZON].mean()
                raw_vals_all.append(feat_values[t])
                target_all.append(future_fn)
                for lb in SLOPE_LOOKBACKS:
                    if not np.isnan(slopes[lb][t]):
                        slope_vals_all[lb].append(slopes[lb][t])
                    else:
                        slope_vals_all[lb].append(np.nan)

        raw_arr = np.array(raw_vals_all)
        target_arr = np.array(target_all)

        # Raw rho
        valid = ~np.isnan(raw_arr) & ~np.isnan(target_arr)
        if valid.sum() < 30:
            continue
        rho_raw, p_raw = spearmanr(raw_arr[valid], target_arr[valid])

        row = {
            "feature": feat_name,
            "source": src,
            "rho_raw": rho_raw,
            "abs_rho_raw": abs(rho_raw),
            "p_raw": p_raw,
        }

        # Slope rhos for each lookback
        for lb in SLOPE_LOOKBACKS:
            s_arr = np.array(slope_vals_all[lb])
            valid_s = ~np.isnan(s_arr) & ~np.isnan(target_arr)
            if valid_s.sum() < 30:
                row[f"rho_slope_{lb}"] = np.nan
                row[f"abs_rho_slope_{lb}"] = np.nan
                row[f"p_slope_{lb}"] = np.nan
                row[f"gain_{lb}"] = np.nan
            else:
                rho_s, p_s = spearmanr(s_arr[valid_s], target_arr[valid_s])
                row[f"rho_slope_{lb}"] = rho_s
                row[f"abs_rho_slope_{lb}"] = abs(rho_s)
                row[f"p_slope_{lb}"] = p_s
                row[f"gain_{lb}"] = abs(rho_s) - abs(rho_raw)

        # Within-sequence variance ratio
        seq_means = train_df.groupby("sequence")[feat].mean()
        between_var = seq_means.var()
        total_var = train_df[feat].var()
        row["within_seq_var_ratio"] = float(
            (total_var - between_var) / (total_var + 1e-10))

        # Best slope lookback
        slope_rhos = [abs(row.get(f"rho_slope_{lb}", 0) or 0) for lb in SLOPE_LOOKBACKS]
        best_lb_idx = int(np.argmax(slope_rhos))
        row["best_slope_lookback"] = SLOPE_LOOKBACKS[best_lb_idx]
        row["best_abs_rho_slope"] = slope_rhos[best_lb_idx]
        row["best_gain"] = row.get(f"gain_{SLOPE_LOOKBACKS[best_lb_idx]}", 0) or 0

        results.append(row)

    return pd.DataFrame(results)


def main():
    df, all_cols = load_data()

    print(f"\nAnalyzing trends for {len(all_cols)} features "
          f"with lookbacks {SLOPE_LOOKBACKS} ...", flush=True)
    results = analyze_trends(df, all_cols)

    # Sort by best slope |rho|
    results = results.sort_values("best_abs_rho_slope", ascending=False).reset_index(drop=True)

    # Save CSV
    results.to_csv(OUT_CSV, index=False)
    print(f"\nSaved CSV: {OUT_CSV}", flush=True)

    # Print top results
    print(f"\n{'='*80}", flush=True)
    print("TOP 30 FEATURES BY SLOPE |ρ| (best lookback)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Feature':40s} {'Src':6s} {'ρ_raw':>8s} {'ρ_slope':>8s} "
          f"{'Gain':>8s} {'LB':>4s} {'WS_var':>7s}", flush=True)
    print("-" * 80, flush=True)
    for _, r in results.head(30).iterrows():
        print(f"{r['feature']:40s} {r['source']:6s} {r['rho_raw']:+.4f} "
              f"{r['best_abs_rho_slope']:+.4f} {r['best_gain']:+.4f} "
              f"{r['best_slope_lookback']:4.0f} {r['within_seq_var_ratio']:.3f}",
              flush=True)

    print(f"\n{'='*80}", flush=True)
    print("TOP 20 FEATURES WHERE SLOPE BEATS RAW (biggest gain)", flush=True)
    print(f"{'='*80}", flush=True)
    gains = results.sort_values("best_gain", ascending=False).head(20)
    for _, r in gains.iterrows():
        print(f"{r['feature']:40s} {r['source']:6s} raw={r['rho_raw']:+.4f} "
              f"slope={r['best_abs_rho_slope']:+.4f} gain={r['best_gain']:+.4f} "
              f"lb={r['best_slope_lookback']:.0f} ws_var={r['within_seq_var_ratio']:.3f}",
              flush=True)

    # PDF
    print("\nGenerating PDF ...", flush=True)
    with PdfPages(str(OUT_PDF)) as pdf:

        # Page 1: Raw rho vs Slope rho scatter for all features
        fig, ax = plt.subplots(figsize=(10, 10))
        detrac_mask = results["source"] == "DETRAC"
        jays_mask = results["source"] == "Jays"
        ax.scatter(results.loc[detrac_mask, "abs_rho_raw"],
                   results.loc[detrac_mask, "best_abs_rho_slope"],
                   c="#4C72B0", alpha=0.6, s=40, label="DETRAC", edgecolors="white", lw=0.5)
        ax.scatter(results.loc[jays_mask, "abs_rho_raw"],
                   results.loc[jays_mask, "best_abs_rho_slope"],
                   c="#55A868", alpha=0.6, s=40, label="Jay's", edgecolors="white", lw=0.5)
        # Label top gainers
        top_gain = results.nlargest(10, "best_gain")
        for _, r in top_gain.iterrows():
            ax.annotate(r["feature"], (r["abs_rho_raw"], r["best_abs_rho_slope"]),
                        fontsize=6, alpha=0.8)
        # Diagonal line (slope = raw)
        lim = max(results["abs_rho_raw"].max(), results["best_abs_rho_slope"].max()) + 0.02
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5, label="slope = raw")
        ax.set_xlabel("|ρ| Raw Feature Value", fontsize=11)
        ax.set_ylabel("|ρ| Feature Slope (best lookback)", fontsize=11)
        ax.set_title("VisDrone Trend Analysis: Raw vs Slope Correlation with Future fn_nano\n"
                      "(Points above diagonal = slope beats raw)")
        ax.legend()
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 2: Top 30 features by slope |rho|, showing raw vs slope side by side
        top30 = results.head(30)
        fig, ax = plt.subplots(figsize=(12, 10))
        y_pos = np.arange(len(top30))
        bars1 = ax.barh(y_pos - 0.2, top30["abs_rho_raw"].values, 0.4,
                         label="|ρ| raw", color="#AAAAAA", alpha=0.7)
        colors_slope = ["#4C72B0" if s == "DETRAC" else "#55A868"
                        for s in top30["source"].values]
        bars2 = ax.barh(y_pos + 0.2, top30["best_abs_rho_slope"].values, 0.4,
                         label="|ρ| slope", color=colors_slope)
        ax.set_yticks(y_pos)
        labels = [f"{r['feature']}  [{r['source'][0]}] (lb={r['best_slope_lookback']:.0f})"
                  for _, r in top30.iterrows()]
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("|Spearman ρ|")
        ax.set_title("Top 30 Features by Slope |ρ| — Raw (gray) vs Slope (colored)\n"
                      "(Blue=DETRAC, Green=Jay's)")
        ax.legend(fontsize=9)
        ax.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 3: Top 20 features by GAIN (slope - raw), only positive gains
        pos_gains = results[results["best_gain"] > 0].nlargest(20, "best_gain")
        if len(pos_gains) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            y_pos = np.arange(len(pos_gains))
            colors_g = ["#4C72B0" if s == "DETRAC" else "#55A868"
                        for s in pos_gains["source"].values]
            ax.barh(y_pos, pos_gains["best_gain"].values, color=colors_g)
            ax.set_yticks(y_pos)
            labels_g = [f"{r['feature']}  [{r['source'][0]}] "
                        f"(raw={r['abs_rho_raw']:.3f} → slope={r['best_abs_rho_slope']:.3f})"
                        for _, r in pos_gains.iterrows()]
            ax.set_yticklabels(labels_g, fontsize=6)
            ax.set_xlabel("Gain: |ρ_slope| − |ρ_raw|")
            ax.set_title("Features Where Slope Beats Raw Value\n"
                          "(These are candidates for trend features)")
            ax.invert_yaxis()
            plt.tight_layout()
            pdf.savefig(fig); plt.close()

        # Page 4: Within-sequence variance vs slope |rho|
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(results.loc[detrac_mask, "within_seq_var_ratio"],
                   results.loc[detrac_mask, "best_abs_rho_slope"],
                   c="#4C72B0", alpha=0.6, s=40, label="DETRAC", edgecolors="white", lw=0.5)
        ax.scatter(results.loc[jays_mask, "within_seq_var_ratio"],
                   results.loc[jays_mask, "best_abs_rho_slope"],
                   c="#55A868", alpha=0.6, s=40, label="Jay's", edgecolors="white", lw=0.5)
        # Label interesting points (high on both axes)
        interesting = results[
            (results["within_seq_var_ratio"] > 0.3) & (results["best_abs_rho_slope"] > 0.1)]
        for _, r in interesting.iterrows():
            ax.annotate(r["feature"], (r["within_seq_var_ratio"], r["best_abs_rho_slope"]),
                        fontsize=6, alpha=0.8)
        ax.set_xlabel("Within-Sequence Variance Ratio", fontsize=11)
        ax.set_ylabel("|ρ| Feature Slope vs Future fn_nano", fontsize=11)
        ax.set_title("Within-Seq Variance vs Slope Predictiveness\n"
                      "(Top-right = high variance AND predictive slope)")
        ax.axvline(0.5, color="red", ls="--", lw=0.8, alpha=0.5, label="50% var ratio")
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 5: Lookback comparison for top features
        top15 = results.head(15)
        fig, ax = plt.subplots(figsize=(12, 7))
        y_pos = np.arange(len(top15))
        width = 0.25
        for i, lb in enumerate(SLOPE_LOOKBACKS):
            col_name = f"abs_rho_slope_{lb}"
            vals = top15[col_name].fillna(0).values
            ax.barh(y_pos + (i - 1) * width, vals, width,
                    label=f"Lookback {lb}f", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{r['feature']} [{r['source'][0]}]"
                            for _, r in top15.iterrows()], fontsize=7)
        ax.set_xlabel("|ρ| of slope")
        ax.set_title("Slope |ρ| by Lookback Window (top 15 features)")
        ax.legend()
        ax.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 6: Summary table
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis("off")
        top20 = results.head(20)
        headers = ["Feature", "Source", "ρ Raw", "ρ Slope", "Gain",
                    "Best LB", "WS Var"]
        rows = []
        for _, r in top20.iterrows():
            rows.append([
                r["feature"],
                r["source"],
                f"{r['rho_raw']:+.4f}",
                f"{r['best_abs_rho_slope']:.4f}",
                f"{r['best_gain']:+.4f}",
                f"{r['best_slope_lookback']:.0f}",
                f"{r['within_seq_var_ratio']:.3f}",
            ])
        table = ax.table(cellText=rows, colLabels=headers, loc="center",
                         cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.4)
        for j in range(len(headers)):
            table[0, j].set_facecolor("#4C72B0")
            table[0, j].set_text_props(color="white", fontweight="bold")
        # Highlight positive gains
        for i, (_, r) in enumerate(top20.iterrows()):
            if r["best_gain"] > 0:
                for j in range(len(headers)):
                    table[i + 1, j].set_facecolor("#E6FFE6")
        ax.set_title("VisDrone Trend Analysis — Top 20 Features by Slope |ρ|\n"
                      "(Green rows = slope beats raw)",
                      fontsize=12, fontweight="bold", pad=20)
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

    print(f"Saved PDF: {OUT_PDF}", flush=True)

    # Summary stats
    n_slope_wins = (results["best_gain"] > 0).sum()
    n_total = len(results)
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total features analyzed: {n_total}", flush=True)
    print(f"Features where slope beats raw: {n_slope_wins}/{n_total} "
          f"({n_slope_wins/n_total:.0%})", flush=True)
    print(f"Best slope feature: {results.iloc[0]['feature']} "
          f"(|ρ|={results.iloc[0]['best_abs_rho_slope']:.4f})", flush=True)
    print(f"Biggest gain: {results.nlargest(1, 'best_gain').iloc[0]['feature']} "
          f"(gain={results.nlargest(1, 'best_gain').iloc[0]['best_gain']:+.4f})", flush=True)
    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
