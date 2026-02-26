#!/usr/bin/env python3
"""Diagnose WHY the LSTM can't predict frame-level fn despite features having
global rho=0.456 with fn.  Three tests:

1. Within-sequence correlation — does feature signal survive within sequences?
2. Simple per-frame models — is the bottleneck features or the LSTM?
3. Detection features — does adding nano@320 output (tp, fp, precision, mean_iou) help?

Output: spatial_experiments_diagnose.pdf (4 pages)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ── Paths & constants ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

FEAT_CSV   = SCRIPT_DIR / "spatial_experiments_all_features.csv"
DETRAC_CSV = ROOT / "res_experiment" / "detrac_results.csv"
PDF_PATH   = SCRIPT_DIR / "spatial_experiments_diagnose.pdf"

EXCLUDE_SEQUENCES = ["MVI_39211"]

VAL_SEQUENCES = [
    "MVI_39361", "MVI_20011", "MVI_41073",                  # sunny
    "MVI_40172", "MVI_40191", "MVI_40712", "MVI_40244",     # cloudy
    "MVI_40992", "MVI_40732", "MVI_39811", "MVI_40775",     # night
    "MVI_63521", "MVI_40892", "MVI_63553",                   # rainy
]

SPATIAL_FEATURES = [
    "top_laplacian_variance", "top_small_component_count", "top_gradient_magnitude_std",
    "mid_laplacian_variance", "mid_small_component_count", "mid_gradient_magnitude_std",
    "bot_laplacian_variance", "bot_small_component_count", "bot_gradient_magnitude_std",
    "global_total_small_components", "global_overall_edge_density",
    "global_FAST_keypoint_count", "global_mean_brightness",
    "global_dark_channel_mean", "global_temporal_diff_mean",
]

# Nano@320 detection features available at deployment (nano is already running)
NANO_FEATURES = ["tp", "fp", "precision", "mean_iou"]
# Oracle feature (not available at deployment)
ORACLE_FEATURES = ["num_gt"]

TARGET = "fn"


# ── Data loading ───────────────────────────────────────────────────

def load_data():
    """Load spatial features + nano@320 detection outputs, merge, filter."""
    print("Loading spatial features ...", flush=True)
    feat_df = pd.read_csv(FEAT_CSV)
    feat_df = feat_df[~feat_df["sequence"].isin(EXCLUDE_SEQUENCES)]
    feat_df = feat_df[feat_df["num_gt"] >= 1].reset_index(drop=True)
    print(f"  Spatial: {len(feat_df)} frames, {feat_df['sequence'].nunique()} sequences")

    print("Loading detrac_results.csv (nano@320) ...", flush=True)
    det_df = pd.read_csv(DETRAC_CSV)
    det_df = det_df[det_df["model"] == "nano_320"].copy()
    det_df = det_df[["sequence", "frame_id"] + NANO_FEATURES].copy()
    print(f"  Detrac nano: {len(det_df)} frames")

    # Merge — inner join on (sequence, frame_id)
    df = feat_df.merge(det_df, on=["sequence", "frame_id"], how="inner")
    print(f"  Merged: {len(df)} frames, {df['sequence'].nunique()} sequences")

    return df


# ── Test 1: Within-sequence correlation ────────────────────────────

def test1_within_seq_correlation(df):
    """Compare global vs within-sequence Spearman rho for each feature."""
    print("\n" + "=" * 70)
    print("TEST 1: WITHIN-SEQUENCE CORRELATION")
    print("=" * 70)

    all_features = SPATIAL_FEATURES + NANO_FEATURES

    # Global rho (pooled)
    global_rho = {}
    for feat in all_features:
        rho, _ = spearmanr(df[feat], df[TARGET])
        global_rho[feat] = rho

    # Within-sequence rho (per sequence, then average)
    within_rhos = {feat: [] for feat in all_features}
    for seq_name, sdf in df.groupby("sequence"):
        if len(sdf) < 10:
            continue
        for feat in all_features:
            if sdf[feat].std() == 0 or sdf[TARGET].std() == 0:
                continue
            rho, _ = spearmanr(sdf[feat], sdf[TARGET])
            if not np.isnan(rho):
                within_rhos[feat].append(rho)

    within_rho_mean = {}
    for feat in all_features:
        vals = within_rhos[feat]
        within_rho_mean[feat] = np.mean(vals) if vals else 0.0

    # Print table
    print(f"\n  {'Feature':<40s} {'Global rho':>10s} {'Within rho':>10s} {'Drop':>8s}")
    print("  " + "-" * 68)
    for feat in sorted(all_features, key=lambda f: abs(global_rho[f]), reverse=True):
        g = global_rho[feat]
        w = within_rho_mean[feat]
        drop = g - w
        print(f"  {feat:<40s} {g:10.4f} {w:10.4f} {drop:8.4f}")

    return global_rho, within_rho_mean


# ── Test 2: Simple per-frame models (spatial only) ─────────────────

def _per_seq_r2(val_df, y_pred):
    """Compute per-sequence R2, return dict {seq: r2}."""
    val_df = val_df.copy()
    val_df["_pred"] = y_pred
    per_seq = {}
    for seq_name, sdf in val_df.groupby("sequence"):
        if sdf[TARGET].std() == 0:
            per_seq[seq_name] = 0.0
            continue
        per_seq[seq_name] = r2_score(sdf[TARGET], sdf["_pred"])
    return per_seq


def test2_simple_models(df):
    """Train simple per-frame models on spatial features only."""
    print("\n" + "=" * 70)
    print("TEST 2: SIMPLE PER-FRAME MODELS (spatial features only)")
    print("=" * 70)

    train_df = df[~df["sequence"].isin(VAL_SEQUENCES)].copy()
    val_df = df[df["sequence"].isin(VAL_SEQUENCES)].copy()
    print(f"  Train: {len(train_df)} frames, Val: {len(val_df)} frames")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[SPATIAL_FEATURES])
    X_val = scaler.transform(val_df[SPATIAL_FEATURES])
    y_train = train_df[TARGET].values
    y_val = val_df[TARGET].values

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RF(100,d10)": RandomForestRegressor(n_estimators=100, max_depth=10,
                                              random_state=42, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        print(f"\n  Training {name} ...", flush=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        overall_r2 = r2_score(y_val, y_pred)
        per_seq = _per_seq_r2(val_df, y_pred)
        mean_per_seq_r2 = np.mean(list(per_seq.values()))
        results[name] = {"overall_r2": overall_r2, "per_seq": per_seq,
                         "mean_per_seq_r2": mean_per_seq_r2}
        print(f"    Overall R2={overall_r2:.4f}  Mean per-seq R2={mean_per_seq_r2:.4f}")

        for seq in VAL_SEQUENCES:
            if seq in per_seq:
                print(f"      {seq}: R2={per_seq[seq]:.4f}")

    return results


# ── Test 3: Detection features impact ──────────────────────────────

def test3_detection_features(df):
    """Test adding nano@320 detection features to spatial features."""
    print("\n" + "=" * 70)
    print("TEST 3: DETECTION FEATURES IMPACT")
    print("=" * 70)

    train_df = df[~df["sequence"].isin(VAL_SEQUENCES)].copy()
    val_df = df[df["sequence"].isin(VAL_SEQUENCES)].copy()

    feature_sets = {
        "Spatial only (15)": SPATIAL_FEATURES,
        "Spatial + nano (19)": SPATIAL_FEATURES + NANO_FEATURES,
        "Spatial + nano + GT (20)": SPATIAL_FEATURES + NANO_FEATURES + ORACLE_FEATURES,
    }

    # Use RF as the best model from Test 2 (most flexible)
    results = {}
    for fs_name, features in feature_sets.items():
        print(f"\n  Feature set: {fs_name} ({len(features)} features)")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features])
        X_val = scaler.transform(val_df[features])
        y_train = train_df[TARGET].values

        model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                       random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        overall_r2 = r2_score(val_df[TARGET].values, y_pred)
        per_seq = _per_seq_r2(val_df, y_pred)
        mean_per_seq_r2 = np.mean(list(per_seq.values()))
        results[fs_name] = {"overall_r2": overall_r2, "per_seq": per_seq,
                            "mean_per_seq_r2": mean_per_seq_r2}
        print(f"    Overall R2={overall_r2:.4f}  Mean per-seq R2={mean_per_seq_r2:.4f}")

        for seq in VAL_SEQUENCES:
            if seq in per_seq:
                print(f"      {seq}: R2={per_seq[seq]:.4f}")

    return results


# ── PDF ────────────────────────────────────────────────────────────

def generate_pdf(global_rho, within_rho_mean, test2_results, test3_results,
                 seq_weather):
    """Generate 4-page diagnostic PDF."""

    with PdfPages(str(PDF_PATH)) as pdf:

        # ── Page 1: Within-sequence correlation ──
        all_features = SPATIAL_FEATURES + NANO_FEATURES
        sorted_feats = sorted(all_features, key=lambda f: within_rho_mean[f])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

        # Left: horizontal bar chart of within-seq rho
        y_pos = np.arange(len(sorted_feats))
        bars = ax1.barh(y_pos, [within_rho_mean[f] for f in sorted_feats],
                        color="steelblue", alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sorted_feats, fontsize=7)
        ax1.set_xlabel("Mean within-sequence Spearman rho")
        ax1.set_title("Within-Sequence Correlation with fn")
        ax1.axvline(0, color="k", lw=0.5, ls="--")

        # Color bars: nano features in orange
        for i, feat in enumerate(sorted_feats):
            if feat in NANO_FEATURES:
                bars[i].set_color("darkorange")
        ax1.legend(handles=[
            plt.Rectangle((0, 0), 1, 1, fc="steelblue", label="Spatial"),
            plt.Rectangle((0, 0), 1, 1, fc="darkorange", label="Nano detection"),
        ], fontsize=8, loc="lower right")

        # Right: scatter global vs within
        for feat in all_features:
            color = "darkorange" if feat in NANO_FEATURES else "steelblue"
            ax2.scatter(global_rho[feat], within_rho_mean[feat], color=color,
                        s=40, zorder=3)
            ax2.annotate(feat.replace("global_", "g_").replace("_", "\n", 1),
                         (global_rho[feat], within_rho_mean[feat]),
                         fontsize=5, ha="center", va="bottom")
        lims = [min(min(global_rho.values()), min(within_rho_mean.values())) - 0.05,
                max(max(global_rho.values()), max(within_rho_mean.values())) + 0.05]
        ax2.plot(lims, lims, "k--", lw=0.5, label="y=x (no drop)")
        ax2.set_xlabel("Global rho (pooled)")
        ax2.set_ylabel("Mean within-sequence rho")
        ax2.set_title("Global vs Within-Sequence rho")
        ax2.legend(fontsize=8)

        fig.suptitle("Page 1: Within-Sequence Feature Correlation",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 2: Simple model comparison (spatial only) ──
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12),
                                         gridspec_kw={"height_ratios": [1, 2]})

        # Top: mean per-seq R2 bar chart
        model_names = list(test2_results.keys())
        mean_r2s = [test2_results[m]["mean_per_seq_r2"] for m in model_names]
        overall_r2s = [test2_results[m]["overall_r2"] for m in model_names]
        x = np.arange(len(model_names))
        bw = 0.35
        ax1.bar(x - bw/2, mean_r2s, bw, label="Mean per-seq R2", color="steelblue")
        ax1.bar(x + bw/2, overall_r2s, bw, label="Overall R2 (misleading)", color="coral",
                alpha=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, fontsize=10)
        ax1.set_ylabel("R2")
        ax1.set_title("Simple Models (15 spatial features only)", fontsize=12,
                       fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.axhline(0, color="k", lw=0.5, ls="--")

        # Bottom: per-sequence R2 grouped bars
        val_seqs = [s for s in VAL_SEQUENCES if s in test2_results[model_names[0]]["per_seq"]]
        x_seq = np.arange(len(val_seqs))
        n_models = len(model_names)
        bw_seq = 0.8 / n_models
        colors = ["steelblue", "seagreen", "coral"]
        for i, mname in enumerate(model_names):
            r2s = [test2_results[mname]["per_seq"].get(s, 0) for s in val_seqs]
            ax2.bar(x_seq + i * bw_seq - 0.4 + bw_seq/2, r2s, bw_seq,
                    label=mname, color=colors[i % len(colors)], alpha=0.8)
        ax2.set_xticks(x_seq)
        labels = [f"{s}\n({seq_weather.get(s, '')})" for s in val_seqs]
        ax2.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax2.set_ylabel("Per-Sequence R2")
        ax2.set_title("Per-Sequence R2 (spatial features only)", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.axhline(0, color="k", lw=0.5, ls="--")

        fig.suptitle("Page 2: Simple Per-Frame Models (No LSTM, No Temporal)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 3: Detection features impact ──
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12),
                                         gridspec_kw={"height_ratios": [1, 2]})

        fs_names = list(test3_results.keys())
        mean_r2s = [test3_results[f]["mean_per_seq_r2"] for f in fs_names]
        overall_r2s = [test3_results[f]["overall_r2"] for f in fs_names]
        x = np.arange(len(fs_names))
        bw = 0.35
        ax1.bar(x - bw/2, mean_r2s, bw, label="Mean per-seq R2", color="steelblue")
        ax1.bar(x + bw/2, overall_r2s, bw, label="Overall R2 (misleading)", color="coral",
                alpha=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(fs_names, fontsize=9)
        ax1.set_ylabel("R2")
        ax1.set_title("Does Adding Nano@320 Detection Output Help?", fontsize=12,
                       fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.axhline(0, color="k", lw=0.5, ls="--")

        # Bottom: per-sequence R2 grouped bars
        colors_fs = ["steelblue", "darkorange", "seagreen"]
        n_fs = len(fs_names)
        bw_seq = 0.8 / n_fs
        for i, fname in enumerate(fs_names):
            r2s = [test3_results[fname]["per_seq"].get(s, 0) for s in val_seqs]
            ax2.bar(x_seq + i * bw_seq - 0.4 + bw_seq/2, r2s, bw_seq,
                    label=fname, color=colors_fs[i % len(colors_fs)], alpha=0.8)
        ax2.set_xticks(x_seq)
        ax2.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax2.set_ylabel("Per-Sequence R2")
        ax2.set_title("Per-Sequence R2 by Feature Set (Random Forest)", fontsize=11)
        ax2.legend(fontsize=7)
        ax2.axhline(0, color="k", lw=0.5, ls="--")

        fig.suptitle("Page 3: Detection Features Impact",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 4: Summary diagnosis ──
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.axis("off")

        # Gather summary numbers
        best_global = max(SPATIAL_FEATURES, key=lambda f: abs(global_rho[f]))
        best_within = max(SPATIAL_FEATURES, key=lambda f: within_rho_mean[f])
        avg_global = np.mean([abs(global_rho[f]) for f in SPATIAL_FEATURES])
        avg_within = np.mean([within_rho_mean[f] for f in SPATIAL_FEATURES])

        best_model = max(test2_results.keys(),
                         key=lambda m: test2_results[m]["mean_per_seq_r2"])
        best_model_r2 = test2_results[best_model]["mean_per_seq_r2"]

        spatial_only_r2 = test3_results["Spatial only (15)"]["mean_per_seq_r2"]
        nano_r2 = test3_results["Spatial + nano (19)"]["mean_per_seq_r2"]
        oracle_r2 = test3_results["Spatial + nano + GT (20)"]["mean_per_seq_r2"]

        lines = [
            "DIAGNOSTIC SUMMARY",
            "=" * 60,
            "",
            "TEST 1: Within-Sequence Correlation",
            f"  Mean |global rho| (spatial features): {avg_global:.3f}",
            f"  Mean within-seq rho (spatial features): {avg_within:.3f}",
            f"  Best global:  {best_global} = {global_rho[best_global]:.3f}",
            f"  Best within:  {best_within} = {within_rho_mean[best_within]:.3f}",
            f"  Signal drop: {avg_global - avg_within:.3f}",
            "",
            "TEST 2: Simple Per-Frame Models (spatial only)",
            f"  Best model: {best_model}",
            f"  Mean per-seq R2: {best_model_r2:.4f}",
            f"  (LSTM intra-seq R2 was: 0.200)",
            "",
            "TEST 3: Detection Features Impact (Random Forest)",
            f"  Spatial only (15):           mean per-seq R2 = {spatial_only_r2:.4f}",
            f"  Spatial + nano (19):         mean per-seq R2 = {nano_r2:.4f}",
            f"  Spatial + nano + GT (20):    mean per-seq R2 = {oracle_r2:.4f}",
            f"  Nano detection lift: {nano_r2 - spatial_only_r2:+.4f}",
            f"  Oracle (+ num_gt) lift: {oracle_r2 - spatial_only_r2:+.4f}",
            "",
            "=" * 60,
            "DIAGNOSIS:",
        ]

        # Auto-generate diagnosis
        if avg_within < 0.1:
            lines.append("  Within-seq rho < 0.1 => spatial features have almost NO")
            lines.append("  intra-sequence signal. The bottleneck is the FEATURES.")
        elif avg_within < 0.2:
            lines.append("  Within-seq rho 0.1-0.2 => weak but present intra-seq signal.")
            lines.append("  Features are the primary bottleneck, model is secondary.")
        else:
            lines.append("  Within-seq rho > 0.2 => moderate intra-seq signal exists.")
            lines.append("  Model may be the bottleneck — try more capacity.")

        if nano_r2 - spatial_only_r2 > 0.05:
            lines.append(f"  Adding nano detections helps (+{nano_r2 - spatial_only_r2:.3f}).")
            lines.append("  RECOMMENDATION: Use spatial + nano features for MPC.")
        else:
            lines.append(f"  Adding nano detections barely helps (+{nano_r2 - spatial_only_r2:.3f}).")

        if best_model_r2 > 0.15:
            lines.append("  Simple models match/beat LSTM => LSTM adds no value here.")
            lines.append("  Use simple per-frame model instead of LSTM for MPC.")
        elif best_model_r2 < 0.05:
            lines.append("  Even simple models fail => per-frame fn is fundamentally")
            lines.append("  hard to predict from image features alone.")

        text = "\n".join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                          edgecolor="gray"))

        fig.suptitle("Page 4: Diagnostic Summary", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"\nPDF saved -> {PDF_PATH}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    df = load_data()
    seq_weather = df.drop_duplicates("sequence").set_index("sequence")["weather"].to_dict()

    global_rho, within_rho_mean = test1_within_seq_correlation(df)
    test2_results = test2_simple_models(df)
    test3_results = test3_detection_features(df)

    generate_pdf(global_rho, within_rho_mean, test2_results, test3_results,
                 seq_weather)

    print("\nDone.")


if __name__ == "__main__":
    main()
