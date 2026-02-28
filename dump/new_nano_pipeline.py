#!/usr/bin/env python3
"""Pipeline: nano detection features experiment.

Compares:
  1) Nano-only (9 features) — inference-time detection stats
  2) Spatial-20 + Nano (29 features) — combined
  3) Spatial-20 baseline (20 features) — for reference

All use GRU-1L-32 with Dirichlet head, 3 seeds, at 0.5s window.
Includes per-sequence breakdown and intra-sequence signal analysis.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "new_all_features.csv"
DETECTION_CSV = SCRIPT_DIR.parent / "res_experiment" / "models_dependent_detection_features.csv"
OUT_PDF = SCRIPT_DIR / "new_nano_pipeline_results.pdf"

# Val sequences (fixed)
VAL_SEQS = [
    "MVI_39361", "MVI_20011", "MVI_20012", "MVI_20032", "MVI_20051",
    "MVI_40172", "MVI_40191", "MVI_40712", "MVI_40244", "MVI_40243",
    "MVI_40992", "MVI_40732", "MVI_39811", "MVI_40775", "MVI_40752",
    "MVI_63521", "MVI_40892", "MVI_63553", "MVI_63525", "MVI_63544",
]

SPATIAL_20 = [
    "fast_keypoints_half", "edge_density_coarse", "gradient_magnitude_mean",
    "gabor_nyquist_energy", "fast_keypoints_full", "glcm_homogeneity",
    "overall_edge_density", "top_edge_density", "top_high_freq_energy",
    "top_gradient_std", "fft_critical_band_ratio", "total_edge_components",
    "top_laplacian_var", "glcm_entropy", "mid_edge_density",
    "dark_channel_mean", "mid_gradient_std", "foreground_blob_count",
    "mid_high_freq_energy", "rms_contrast",
]

NANO_9 = [
    "nano_num_detections", "nano_mean_confidence", "nano_min_confidence",
    "nano_confidence_std", "nano_mean_box_area", "nano_min_box_area",
    "nano_box_area_std", "nano_detection_density", "nano_edge_box_ratio",
]

FPS = 30
WINDOW = 15  # 0.5s at 30fps
HORIZON = 15
TRAIN_STRIDE = 8
EVAL_STRIDE = 15
SEEDS = [42, 123, 7]

# ── Data ────────────────────────────────────────────────────────────────────


def load_data():
    """Load and merge features + detection features."""
    print("Loading data ...", flush=True)
    feat = pd.read_csv(FEATURES_CSV)
    det = pd.read_csv(DETECTION_CSV)
    nano_cols = ["sequence", "frame_id"] + NANO_9
    det = det[nano_cols]
    merged = feat.merge(det, on=["sequence", "frame_id"], how="left")
    # Fill any NaN nano features with 0
    for c in NANO_9:
        merged[c] = merged[c].fillna(0.0)
    print(f"  {len(merged)} rows, nano features merged", flush=True)
    return merged


def compute_threshold(df):
    """Compute median threshold over all sequences (same as other pipelines)."""
    all_targets = []
    for seq, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        fn = sdf["fn_nano"].values.astype(np.float32)
        n = len(sdf)
        total_need = WINDOW + HORIZON
        for t in range(0, n - total_need + 1, WINDOW):
            all_targets.append(fn[t + WINDOW:t + WINDOW + HORIZON].mean())
    threshold = float(np.median(all_targets))
    return threshold


def build_windows(df, feature_cols, threshold, split="train"):
    """Build shifted-label windows: input = W frames, target = mean(fn_nano) over NEXT W frames."""
    sequences = sorted(df["sequence"].unique())
    train_seqs = [s for s in sequences if s not in VAL_SEQS]
    val_seqs = [s for s in sequences if s in VAL_SEQS]
    stride = TRAIN_STRIDE if split == "train" else EVAL_STRIDE

    def _make(seqs):
        X_all, y_all, seq_labels = [], [], []
        tp_nano_last, tp_medium_last = [], []
        for seq in seqs:
            sdf = df[df["sequence"] == seq].sort_values("frame_id")
            feats = sdf[feature_cols].values.astype(np.float32)
            fn = sdf["fn_nano"].values.astype(np.float32)
            tp_n = sdf["tp_nano"].values
            tp_m = sdf["tp_medium"].values
            n = len(sdf)
            s = TRAIN_STRIDE if seq not in VAL_SEQS else EVAL_STRIDE
            for i in range(0, n - WINDOW - HORIZON + 1, s):
                x = feats[i:i + WINDOW]
                future_fn = fn[i + WINDOW:i + WINDOW + HORIZON]
                y = 1 if future_fn.mean() >= threshold else 0
                X_all.append(x)
                y_all.append(y)
                seq_labels.append(seq)
                tp_nano_last.append(tp_n[i + WINDOW - 1])
                tp_medium_last.append(tp_m[i + WINDOW - 1])
        return (np.array(X_all), np.array(y_all), seq_labels,
                np.array(tp_nano_last), np.array(tp_medium_last))

    X_tr, y_tr, _, _, _ = _make(train_seqs)
    X_va, y_va, va_seqs, tp_n_last, tp_m_last = _make(val_seqs)
    print(f"  Windows: train={len(X_tr)} (hard={y_tr.mean():.1%})  "
          f"val={len(X_va)} (hard={y_va.mean():.1%})", flush=True)
    return X_tr, y_tr, X_va, y_va, va_seqs, tp_n_last, tp_m_last


# ── Model ───────────────────────────────────────────────────────────────────


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class DirichletGRU(nn.Module):
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        _, h = self.gru(x)
        alpha = torch.exp(self.fc(h[-1])) + 1.0
        return alpha


def dirichlet_loss(alpha, y, epoch, max_epoch=60):
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    # Type II ML
    nll = -torch.log(p.gather(1, y.unsqueeze(1)) + 1e-8).squeeze(1)
    # KL annealing
    lam = min(1.0, epoch / (max_epoch * 0.6))
    alpha0 = torch.ones_like(alpha)
    kl = (torch.lgamma(alpha.sum(1)) - torch.lgamma(alpha0.sum(1))
          - (torch.lgamma(alpha) - torch.lgamma(alpha0)).sum(1)
          + ((alpha - alpha0) * (torch.digamma(alpha) - torch.digamma(S))).sum(1))
    return (nll + lam * kl).mean()


def train_gru(X_tr, y_tr, X_va, y_va, seed, n_features):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DirichletGRU(n_features, hidden=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

    ds_tr = SeqDataset(X_tr, y_tr)
    dl_tr = DataLoader(ds_tr, batch_size=128, shuffle=True, drop_last=False)

    best_acc, best_state = 0, None
    for epoch in range(60):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            alpha = model(xb)
            loss = dirichlet_loss(alpha, yb, epoch, 60)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
            av = model(Xv)
            preds = (av[:, 1] / av.sum(1)).cpu().numpy()
            pred_labels = (preds >= 0.5).astype(int)
            acc = (pred_labels == y_va).mean()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
        av = model(Xv)
        preds = (av[:, 1] / av.sum(1)).cpu().numpy()
        pred_labels = (preds >= 0.5).astype(int)

    return best_acc, pred_labels


def train_rf(X_tr, y_tr, X_va, y_va, seed):
    # Flatten sequences for RF
    X_tr_flat = X_tr.reshape(len(X_tr), -1)
    X_va_flat = X_va.reshape(len(X_va), -1)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1)
    rf.fit(X_tr_flat, y_tr)
    pred = rf.predict(X_va_flat)
    acc = (pred == y_va).mean()
    return acc, pred, rf.feature_importances_


def medium_sample_baseline(y_va, tp_nano_last, tp_medium_last):
    """Medium-sampling baseline: compare tp on last frame of input window."""
    pred = (tp_medium_last > tp_nano_last).astype(int)
    return (pred == y_va).mean()


# ── Intra-sequence analysis ─────────────────────────────────────────────────


def intra_sequence_analysis(y_va, pred_labels, va_seqs):
    """Per-sequence breakdown + intra-sequence signal check."""
    seq_names = sorted(set(va_seqs))
    results = {}
    for seq in seq_names:
        mask = np.array([s == seq for s in va_seqs])
        y_s = y_va[mask]
        p_s = pred_labels[mask]
        n = len(y_s)
        if n == 0:
            continue
        acc = (y_s == p_s).mean()
        hard_frac = y_s.mean()

        # Intra: accuracy on easy and hard windows separately
        easy_mask = y_s == 0
        hard_mask = y_s == 1
        easy_acc = (p_s[easy_mask] == 0).mean() if easy_mask.sum() > 0 else None
        hard_acc = (p_s[hard_mask] == 1).mean() if hard_mask.sum() > 0 else None

        # Transition accuracy
        transitions = 0
        trans_correct = 0
        for j in range(1, n):
            if y_s[j] != y_s[j - 1]:
                transitions += 1
                if p_s[j] == y_s[j]:
                    trans_correct += 1
        trans_acc = trans_correct / transitions if transitions > 0 else None

        results[seq] = {
            "n": n, "acc": acc, "hard_frac": hard_frac,
            "easy_acc": easy_acc, "hard_acc": hard_acc,
            "trans_acc": trans_acc, "transitions": transitions,
            "pred_hard_frac": p_s.mean(),
        }
    return results


# ── Correlation analysis ────────────────────────────────────────────────────


def nano_feature_correlations(df):
    """Spearman correlation of nano features with shifted fn_nano target."""
    from scipy.stats import spearmanr
    sequences = sorted(df["sequence"].unique())
    all_feats, all_targets = [], []
    for seq in sequences:
        sdf = df[df["sequence"] == seq].sort_values("frame_id")
        fn = sdf["fn_nano"].values
        n = len(sdf)
        for i in range(0, n - WINDOW - HORIZON + 1, WINDOW):
            x = sdf[NANO_9].iloc[i:i + WINDOW].mean().values
            future_fn = fn[i + WINDOW:i + WINDOW + HORIZON].mean()
            all_feats.append(x)
            all_targets.append(future_fn)
    all_feats = np.array(all_feats)
    all_targets = np.array(all_targets)

    corrs = {}
    for j, col in enumerate(NANO_9):
        rho, pval = spearmanr(all_feats[:, j], all_targets)
        corrs[col] = (rho, pval)
    return corrs


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    df = load_data()

    # Compute threshold (median of all target values, same as other pipelines)
    threshold = compute_threshold(df)
    print(f"Threshold: {threshold:.3f}", flush=True)

    # Nano feature correlations
    print("\n=== Nano feature Spearman correlations with future fn_nano ===", flush=True)
    corrs = nano_feature_correlations(df)
    corr_sorted = sorted(corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)
    for feat, (rho, pval) in corr_sorted:
        print(f"  {feat:30s}  ρ={rho:+.4f}  p={pval:.2e}", flush=True)

    # Feature sets
    feature_sets = {
        "Nano-9": NANO_9,
        "Spatial-20 + Nano-9": SPATIAL_20 + NANO_9,
        "Spatial-20 (baseline)": SPATIAL_20,
    }

    # Compute MedSample baseline once (using spatial-20 to get val windows)
    scaler_base = StandardScaler()
    train_df = df[~df["sequence"].isin(VAL_SEQS)]
    scaler_base.fit(train_df[SPATIAL_20].values)
    df_base = df.copy()
    df_base[SPATIAL_20] = scaler_base.transform(df_base[SPATIAL_20].values)
    _, _, _, y_va_base, _, tp_n_last, tp_m_last = build_windows(df_base, SPATIAL_20, threshold)
    med_acc = medium_sample_baseline(y_va_base, tp_n_last, tp_m_last)
    print(f"\nMedSample baseline: {med_acc:.1%}", flush=True)

    all_results = {}

    for fs_name, fs_cols in feature_sets.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Feature set: {fs_name} ({len(fs_cols)} features)", flush=True)
        print(f"{'='*60}", flush=True)

        # Normalize: fit on train only
        scaler = StandardScaler()
        train_data = df[~df["sequence"].isin(VAL_SEQS)]
        scaler.fit(train_data[fs_cols].values)
        df_copy = df.copy()
        df_copy[fs_cols] = scaler.transform(df_copy[fs_cols].values)

        X_tr, y_tr, X_va, y_va, va_seqs, _, _ = build_windows(df_copy, fs_cols, threshold)

        # GRU 3-seed
        gru_accs = []
        best_gru_acc = 0
        best_gru_preds = None
        for seed in SEEDS:
            acc, preds = train_gru(X_tr, y_tr, X_va, y_va, seed, len(fs_cols))
            gru_accs.append(acc)
            print(f"  GRU seed={seed}: {acc:.1%}", flush=True)
            if acc > best_gru_acc:
                best_gru_acc = acc
                best_gru_preds = preds

        # RF
        rf_acc, rf_preds, rf_imp = train_rf(X_tr, y_tr, X_va, y_va, 42)
        print(f"  RF: {rf_acc:.1%}", flush=True)

        # Intra-sequence analysis (best GRU)
        intra = intra_sequence_analysis(y_va, best_gru_preds, va_seqs)

        # Count mixed sequences with real intra-sequence signal
        mixed_with_signal = 0
        mixed_total = 0
        for seq, r in intra.items():
            if r["hard_frac"] > 0.05 and r["hard_frac"] < 0.95:
                mixed_total += 1
                if r["easy_acc"] is not None and r["hard_acc"] is not None:
                    if r["easy_acc"] > 0.5 and r["hard_acc"] > 0.5:
                        mixed_with_signal += 1

        print(f"  Intra-sequence signal: {mixed_with_signal}/{mixed_total} mixed sequences", flush=True)

        # Aggregate transition accuracy
        total_trans = sum(r["transitions"] for r in intra.values() if r["transitions"])
        total_trans_correct = sum(
            int(r["trans_acc"] * r["transitions"])
            for r in intra.values()
            if r["trans_acc"] is not None and r["transitions"] > 0
        )
        agg_trans_acc = total_trans_correct / total_trans if total_trans > 0 else 0
        print(f"  Aggregate transition accuracy: {agg_trans_acc:.1%} ({total_trans} transitions)", flush=True)

        all_results[fs_name] = {
            "gru_accs": gru_accs,
            "gru_mean": np.mean(gru_accs),
            "gru_std": np.std(gru_accs),
            "rf_acc": rf_acc,
            "rf_importances": dict(zip(fs_cols, rf_imp)) if len(rf_imp) == len(fs_cols) else {},
            "intra": intra,
            "mixed_signal": mixed_with_signal,
            "mixed_total": mixed_total,
            "trans_acc": agg_trans_acc,
        }

    # ── PDF Report ───────────────────────────────────────────────────────

    print(f"\nGenerating PDF report ...", flush=True)
    with PdfPages(str(OUT_PDF)) as pdf:

        # Page 1: Summary bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        fs_names = list(all_results.keys())
        gru_means = [all_results[n]["gru_mean"] for n in fs_names]
        gru_stds = [all_results[n]["gru_std"] for n in fs_names]
        rf_accs = [all_results[n]["rf_acc"] for n in fs_names]

        x = np.arange(len(fs_names))
        width = 0.3
        bars1 = ax.bar(x - width / 2, gru_means, width, yerr=gru_stds, label="GRU-1L-32", capsize=4, color="#4C72B0")
        bars2 = ax.bar(x + width / 2, rf_accs, width, label="RF", color="#DD8452")
        ax.axhline(med_acc, color="red", ls="--", lw=1.5, label=f"MedSample ({med_acc:.1%})")
        ax.set_xticks(x)
        ax.set_xticklabels(fs_names, fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title("Nano Detection Features Experiment (0.5s window)")
        ax.set_ylim(0.5, 1.0)
        ax.legend()

        for b in bars1:
            ax.annotate(f"{b.get_height():.1%}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
        for b in bars2:
            ax.annotate(f"{b.get_height():.1%}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 2: Nano feature correlations
        fig, ax = plt.subplots(figsize=(8, 5))
        feats_sorted = [f for f, _ in corr_sorted]
        rhos = [v[0] for _, v in corr_sorted]
        colors = ["#4C72B0" if r > 0 else "#C44E52" for r in rhos]
        ax.barh(range(len(feats_sorted)), rhos, color=colors)
        ax.set_yticks(range(len(feats_sorted)))
        ax.set_yticklabels(feats_sorted, fontsize=8)
        ax.set_xlabel("Spearman ρ")
        ax.set_title("Nano Feature Correlations with Future fn_nano")
        ax.axvline(0, color="black", lw=0.5)
        ax.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 3: RF feature importances for combined set
        combined_key = "Spatial-20 + Nano-9"
        if combined_key in all_results and all_results[combined_key]["rf_importances"]:
            imp = all_results[combined_key]["rf_importances"]
            imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20]
            fig, ax = plt.subplots(figsize=(8, 6))
            feats = [f for f, _ in imp_sorted]
            vals = [v for _, v in imp_sorted]
            colors = ["#DD8452" if f.startswith("nano_") else "#4C72B0" for f in feats]
            ax.barh(range(len(feats)), vals, color=colors)
            ax.set_yticks(range(len(feats)))
            ax.set_yticklabels(feats, fontsize=8)
            ax.set_xlabel("RF Feature Importance")
            ax.set_title(f"Top 20 Features ({combined_key})\nOrange=Nano, Blue=Spatial")
            ax.invert_yaxis()
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

        # Page 4: Per-sequence intra-sequence analysis (best feature set)
        best_fs = max(all_results.keys(), key=lambda k: all_results[k]["gru_mean"])
        intra = all_results[best_fs]["intra"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        seq_names = sorted(intra.keys())
        easy_accs = [intra[s]["easy_acc"] if intra[s]["easy_acc"] is not None else 0 for s in seq_names]
        hard_accs = [intra[s]["hard_acc"] if intra[s]["hard_acc"] is not None else 0 for s in seq_names]
        hard_fracs = [intra[s]["hard_frac"] for s in seq_names]

        ax = axes[0]
        y_pos = np.arange(len(seq_names))
        ax.barh(y_pos - 0.2, easy_accs, 0.4, label="Easy acc", color="#55A868")
        ax.barh(y_pos + 0.2, hard_accs, 0.4, label="Hard acc", color="#C44E52")
        ax.axvline(0.5, color="black", ls="--", lw=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{s}\n({h:.0%} hard)" for s, h in zip(seq_names, hard_fracs)], fontsize=6)
        ax.set_xlabel("Accuracy")
        ax.set_title(f"Per-Sequence Accuracy ({best_fs})")
        ax.legend(fontsize=8)

        ax = axes[1]
        trans_accs = [intra[s]["trans_acc"] if intra[s]["trans_acc"] is not None else 0 for s in seq_names]
        n_trans = [intra[s]["transitions"] for s in seq_names]
        colors = ["#4C72B0" if t > 0 else "#CCCCCC" for t in n_trans]
        ax.barh(y_pos, trans_accs, color=colors)
        ax.axvline(0.5, color="red", ls="--", lw=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{s} ({n})" for s, n in zip(seq_names, n_trans)], fontsize=6)
        ax.set_xlabel("Transition Accuracy")
        ax.set_title("Transition Accuracy (n transitions)")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"\nSaved PDF: {OUT_PDF}", flush=True)

    # ── Final summary ────────────────────────────────────────────────────

    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"MedSample baseline: {med_acc:.1%}", flush=True)
    for fs_name, r in all_results.items():
        print(f"\n{fs_name}:", flush=True)
        print(f"  GRU: {r['gru_mean']:.1%} ± {r['gru_std']:.1%}  (seeds: {[f'{a:.1%}' for a in r['gru_accs']]})", flush=True)
        print(f"  RF:  {r['rf_acc']:.1%}", flush=True)
        print(f"  Intra-sequence signal: {r['mixed_signal']}/{r['mixed_total']} mixed seqs", flush=True)
        print(f"  Transition accuracy:   {r['trans_acc']:.1%}", flush=True)
    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
