#!/usr/bin/env python3
"""Multi-seed validation for GRU-1L-32 and LSTM-2L-64 at 0.5s window.

Runs 5 seeds each, reports mean ± std for accuracy, F1, AUC, miss rate.
Also reports per-sequence accuracy for the best GRU seed.
Also checks class balance across window configs to explain the 1.0s dip.

Output: new_features/new_seed_validation_results.pdf
"""

import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "new_all_features.csv"
OUT_PDF = SCRIPT_DIR / "new_seed_validation_results.pdf"

META_COLS = ["sequence", "frame_id", "weather", "fn_nano", "num_gt",
             "tp_nano", "tp_medium", "recoverable_fn"]
TARGET = "fn_nano"
N_CLASSES = 2
KL_ANNEALING_EPOCHS = 10

SELECTED_FEATURES = [
    "fast_keypoints_half", "edge_density_coarse", "gradient_magnitude_mean",
    "gabor_nyquist_energy", "fast_keypoints_full", "glcm_homogeneity",
    "overall_edge_density", "top_edge_density", "top_high_freq_energy",
    "top_gradient_std", "fft_critical_band_ratio", "total_edge_components",
    "top_laplacian_var", "glcm_entropy", "mid_edge_density",
    "dark_channel_mean", "mid_gradient_std", "foreground_blob_count",
    "mid_high_freq_energy", "rms_contrast",
]

VAL_SEQUENCES = [
    "MVI_39361", "MVI_20011", "MVI_20012", "MVI_20032", "MVI_20051",
    "MVI_40172", "MVI_40191", "MVI_40712", "MVI_40244", "MVI_40243",
    "MVI_40992", "MVI_40732", "MVI_39811", "MVI_40775", "MVI_40752",
    "MVI_63521", "MVI_40892", "MVI_63553", "MVI_63525", "MVI_63544",
]

SEEDS = [42, 123, 456, 789, 2024]

# All window configs to check class balance
WINDOW_CONFIGS = [
    {"window": 15, "horizon": 15, "train_stride": 8,  "eval_stride": 15, "label": "0.5s"},
    {"window": 30, "horizon": 30, "train_stride": 15, "eval_stride": 30, "label": "1.0s"},
    {"window": 45, "horizon": 45, "train_stride": 23, "eval_stride": 45, "label": "1.5s"},
    {"window": 60, "horizon": 60, "train_stride": 30, "eval_stride": 60, "label": "2.0s"},
]

# Models to validate with multi-seed
MODEL_CONFIGS = [
    {"type": "gru",  "layers": 1, "hidden": 32, "label": "GRU-1L-32"},
    {"type": "lstm", "layers": 2, "hidden": 64, "label": "LSTM-2L-64"},
    {"type": "lstm", "layers": 1, "hidden": 32, "label": "LSTM-1L-32"},
]


# ═══════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════

def build_windows(df, feature_cols, target_col, window, horizon,
                  stride, scaler=None, threshold=None):
    total_need = window + horizon
    X_list, y_list, seq_ids = [], [], []
    tp_nano_last, tp_medium_last = [], []

    for seq_name, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        n = len(sdf)
        if n < total_need:
            continue
        feat_vals = sdf[feature_cols].values
        target_vals = sdf[target_col].values
        tp_n = sdf["tp_nano"].values
        tp_m = sdf["tp_medium"].values
        if scaler is not None:
            feat_vals = scaler.transform(feat_vals)
        for t in range(0, n - total_need + 1, stride):
            X_list.append(feat_vals[t : t + window])
            y_list.append(int(target_vals[t + window : t + window + horizon].mean() >= threshold))
            seq_ids.append((seq_name, sdf["frame_id"].iloc[t]))
            tp_nano_last.append(tp_n[t + window - 1])
            tp_medium_last.append(tp_m[t + window - 1])

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.int64),
            seq_ids,
            {"tp_nano_last": np.array(tp_nano_last),
             "tp_medium_last": np.array(tp_medium_last)})


def build_rf_features(X):
    n, w, f = X.shape
    tc = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
    tv = (tc ** 2).sum()
    return np.hstack([X.mean(1), X.std(1),
                      (np.einsum("ijk,j->ik", X.astype(np.float64), tc) / tv).astype(np.float32)])


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

def dirichlet_loss(alphas, targets, epoch):
    y_oh = torch.zeros_like(alphas)
    y_oh.scatter_(1, targets.unsqueeze(1), 1.0)
    S = alphas.sum(dim=1, keepdim=True)
    loss_ml = (y_oh * (torch.digamma(S) - torch.digamma(alphas))).sum(dim=1)
    alpha_tilde = y_oh + (1.0 - y_oh) * alphas
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    kl = (torch.lgamma(S_tilde.squeeze(1))
          - torch.lgamma(torch.tensor(float(N_CLASSES)))
          - torch.lgamma(alpha_tilde).sum(dim=1)
          + ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde)
             - torch.digamma(S_tilde))).sum(dim=1))
    lam = min(1.0, epoch / max(KL_ANNEALING_EPOCHS, 1))
    return (loss_ml + lam * kl).mean()


class DirichletSeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, rnn_type):
        super().__init__()
        self.rnn_type = rnn_type
        drop = 0.2 if num_layers > 1 else 0.0
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=drop)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=drop)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, max(16, hidden_size // 2)),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(max(16, hidden_size // 2), N_CLASSES),
            nn.Softplus())

    def forward(self, x):
        if self.rnn_type == "lstm":
            _, (h, _) = self.rnn(x)
        else:
            _, h = self.rnn(x)
        return self.head(h[-1]) + 1.0

    def predict(self, x):
        a = self.forward(x)
        S = a.sum(1, keepdim=True)
        p = a / S
        return p.argmax(1), p[:, 1], N_CLASSES / S.squeeze(1)


def train_one(X_tr, y_tr, X_va, y_va, mcfg, n_feat, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DirichletSeqModel(n_feat, mcfg["hidden"], mcfg["layers"],
                               mcfg["type"]).to(device)

    loader_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr),
                                          torch.from_numpy(y_tr)),
                           batch_size=64, shuffle=True,
                           generator=torch.Generator().manual_seed(seed))
    loader_va = DataLoader(TensorDataset(torch.from_numpy(X_va),
                                          torch.from_numpy(y_va)),
                           batch_size=256)

    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=10, min_lr=1e-6)

    best_acc, best_state, pat = 0.0, None, 0
    best_ep = 0

    for ep in range(200):
        model.train()
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            loss = dirichlet_loss(model(xb), yb, ep)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for xb, yb in loader_va:
                xb, yb = xb.to(device), yb.to(device)
                a = model(xb)
                vc += ((a / a.sum(1, keepdim=True)).argmax(1) == yb).sum().item()
                vt += len(yb)
        va = vc / vt
        sched.step(va)
        if va > best_acc:
            best_acc = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep; pat = 0
        else:
            pat += 1
        if pat >= 25:
            break

    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, best_acc, best_ep


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load
    df = pd.read_csv(FEATURES_CSV)
    for c in SELECTED_FEATURES:
        if "temporal" in c or c == "motion_pixel_ratio":
            df[c] = df[c].fillna(0.0)

    actual_val = [s for s in VAL_SEQUENCES if s in set(df["sequence"].unique())]
    train_df = df[~df["sequence"].isin(actual_val)].copy()
    val_df = df[df["sequence"].isin(actual_val)].copy()

    # Threshold
    all_targets = []
    for _, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        for t in range(0, len(sdf) - 60 + 1, 30):
            all_targets.append(sdf[TARGET].iloc[t+30:t+60].mean())
    threshold = float(np.median(all_targets))
    print(f"Threshold: {threshold:.3f}\n")

    scaler = StandardScaler()
    scaler.fit(train_df[SELECTED_FEATURES].values)

    # ═══════════════════════════════════════════════════════════════
    # 1. CLASS BALANCE CHECK ACROSS WINDOWS
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("CLASS BALANCE ACROSS WINDOW CONFIGS")
    print("=" * 70)

    for wcfg in WINDOW_CONFIGS:
        w, h = wcfg["window"], wcfg["horizon"]
        ts, es = wcfg["train_stride"], wcfg["eval_stride"]

        X_tr, y_tr, _, _ = build_windows(
            train_df, SELECTED_FEATURES, TARGET, w, h, ts, scaler, threshold)
        X_va, y_va, _, _ = build_windows(
            val_df, SELECTED_FEATURES, TARGET, w, h, es, scaler, threshold)

        print(f"  {wcfg['label']:5s}: train={len(X_tr):6,} ({y_tr.mean():.1%} hard) "
              f"val={len(X_va):5,} ({y_va.mean():.1%} hard) "
              f"majority={max(y_va.mean(), 1-y_va.mean()):.1%}")

    # ═══════════════════════════════════════════════════════════════
    # 2. MULTI-SEED VALIDATION AT 0.5s
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("MULTI-SEED VALIDATION (0.5s window, 5 seeds)")
    print("=" * 70)

    wcfg = WINDOW_CONFIGS[0]  # 0.5s
    X_tr, y_tr, tr_ids, tr_meta = build_windows(
        train_df, SELECTED_FEATURES, TARGET,
        wcfg["window"], wcfg["horizon"], wcfg["train_stride"],
        scaler, threshold)
    X_va, y_va, va_ids, va_meta = build_windows(
        val_df, SELECTED_FEATURES, TARGET,
        wcfg["window"], wcfg["horizon"], wcfg["eval_stride"],
        scaler, threshold)

    print(f"  Train: {len(X_tr):,} windows, Val: {len(X_va):,} windows\n")

    # Medium-sampling baseline (deterministic — no seed)
    tr_gap = tr_meta["tp_medium_last"] - tr_meta["tp_nano_last"]
    va_gap = va_meta["tp_medium_last"] - va_meta["tp_nano_last"]
    best_mg_acc, best_mg_thr = 0, 0
    for thr in range(-2, 8):
        preds = (tr_gap >= thr).astype(int)
        acc = accuracy_score(y_tr, preds)
        if acc > best_mg_acc:
            best_mg_acc = acc; best_mg_thr = thr
    med_pred = (va_gap >= best_mg_thr).astype(int)
    med_acc = accuracy_score(y_va, med_pred)
    med_f1 = f1_score(y_va, med_pred, zero_division=0)
    cm = confusion_matrix(y_va, med_pred)
    tn, fp, fn_c, tp = cm.ravel()
    med_miss = fn_c / (fn_c + tp) if (fn_c + tp) > 0 else 0
    print(f"  MedSample (deterministic): Acc={med_acc:.3f} F1={med_f1:.3f} "
          f"Miss={med_miss:.1%}\n")

    # RF baseline (single seed but deterministic given data)
    X_tr_rf = build_rf_features(X_tr)
    X_va_rf = build_rf_features(X_va)
    rf = RandomForestClassifier(n_estimators=500, max_depth=20,
                                 min_samples_leaf=10, max_features="sqrt",
                                 n_jobs=-1, random_state=42)
    rf.fit(X_tr_rf, y_tr)
    rf_pred = rf.predict(X_va_rf)
    rf_prob = rf.predict_proba(X_va_rf)[:, 1]
    rf_acc = accuracy_score(y_va, rf_pred)
    rf_f1 = f1_score(y_va, rf_pred, zero_division=0)
    rf_auc = roc_auc_score(y_va, rf_prob)
    cm = confusion_matrix(y_va, rf_pred)
    tn, fp, fn_c, tp = cm.ravel()
    rf_miss = fn_c / (fn_c + tp) if (fn_c + tp) > 0 else 0
    print(f"  RF (deterministic): Acc={rf_acc:.3f} F1={rf_f1:.3f} "
          f"AUC={rf_auc:.3f} Miss={rf_miss:.1%}\n")

    all_seed_results = {}

    for mcfg in MODEL_CONFIGS:
        mlabel = mcfg["label"]
        print(f"  {mlabel}:")
        seed_metrics = {"acc": [], "f1": [], "auc": [], "miss": [], "epoch": []}
        best_seed_model = None
        best_seed_acc = 0

        for seed in SEEDS:
            model, best_acc, best_ep = train_one(
                X_tr, y_tr, X_va, y_va, mcfg,
                len(SELECTED_FEATURES), device, seed)

            with torch.no_grad():
                pred, prob, unc = model.predict(
                    torch.from_numpy(X_va).to(device))
            pred = pred.cpu().numpy()
            prob = prob.cpu().numpy()

            acc = accuracy_score(y_va, pred)
            f1 = f1_score(y_va, pred, zero_division=0)
            auc = roc_auc_score(y_va, prob)
            cm = confusion_matrix(y_va, pred)
            tn, fp, fn_c, tp = cm.ravel()
            miss = fn_c / (fn_c + tp) if (fn_c + tp) > 0 else 0

            seed_metrics["acc"].append(acc)
            seed_metrics["f1"].append(f1)
            seed_metrics["auc"].append(auc)
            seed_metrics["miss"].append(miss)
            seed_metrics["epoch"].append(best_ep)

            if acc > best_seed_acc:
                best_seed_acc = acc
                best_seed_model = (pred, prob, unc.cpu().numpy() if isinstance(unc, torch.Tensor) else unc)

            print(f"    seed={seed}: Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f} "
                  f"Miss={miss:.1%} ep={best_ep}")

        # Summary
        for metric in ["acc", "f1", "auc", "miss"]:
            vals = np.array(seed_metrics[metric])
            print(f"    {metric:>4s}: {vals.mean():.3f} ± {vals.std():.3f} "
                  f"[{vals.min():.3f}, {vals.max():.3f}]")

        all_seed_results[mlabel] = {
            "metrics": seed_metrics,
            "best_pred": best_seed_model,
        }
        print()

    # ═══════════════════════════════════════════════════════════════
    # 3. PER-SEQUENCE ACCURACY FOR BEST GRU SEED
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PER-SEQUENCE ACCURACY — GRU-1L-32 (best seed, 0.5s)")
    print("=" * 70)

    gru_pred = all_seed_results["GRU-1L-32"]["best_pred"][0]

    seq_data = defaultdict(lambda: {"t": [], "p": []})
    for i, (sn, _) in enumerate(va_ids):
        seq_data[sn]["t"].append(y_va[i])
        seq_data[sn]["p"].append(gru_pred[i])

    mixed_accs = []
    print(f"\n  {'Sequence':<20s} {'Acc':>6s} {'N':>5s} {'Mixed':>5s} {'Hard%':>6s}")
    print(f"  {'-'*47}")
    for sn in sorted(seq_data, key=lambda s: accuracy_score(
            np.array(seq_data[s]["t"]), np.array(seq_data[s]["p"]))):
        yt = np.array(seq_data[sn]["t"])
        yp = np.array(seq_data[sn]["p"])
        acc = accuracy_score(yt, yp)
        mixed = len(np.unique(yt)) > 1
        hard_pct = yt.mean()
        if mixed:
            mixed_accs.append(acc)
        print(f"  {sn:<20s} {acc:6.3f} {len(yt):5d} {'yes' if mixed else 'no':>5s} "
              f"{hard_pct:6.1%}")

    if mixed_accs:
        print(f"\n  Mixed sequences: {len(mixed_accs)}")
        print(f"  Median: {np.median(mixed_accs):.3f}, Mean: {np.mean(mixed_accs):.3f}")
        for thr in [50, 60, 70, 75, 80]:
            n = sum(1 for a in mixed_accs if a * 100 >= thr)
            print(f"    above {thr}%: {n}/{len(mixed_accs)}")

    # ═══════════════════════════════════════════════════════════════
    # 4. ALSO RUN MULTI-SEED AT 1.0s TO COMPARE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("MULTI-SEED AT 1.0s (for comparison)")
    print("=" * 70)

    wcfg_1s = WINDOW_CONFIGS[1]
    X_tr_1s, y_tr_1s, _, _ = build_windows(
        train_df, SELECTED_FEATURES, TARGET,
        wcfg_1s["window"], wcfg_1s["horizon"], wcfg_1s["train_stride"],
        scaler, threshold)
    X_va_1s, y_va_1s, _, _ = build_windows(
        val_df, SELECTED_FEATURES, TARGET,
        wcfg_1s["window"], wcfg_1s["horizon"], wcfg_1s["eval_stride"],
        scaler, threshold)

    print(f"  Train: {len(X_tr_1s):,}, Val: {len(X_va_1s):,}\n")

    for mcfg in [{"type": "gru", "layers": 1, "hidden": 32, "label": "GRU-1L-32"},
                 {"type": "lstm", "layers": 2, "hidden": 64, "label": "LSTM-2L-64"}]:
        mlabel = mcfg["label"]
        print(f"  {mlabel} @ 1.0s:")
        accs_1s = []
        for seed in SEEDS:
            model, _, ep = train_one(
                X_tr_1s, y_tr_1s, X_va_1s, y_va_1s, mcfg,
                len(SELECTED_FEATURES), device, seed)
            with torch.no_grad():
                pred, _, _ = model.predict(torch.from_numpy(X_va_1s).to(device))
            acc = accuracy_score(y_va_1s, pred.cpu().numpy())
            accs_1s.append(acc)
            print(f"    seed={seed}: Acc={acc:.3f} ep={ep}")
        accs_1s = np.array(accs_1s)
        print(f"    Mean: {accs_1s.mean():.3f} ± {accs_1s.std():.3f}\n")

    # ═══════════════════════════════════════════════════════════════
    # 5. PDF
    # ═══════════════════════════════════════════════════════════════
    print("Generating PDF ...", flush=True)

    with PdfPages(OUT_PDF) as pdf:

        # Page 1: Seed variance comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Accuracy across seeds
        ax = axes[0]
        x_pos = np.arange(len(MODEL_CONFIGS))
        for i, mcfg in enumerate(MODEL_CONFIGS):
            ml = mcfg["label"]
            vals = all_seed_results[ml]["metrics"]["acc"]
            ax.bar(i, np.mean(vals), yerr=np.std(vals), capsize=5,
                   color=["#388E3C", "#1976D2", "#42A5F5"][i], alpha=0.8)
            for v in vals:
                ax.scatter(i + np.random.uniform(-0.15, 0.15), v,
                          color="black", s=20, zorder=5, alpha=0.6)
        ax.axhline(med_acc, color="red", ls="--", lw=1.5, label=f"MedSample ({med_acc:.3f})")
        ax.axhline(rf_acc, color="orange", ls="--", lw=1.5, label=f"RF ({rf_acc:.3f})")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m["label"] for m in MODEL_CONFIGS], fontsize=9)
        ax.set_ylabel("Val Accuracy")
        ax.set_title("Accuracy Across 5 Seeds (0.5s window)")
        ax.legend(fontsize=8)
        ax.set_ylim(0.7, 0.95)

        # Miss rate across seeds
        ax = axes[1]
        for i, mcfg in enumerate(MODEL_CONFIGS):
            ml = mcfg["label"]
            vals = [m * 100 for m in all_seed_results[ml]["metrics"]["miss"]]
            ax.bar(i, np.mean(vals), yerr=np.std(vals), capsize=5,
                   color=["#388E3C", "#1976D2", "#42A5F5"][i], alpha=0.8)
            for v in vals:
                ax.scatter(i + np.random.uniform(-0.15, 0.15), v,
                          color="black", s=20, zorder=5, alpha=0.6)
        ax.axhline(med_miss * 100, color="red", ls="--", lw=1.5,
                   label=f"MedSample ({med_miss:.1%})")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m["label"] for m in MODEL_CONFIGS], fontsize=9)
        ax.set_ylabel("Miss Rate %")
        ax.set_title("Miss Rate Across 5 Seeds (0.5s window)")
        ax.legend(fontsize=8)

        fig.suptitle("Multi-Seed Validation — Is the GRU Result Real?",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close()

        # Page 2: Summary
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")

        lines = [
            "=" * 70,
            "MULTI-SEED VALIDATION SUMMARY (0.5s window, DETRAC)",
            "=" * 70, "",
        ]

        for mcfg in MODEL_CONFIGS:
            ml = mcfg["label"]
            m = all_seed_results[ml]["metrics"]
            lines.append(f"  {ml}:")
            lines.append(f"    Acc:  {np.mean(m['acc']):.3f} ± {np.std(m['acc']):.3f} "
                        f"[{np.min(m['acc']):.3f} — {np.max(m['acc']):.3f}]")
            lines.append(f"    F1:   {np.mean(m['f1']):.3f} ± {np.std(m['f1']):.3f}")
            lines.append(f"    AUC:  {np.mean(m['auc']):.3f} ± {np.std(m['auc']):.3f}")
            lines.append(f"    Miss: {np.mean(m['miss']):.3f} ± {np.std(m['miss']):.3f}")
            lines.append("")

        lines.extend([
            f"  Baselines (deterministic):",
            f"    MedSample: Acc={med_acc:.3f} Miss={med_miss:.1%}",
            f"    RF:        Acc={rf_acc:.3f} Miss={rf_miss:.1%}",
            "", "=" * 70,
        ])

        # Verdict
        gru_mean = np.mean(all_seed_results["GRU-1L-32"]["metrics"]["acc"])
        gru_std = np.std(all_seed_results["GRU-1L-32"]["metrics"]["acc"])
        if gru_mean - gru_std > med_acc:
            lines.append(f"VERDICT: GRU consistently beats MedSample "
                        f"({gru_mean:.3f}±{gru_std:.3f} vs {med_acc:.3f})")
        elif gru_mean > med_acc:
            lines.append(f"VERDICT: GRU beats MedSample on average but overlaps "
                        f"({gru_mean:.3f}±{gru_std:.3f} vs {med_acc:.3f})")
        else:
            lines.append(f"VERDICT: GRU does NOT reliably beat MedSample "
                        f"({gru_mean:.3f}±{gru_std:.3f} vs {med_acc:.3f})")

        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontfamily="monospace", fontsize=9, verticalalignment="top")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"PDF: {OUT_PDF}")
    print("\nDone.")


if __name__ == "__main__":
    main()
