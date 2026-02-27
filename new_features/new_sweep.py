#!/usr/bin/env python3
"""Hyperparameter sweep: model architecture × window size.

Tests DETRAC's 20 selected features with:
  - 4 model configs: LSTM-2L-64, LSTM-1L-32, LSTM-1L-16, GRU-1L-32
  - 4 window sizes: 0.5s, 1.0s, 1.5s, 2.0s
  - RF baseline for each window size
  - Medium-sampling baseline

All on DETRAC dataset, using the same val split as before.

Output: new_features/new_sweep_results.pdf
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
OUT_PDF = SCRIPT_DIR / "new_sweep_results.pdf"

META_COLS = ["sequence", "frame_id", "weather", "fn_nano", "num_gt",
             "tp_nano", "tp_medium", "recoverable_fn"]

TARGET = "fn_nano"
N_CLASSES = 2
KL_ANNEALING_EPOCHS = 10
FPS = 30  # DETRAC fps

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

# ── Sweep configs ──────────────────────────────────────────────────
WINDOW_CONFIGS = [
    {"window": 15, "horizon": 15, "train_stride": 8,  "eval_stride": 15, "label": "0.5s"},
    {"window": 30, "horizon": 30, "train_stride": 15, "eval_stride": 30, "label": "1.0s"},
    {"window": 45, "horizon": 45, "train_stride": 23, "eval_stride": 45, "label": "1.5s"},
    {"window": 60, "horizon": 60, "train_stride": 30, "eval_stride": 60, "label": "2.0s"},
]

MODEL_CONFIGS = [
    {"type": "lstm", "layers": 2, "hidden": 64, "label": "LSTM-2L-64"},
    {"type": "lstm", "layers": 1, "hidden": 32, "label": "LSTM-1L-32"},
    {"type": "lstm", "layers": 1, "hidden": 16, "label": "LSTM-1L-16"},
    {"type": "gru",  "layers": 1, "hidden": 32, "label": "GRU-1L-32"},
]


# ═══════════════════════════════════════════════════════════════════
# Data Pipeline
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
            x_window = feat_vals[t : t + window]
            target_mean = target_vals[t + window : t + window + horizon].mean()
            X_list.append(x_window)
            y_list.append(int(target_mean >= threshold))
            seq_ids.append((seq_name, sdf["frame_id"].iloc[t]))
            last_idx = t + window - 1
            tp_nano_last.append(tp_n[last_idx])
            tp_medium_last.append(tp_m[last_idx])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    meta = {
        "tp_nano_last": np.array(tp_nano_last),
        "tp_medium_last": np.array(tp_medium_last),
    }
    return X, y, seq_ids, meta


def build_rf_features(X_windows):
    n_samples, window_len, n_feat = X_windows.shape
    t_centered = np.arange(window_len, dtype=np.float64) - (window_len - 1) / 2.0
    t_var = (t_centered ** 2).sum()
    means = X_windows.mean(axis=1)
    stds = X_windows.std(axis=1)
    slopes = np.einsum("ijk,j->ik", X_windows.astype(np.float64),
                       t_centered) / t_var
    return np.hstack([means, stds, slopes.astype(np.float32)])


# ═══════════════════════════════════════════════════════════════════
# Dirichlet Loss + Models
# ═══════════════════════════════════════════════════════════════════

def dirichlet_loss(alphas, targets, epoch):
    y_oh = torch.zeros_like(alphas)
    y_oh.scatter_(1, targets.unsqueeze(1), 1.0)
    S = alphas.sum(dim=1, keepdim=True)
    loss_ml = (y_oh * (torch.digamma(S) - torch.digamma(alphas))).sum(dim=1)
    alpha_tilde = y_oh + (1.0 - y_oh) * alphas
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(S_tilde.squeeze(1))
        - torch.lgamma(torch.tensor(float(N_CLASSES)))
        - torch.lgamma(alpha_tilde).sum(dim=1)
        + ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde)
           - torch.digamma(S_tilde))).sum(dim=1)
    )
    lam = min(1.0, epoch / max(KL_ANNEALING_EPOCHS, 1))
    return (loss_ml + lam * kl).mean()


class DirichletSeqModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1,
                 dropout=0.2, rnn_type="lstm"):
        super().__init__()
        self.rnn_type = rnn_type
        rnn_dropout = dropout if num_layers > 1 else 0.0

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=rnn_dropout)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              dropout=rnn_dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, max(16, hidden_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_size // 2), N_CLASSES),
            nn.Softplus(),
        )

    def forward(self, x):
        if self.rnn_type == "lstm":
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)
        return self.head(h_n[-1]) + 1.0

    def predict(self, x):
        alphas = self.forward(x)
        S = alphas.sum(dim=1, keepdim=True)
        probs = alphas / S
        uncertainty = N_CLASSES / S.squeeze(1)
        return probs.argmax(dim=1), probs[:, 1], uncertainty


def train_model(X_train, y_train, X_val, y_val, model_cfg, n_feat, device):
    model = DirichletSeqModel(
        input_size=n_feat,
        hidden_size=model_cfg["hidden"],
        num_layers=model_cfg["layers"],
        rnn_type=model_cfg["type"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=64, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=256)

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6)

    best_acc, best_state, patience = 0.0, None, 0
    best_epoch = 0

    for epoch in range(200):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            alphas = model(xb)
            loss = dirichlet_loss(alphas, yb, epoch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                alphas = model(xb)
                v_correct += ((alphas / alphas.sum(1, keepdim=True)).argmax(1) == yb).sum().item()
                v_total += len(yb)

        va = v_correct / v_total
        scheduler.step(va)

        if va > best_acc:
            best_acc = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience = 0
        else:
            patience += 1

        if patience >= 25:
            break

    model.load_state_dict(best_state)
    model.to(device).eval()

    return model, best_acc, best_epoch, n_params


# ═══════════════════════════════════════════════════════════════════
# Per-sequence evaluation
# ═══════════════════════════════════════════════════════════════════

def per_seq_accuracy(y_true, y_pred, seq_ids):
    seq_data = defaultdict(lambda: {"t": [], "p": []})
    for i, (sn, _) in enumerate(seq_ids):
        seq_data[sn]["t"].append(y_true[i])
        seq_data[sn]["p"].append(y_pred[i])

    accs = {}
    for sn, d in seq_data.items():
        yt, yp = np.array(d["t"]), np.array(d["p"])
        mixed = len(np.unique(yt)) > 1
        accs[sn] = {"acc": accuracy_score(yt, yp), "mixed": mixed, "n": len(yt)}
    return accs


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load ──
    print("Loading data ...", flush=True)
    df = pd.read_csv(FEATURES_CSV)
    for c in SELECTED_FEATURES:
        if "temporal" in c or c == "motion_pixel_ratio":
            df[c] = df[c].fillna(0.0)

    actual_val = [s for s in VAL_SEQUENCES if s in set(df["sequence"].unique())]
    train_df = df[~df["sequence"].isin(actual_val)].copy()
    val_df = df[df["sequence"].isin(actual_val)].copy()

    print(f"  {len(df):,} frames, {df['sequence'].nunique()} seqs")
    print(f"  Train: {train_df['sequence'].nunique()} seqs, "
          f"Val: {val_df['sequence'].nunique()} seqs")
    print(f"  Features: {len(SELECTED_FEATURES)}")

    # ── Global threshold (median across all non-overlapping windows) ──
    # Use the 1.0s window config for threshold computation
    all_targets = []
    for _, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        for t in range(0, len(sdf) - 60 + 1, 30):
            all_targets.append(sdf[TARGET].iloc[t+30:t+60].mean())
    threshold = float(np.median(all_targets))
    print(f"  Threshold (median fn_nano): {threshold:.3f}\n")

    # ── Scaler (fit on train) ──
    scaler = StandardScaler()
    scaler.fit(train_df[SELECTED_FEATURES].values)

    # ═══════════════════════════════════════════════════════════════
    # SWEEP
    # ═══════════════════════════════════════════════════════════════
    results = []  # list of dicts

    for wcfg in WINDOW_CONFIGS:
        window = wcfg["window"]
        horizon = wcfg["horizon"]
        train_stride = wcfg["train_stride"]
        eval_stride = wcfg["eval_stride"]
        wlabel = wcfg["label"]

        print(f"{'=' * 70}")
        print(f"WINDOW: {wlabel} (w={window}, h={horizon}, "
              f"train_stride={train_stride}, eval_stride={eval_stride})")
        print(f"{'=' * 70}")

        # Build data for this window config
        X_train, y_train, train_ids, train_meta = build_windows(
            train_df, SELECTED_FEATURES, TARGET, window, horizon,
            train_stride, scaler, threshold)
        X_val, y_val, val_ids, val_meta = build_windows(
            val_df, SELECTED_FEATURES, TARGET, window, horizon,
            eval_stride, scaler, threshold)

        print(f"  Train: {len(X_train):,} ({y_train.mean():.1%} hard)")
        print(f"  Val:   {len(X_val):,} ({y_val.mean():.1%} hard)")
        majority_acc = max(y_val.mean(), 1 - y_val.mean())

        X_train_rf = build_rf_features(X_train)
        X_val_rf = build_rf_features(X_val)

        # ── Medium-sampling baseline ──
        train_gap = train_meta["tp_medium_last"] - train_meta["tp_nano_last"]
        val_gap = val_meta["tp_medium_last"] - val_meta["tp_nano_last"]
        best_mg_acc, best_mg_thr = 0, 0
        for thr in range(-2, 8):
            preds = (train_gap >= thr).astype(int)
            acc = accuracy_score(y_train, preds)
            if acc > best_mg_acc:
                best_mg_acc = acc
                best_mg_thr = thr
        med_pred = (val_gap >= best_mg_thr).astype(int)
        med_acc = accuracy_score(y_val, med_pred)
        med_f1 = f1_score(y_val, med_pred, zero_division=0)

        cm_med = confusion_matrix(y_val, med_pred)
        tn, fp, fn_c, tp = cm_med.ravel()
        med_miss = fn_c / (fn_c + tp) if (fn_c + tp) > 0 else 0

        results.append({
            "window": wlabel, "model": "MedSample",
            "params": 0, "best_epoch": "-",
            "val_acc": med_acc, "val_f1": med_f1,
            "miss_rate": med_miss, "auc": 0.0,
            "majority_acc": majority_acc,
        })

        # ── RF ──
        print(f"\n  RF ...", end="", flush=True)
        rf = RandomForestClassifier(n_estimators=500, max_depth=20,
                                     min_samples_leaf=10, max_features="sqrt",
                                     n_jobs=-1, random_state=42)
        rf.fit(X_train_rf, y_train)
        rf_pred = rf.predict(X_val_rf)
        rf_prob = rf.predict_proba(X_val_rf)[:, 1]
        rf_acc = accuracy_score(y_val, rf_pred)
        rf_f1 = f1_score(y_val, rf_pred, zero_division=0)
        rf_auc = roc_auc_score(y_val, rf_prob)

        cm_rf = confusion_matrix(y_val, rf_pred)
        tn, fp, fn_c, tp = cm_rf.ravel()
        rf_miss = fn_c / (fn_c + tp) if (fn_c + tp) > 0 else 0

        # Per-seq
        sa = per_seq_accuracy(y_val, rf_pred, val_ids)
        mixed_a = [v["acc"] for v in sa.values() if v["mixed"]]
        rf_mixed_med = float(np.median(mixed_a)) if mixed_a else 0.0

        print(f" Acc={rf_acc:.3f} F1={rf_f1:.3f} AUC={rf_auc:.3f} "
              f"miss={rf_miss:.3f} mixed_med={rf_mixed_med:.3f}")

        results.append({
            "window": wlabel, "model": "RF",
            "params": "-", "best_epoch": "-",
            "val_acc": rf_acc, "val_f1": rf_f1,
            "miss_rate": rf_miss, "auc": rf_auc,
            "majority_acc": majority_acc,
            "mixed_median": rf_mixed_med,
        })

        # ── Neural models ──
        for mcfg in MODEL_CONFIGS:
            mlabel = mcfg["label"]
            print(f"  {mlabel} ...", end="", flush=True)

            model, best_acc, best_epoch, n_params = train_model(
                X_train, y_train, X_val, y_val, mcfg,
                len(SELECTED_FEATURES), device)

            with torch.no_grad():
                pred, prob, unc = model.predict(
                    torch.from_numpy(X_val).to(device))
            pred = pred.cpu().numpy()
            prob = prob.cpu().numpy()
            unc = unc.cpu().numpy()

            acc = accuracy_score(y_val, pred)
            f1 = f1_score(y_val, pred, zero_division=0)
            auc = roc_auc_score(y_val, prob)

            cm = confusion_matrix(y_val, pred)
            tn, fp, fn_c, tp = cm.ravel()
            miss = fn_c / (fn_c + tp) if (fn_c + tp) > 0 else 0

            # Uncertainty calibration
            correct = (pred == y_val)
            if correct.sum() > 0 and (~correct).sum() > 0:
                unc_gap = unc[~correct].mean() - unc[correct].mean()
            else:
                unc_gap = 0.0

            # Per-seq
            sa = per_seq_accuracy(y_val, pred, val_ids)
            mixed_a = [v["acc"] for v in sa.values() if v["mixed"]]
            mixed_med = float(np.median(mixed_a)) if mixed_a else 0.0

            print(f" Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f} "
                  f"miss={miss:.3f} ep={best_epoch} params={n_params:,} "
                  f"unc_gap={unc_gap:+.3f} mixed_med={mixed_med:.3f}")

            results.append({
                "window": wlabel, "model": mlabel,
                "params": n_params, "best_epoch": best_epoch,
                "val_acc": acc, "val_f1": f1,
                "miss_rate": miss, "auc": auc,
                "majority_acc": majority_acc,
                "unc_gap": unc_gap,
                "mixed_median": mixed_med,
            })

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("SWEEP RESULTS SUMMARY")
    print(f"{'=' * 90}")
    print(f"\n  {'Window':<7s} {'Model':<15s} {'Params':>8s} {'Epoch':>6s} "
          f"{'Acc':>6s} {'F1':>6s} {'AUC':>6s} {'Miss%':>6s} {'MixMed':>7s}")
    print(f"  {'-'*72}")

    for r in results:
        params_str = f"{r['params']:,}" if isinstance(r['params'], int) else r['params']
        epoch_str = str(r.get('best_epoch', '-'))
        mix_str = f"{r.get('mixed_median', 0):.3f}" if 'mixed_median' in r else "  —  "
        print(f"  {r['window']:<7s} {r['model']:<15s} {params_str:>8s} {epoch_str:>6s} "
              f"{r['val_acc']:6.3f} {r['val_f1']:6.3f} {r.get('auc', 0):6.3f} "
              f"{r['miss_rate']*100:5.1f}% {mix_str}")

    # Find best config
    neural_results = [r for r in results if r["model"] not in ("RF", "MedSample")]
    if neural_results:
        best = max(neural_results, key=lambda r: r["val_acc"])
        print(f"\n  BEST NEURAL: {best['model']} @ {best['window']} "
              f"→ Acc={best['val_acc']:.3f} F1={best['val_f1']:.3f}")

    rf_results = [r for r in results if r["model"] == "RF"]
    if rf_results:
        best_rf = max(rf_results, key=lambda r: r["val_acc"])
        print(f"  BEST RF:     RF @ {best_rf['window']} "
              f"→ Acc={best_rf['val_acc']:.3f} F1={best_rf['val_f1']:.3f}")

    med_results = [r for r in results if r["model"] == "MedSample"]
    if med_results:
        best_med = max(med_results, key=lambda r: r["val_acc"])
        print(f"  MED SAMPLE:  @ {best_med['window']} "
              f"→ Acc={best_med['val_acc']:.3f} F1={best_med['val_f1']:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # PDF
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  Generating PDF ...", flush=True)

    with PdfPages(OUT_PDF) as pdf:

        # Page 1: Accuracy heatmap (model × window)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for ax_idx, metric in enumerate(["val_acc", "val_f1", "miss_rate"]):
            ax = axes[ax_idx]

            windows = [w["label"] for w in WINDOW_CONFIGS]
            models = [m["label"] for m in MODEL_CONFIGS] + ["RF", "MedSample"]

            grid = np.zeros((len(models), len(windows)))
            for r in results:
                wi = windows.index(r["window"])
                mi = models.index(r["model"])
                grid[mi, wi] = r[metric]

            if metric == "miss_rate":
                grid *= 100  # show as percentage
                cmap = "YlOrRd"
                title = "Miss Rate %"
            else:
                cmap = "YlGn"
                title = metric.replace("val_", "").upper()

            im = ax.imshow(grid, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(windows)))
            ax.set_xticklabels(windows)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models)
            ax.set_xlabel("Input Window")
            ax.set_title(title)

            for i in range(len(models)):
                for j in range(len(windows)):
                    val = grid[i, j]
                    fmt = f"{val:.1f}" if metric == "miss_rate" else f"{val:.3f}"
                    color = "white" if val > grid.mean() else "black"
                    ax.text(j, i, fmt, ha="center", va="center",
                            fontsize=9, color=color, fontweight="bold")

            plt.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle("Hyperparameter Sweep: Model × Window Size\n"
                     f"(DETRAC, 20 features, threshold={threshold:.2f})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig)
        plt.close()

        # Page 2: Line plots — accuracy vs window per model
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        windows = [w["label"] for w in WINDOW_CONFIGS]
        colors_map = {
            "LSTM-2L-64": "#1976D2", "LSTM-1L-32": "#42A5F5",
            "LSTM-1L-16": "#90CAF9", "GRU-1L-32": "#388E3C",
            "RF": "#F57C00", "MedSample": "#D32F2F",
        }

        for ax_idx, metric in enumerate(["val_acc", "val_f1"]):
            ax = axes[ax_idx]
            models_all = [m["label"] for m in MODEL_CONFIGS] + ["RF", "MedSample"]

            for model_name in models_all:
                model_results = [r for r in results if r["model"] == model_name]
                model_results.sort(key=lambda r: windows.index(r["window"]))
                vals = [r[metric] for r in model_results]
                color = colors_map.get(model_name, "gray")
                ls = "--" if model_name == "MedSample" else "-"
                marker = "s" if model_name == "RF" else "o"
                ax.plot(range(len(vals)), vals, f"{ls}", marker=marker,
                        color=color, label=model_name, linewidth=2, markersize=6)

            ax.set_xticks(range(len(windows)))
            ax.set_xticklabels(windows)
            ax.set_xlabel("Input Window")
            ax.set_ylabel(metric.replace("val_", "").upper())
            ax.set_title(f"{'Accuracy' if 'acc' in metric else 'F1'} vs Window Size")
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 3: Summary table
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis("off")

        header = ["Window", "Model", "Params", "Epoch", "Acc", "F1",
                  "AUC", "Miss%", "Mixed Med"]
        rows = []
        for r in results:
            params_str = f"{r['params']:,}" if isinstance(r['params'], int) else str(r['params'])
            epoch_str = str(r.get('best_epoch', '-'))
            mix_str = f"{r.get('mixed_median', 0):.3f}" if 'mixed_median' in r else "—"
            rows.append([
                r["window"], r["model"], params_str, epoch_str,
                f"{r['val_acc']:.3f}", f"{r['val_f1']:.3f}",
                f"{r.get('auc', 0):.3f}" if r.get('auc', 0) > 0 else "—",
                f"{r['miss_rate']*100:.1f}%", mix_str,
            ])

        table = ax.table(cellText=rows, colLabels=header,
                         loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.5)
        for j in range(len(header)):
            table[0, j].set_facecolor("#37474F")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Highlight best per window
        for i, r in enumerate(results):
            if r["model"] not in ("MedSample",):
                # Check if best for this window
                same_window = [x for x in results
                              if x["window"] == r["window"] and x["model"] != "MedSample"]
                best_acc = max(x["val_acc"] for x in same_window)
                if r["val_acc"] == best_acc:
                    for j in range(len(header)):
                        table[i + 1, j].set_facecolor("#E8F5E9")

        ax.set_title("Hyperparameter Sweep — Full Results",
                     fontsize=14, fontweight="bold", pad=30)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"  PDF: {OUT_PDF}")
    print("\nDone.")


if __name__ == "__main__":
    main()
