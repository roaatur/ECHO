#!/usr/bin/env python3
"""Dirichlet Evidential Scene Complexity Classification on DETRAC.

Three model variants, all with Dirichlet evidential heads:
  A. Small LSTM:  1 layer, hidden=32, dropout=0.5  (~14K params)
  B. MLP-mean:    mean-pool 30 frames → 15-dim → MLP(15→32→2)  (~600 params)
  C. MLP-stats:   mean+std+slope per feature → 45-dim → MLP(45→32→2)  (~1.5K params)

Usage:
    python dirichlet/dirichlet_train.py
"""

from pathlib import Path
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Paths ──────────────────────────────────────────────────────────
ECHO_ROOT = Path(__file__).resolve().parent.parent
FEAT_CSV = ECHO_ROOT / "spatial_experiments" / "spatial_experiments_all_features.csv"
DET_CSV = ECHO_ROOT / "res_experiment" / "detrac_results.csv"
OUT_DIR = Path(__file__).resolve().parent

PDF_PATH = OUT_DIR / "dirichlet_results.pdf"
CSV_PATH = OUT_DIR / "dirichlet_results.csv"
SCALER_PATH = OUT_DIR / "dirichlet_scaler.pkl"

# ── Constants ──────────────────────────────────────────────────────
FEATURE_COLS = [
    "top_laplacian_variance", "top_small_component_count", "top_gradient_magnitude_std",
    "mid_laplacian_variance", "mid_small_component_count", "mid_gradient_magnitude_std",
    "bot_laplacian_variance", "bot_small_component_count", "bot_gradient_magnitude_std",
    "global_total_small_components", "global_overall_edge_density",
    "global_FAST_keypoint_count", "global_mean_brightness",
    "global_dark_channel_mean", "global_temporal_diff_mean",
]

WINDOW_SIZE = 30
TRAIN_STRIDE = 15
NUM_CLASSES = 2
BATCH_SIZE = 64
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 20
ANNEALING_STEP = 10
SEED = 42
LR = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════
#  DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════

def load_and_merge():
    print("Loading data...", flush=True)
    feat_df = pd.read_csv(FEAT_CSV)
    det_df = pd.read_csv(DET_CSV)
    nano = det_df[det_df["model"] == "nano_320"][["sequence", "frame_id", "fn"]].copy()
    nano = nano.rename(columns={"fn": "fn_nano"})
    df = feat_df.merge(nano, on=["sequence", "frame_id"])
    print(f"  {len(df):,} frames, {df['sequence'].nunique()} sequences", flush=True)
    return df


def build_windows(df, sequences, stride):
    windows = []
    for seq in sequences:
        group = df[df["sequence"] == seq].sort_values("frame_id").reset_index(drop=True)
        start = 0
        w_idx = 0
        while start + WINDOW_SIZE <= len(group):
            chunk = group.iloc[start:start + WINDOW_SIZE]
            windows.append({
                "sequence": seq,
                "window_idx": w_idx,
                "features": chunk[FEATURE_COLS].values,    # (30, 15)
                "mean_fn_nano": chunk["fn_nano"].mean(),
            })
            start += stride
            w_idx += 1
    return windows


def split_sequences(df):
    unique_seqs = sorted(df["sequence"].unique())
    rng = np.random.RandomState(SEED)
    rng.shuffle(unique_seqs)
    n = len(unique_seqs)
    return (unique_seqs[:int(0.7 * n)],
            unique_seqs[int(0.7 * n):int(0.85 * n)],
            unique_seqs[int(0.85 * n):])


def normalize_features(train_windows, val_windows, test_windows):
    train_flat = np.vstack([w["features"] for w in train_windows])
    scaler = StandardScaler().fit(train_flat)
    for split in [train_windows, val_windows, test_windows]:
        for w in split:
            w["features"] = scaler.transform(w["features"])
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    return scaler


# ── Datasets ──────────────────────────────────────────────────────

class SeqDataset(Dataset):
    """For LSTM: returns (30, 15) sequences."""
    def __init__(self, windows):
        self.X = torch.FloatTensor(np.array([w["features"] for w in windows]))
        self.y = torch.LongTensor([w["label"] for w in windows])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MeanDataset(Dataset):
    """For MLP-mean: returns mean-pooled (15,) vector."""
    def __init__(self, windows):
        self.X = torch.FloatTensor(np.array([w["features"].mean(axis=0) for w in windows]))
        self.y = torch.LongTensor([w["label"] for w in windows])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StatsDataset(Dataset):
    """For MLP-stats: returns (mean, std, slope) × 15 = (45,) vector."""
    def __init__(self, windows):
        vecs = []
        for w in windows:
            feats = w["features"]  # (30, 15)
            mean = feats.mean(axis=0)                         # (15,)
            std = feats.std(axis=0)                           # (15,)
            # slope via linear regression: slope = cov(t, x) / var(t)
            t = np.arange(WINDOW_SIZE, dtype=np.float32)
            t_centered = t - t.mean()
            t_var = t_centered @ t_centered
            slope = (t_centered @ feats) / max(t_var, 1e-8)  # (15,)
            vecs.append(np.concatenate([mean, std, slope]))
        self.X = torch.FloatTensor(np.array(vecs))
        self.y = torch.LongTensor([w["label"] for w in windows])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════
#  MODELS — all output Dirichlet alpha (K=2)
# ═══════════════════════════════════════════════════════════════════

class DirichletSmallLSTM(nn.Module):
    """Variant A: 1-layer LSTM, hidden=32, dropout=0.5."""
    name = "A_SmallLSTM"

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=15, hidden_size=32,
                            num_layers=1, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(32, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1, :])
        return nn.functional.softplus(self.fc(h)) + 1


class DirichletMLPMean(nn.Module):
    """Variant B: MLP on mean-pooled features (15 → 32 → 2)."""
    name = "B_MLP_mean"

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x):
        return nn.functional.softplus(self.net(x)) + 1


class DirichletMLPStats(nn.Module):
    """Variant C: MLP on (mean, std, slope) × 15 = 45 → 32 → 2."""
    name = "C_MLP_stats"

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(45, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x):
        return nn.functional.softplus(self.net(x)) + 1


# ═══════════════════════════════════════════════════════════════════
#  DIRICHLET LOSS
# ═══════════════════════════════════════════════════════════════════

def dirichlet_loss(alpha, target, epoch):
    y_oh = torch.zeros_like(alpha).scatter_(1, target.unsqueeze(1), 1)
    S = alpha.sum(dim=1, keepdim=True)

    # Type II ML
    loss_ml = torch.sum(y_oh * (torch.digamma(S) - torch.digamma(alpha)), dim=1)

    # KL regularizer (annealed)
    lam = min(1.0, epoch / max(ANNEALING_STEP, 1))
    alpha_t = y_oh + (1 - y_oh) * (alpha - 1) + 1
    S_t = alpha_t.sum(dim=1, keepdim=True)
    kl = (torch.lgamma(S_t.squeeze(1))
          - torch.lgamma(torch.tensor(float(NUM_CLASSES), device=alpha.device))
          - torch.sum(torch.lgamma(alpha_t), dim=1)
          + torch.sum((alpha_t - 1) * (torch.digamma(alpha_t) - torch.digamma(S_t)), dim=1))

    return (loss_ml + lam * kl).mean()


# ═══════════════════════════════════════════════════════════════════
#  TRAIN / EVALUATE ONE MODEL
# ═══════════════════════════════════════════════════════════════════

def train_one(model, train_loader, val_loader, n_train, n_val, model_path):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

    best_val = float("inf")
    patience = 0
    hist = {"tl": [], "vl": [], "ta": [], "va": []}

    for ep in range(NUM_EPOCHS):
        # Train
        model.train()
        tl, tc = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            a = model(xb)
            loss = dirichlet_loss(a, yb, ep)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(yb)
            tc += ((a / a.sum(1, keepdim=True)).argmax(1) == yb).sum().item()

        # Val
        model.eval()
        vl, vc = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                a = model(xb)
                vl += dirichlet_loss(a, yb, ep).item() * len(yb)
                vc += ((a / a.sum(1, keepdim=True)).argmax(1) == yb).sum().item()

        tl /= n_train; vl /= n_val
        ta = tc / n_train; va = vc / n_val
        sched.step(vl)
        hist["tl"].append(tl); hist["vl"].append(vl)
        hist["ta"].append(ta); hist["va"].append(va)

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), model_path)
            patience = 0
        else:
            patience += 1

        if ep % 10 == 0 or patience >= EARLY_STOP_PATIENCE:
            print(f"    ep {ep:3d}  tl={tl:.4f} vl={vl:.4f}  ta={ta:.3f} va={va:.3f}", flush=True)

        if patience >= EARLY_STOP_PATIENCE:
            print(f"    Early stop at epoch {ep}", flush=True)
            break

    print(f"    Best val_loss={best_val:.4f}", flush=True)
    return hist


def evaluate_one(model, test_loader, test_windows, model_path):
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(DEVICE)
    model.eval()

    all_a, all_y = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_a.append(model(xb.to(DEVICE)).cpu().numpy())
            all_y.append(yb.numpy())

    alphas = np.concatenate(all_a)
    labels = np.concatenate(all_y)
    S = alphas.sum(axis=1)
    probs = alphas / S[:, None]
    preds = probs.argmax(axis=1)
    epi = NUM_CLASSES / S
    ale = -np.sum(probs * np.log(probs + 1e-8), axis=1)
    correct = preds == labels

    acc = accuracy_score(labels, preds)
    epi_c = epi[correct].mean() if correct.sum() > 0 else 0
    epi_w = epi[~correct].mean() if (~correct).sum() > 0 else 0

    # Per-sequence accuracy
    seq_names = [w["sequence"] for w in test_windows]
    seq_acc = {}
    for seq in set(seq_names):
        m = np.array([s == seq for s in seq_names])
        seq_acc[seq] = accuracy_score(labels[m], preds[m])

    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = 0.0

    return {
        "alphas": alphas, "labels": labels, "predictions": preds,
        "probs": probs, "S": S, "epistemic": epi, "aleatoric": ale,
        "correct": correct, "accuracy": acc, "auc": auc,
        "epi_correct": epi_c, "epi_wrong": epi_w,
        "seq_acc": seq_acc,
    }


def compute_baselines(train_windows, test_windows, labels):
    majority = max(labels.mean(), 1 - labels.mean())

    # FAST threshold baseline
    fi = FEATURE_COLS.index("global_FAST_keypoint_count")
    train_fast = np.array([w["features"][:, fi].mean() for w in train_windows])
    train_lab = np.array([w["label"] for w in train_windows])
    test_fast = np.array([w["features"][:, fi].mean() for w in test_windows])

    best_acc = 0
    for pct in range(10, 91, 5):
        t = np.percentile(train_fast, pct)
        acc = accuracy_score(labels, (test_fast >= t).astype(int))
        best_acc = max(best_acc, acc)

    return {"majority": majority, "fast_thresh": best_acc}


# ═══════════════════════════════════════════════════════════════════
#  PDF REPORT
# ═══════════════════════════════════════════════════════════════════

def generate_pdf(all_results, baselines, threshold, threshold_type,
                 train_windows, val_windows, test_windows,
                 train_seqs, val_seqs, test_seqs):
    """4-page PDF comparing all model variants."""

    model_names = list(all_results.keys())
    colors_map = {"A_SmallLSTM": "#e74c3c", "B_MLP_mean": "#3498db", "C_MLP_stats": "#2ecc71"}

    with PdfPages(str(PDF_PATH)) as pdf:

        # ── Page 1: Training curves + data summary ───────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (0,0) Loss curves
        ax = axes[0, 0]
        for name in model_names:
            h = all_results[name]["history"]
            c = colors_map.get(name, "gray")
            ax.plot(h["tl"], ls="-", color=c, alpha=0.4, lw=0.8)
            ax.plot(h["vl"], ls="-", color=c, lw=1.5, label=f"{name} (val)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Dirichlet Loss")
        ax.set_title("Loss Curves (solid=val, faint=train)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # (0,1) Accuracy curves
        ax = axes[0, 1]
        for name in model_names:
            h = all_results[name]["history"]
            c = colors_map.get(name, "gray")
            ax.plot(h["ta"], ls="-", color=c, alpha=0.4, lw=0.8)
            ax.plot(h["va"], ls="-", color=c, lw=1.5, label=f"{name} (val)")
        ax.axhline(baselines["majority"], color="gray", ls="--", lw=0.8,
                    label=f"Majority ({baselines['majority']:.3f})")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Curves (overfitting monitor)")
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
        ax.set_ylim(0.3, 1.0)

        # (1,0) Class balance
        ax = axes[1, 0]
        all_fn = [w["mean_fn_nano"] for w in train_windows + val_windows + test_windows]
        ax.hist(all_fn, bins=50, color="#95a5a6", alpha=0.7, edgecolor="white")
        ax.axvline(threshold, color="red", ls="--", lw=2,
                    label=f"Threshold={threshold:.2f} ({threshold_type})")
        ax.set_xlabel("Window mean fn_nano"); ax.set_ylabel("Count")
        ax.set_title("Label Distribution"); ax.legend()

        # (1,1) Data summary text
        ax = axes[1, 1]; ax.axis("off")
        n_te, n_te_h = sum(1 for w in train_windows if w["label"] == 0), sum(1 for w in train_windows if w["label"] == 1)
        txt = (
            f"Data Summary\n{'='*40}\n\n"
            f"Window: {WINDOW_SIZE} frames, train stride: {TRAIN_STRIDE}\n\n"
            f"{'Split':<8s} {'Seqs':>5s} {'Win':>6s} {'Easy':>6s} {'Hard':>6s}\n{'-'*34}\n"
            f"{'Train':<8s} {len(train_seqs):>5d} {len(train_windows):>6d} {n_te:>6d} {n_te_h:>6d}\n"
            f"{'Val':<8s} {len(val_seqs):>5d} {len(val_windows):>6d}\n"
            f"{'Test':<8s} {len(test_seqs):>5d} {len(test_windows):>6d}\n\n"
            f"Models:\n"
        )
        for name in model_names:
            r = all_results[name]
            txt += f"  {name}: {r['n_params']:,} params, acc={r['eval']['accuracy']:.3f}\n"
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace")

        fig.suptitle("Page 1: Training Overview (3 Models)", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150); plt.close(fig)

        # ── Page 2: Test results comparison ──────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (0,0) Accuracy comparison bar chart
        ax = axes[0, 0]
        names_all = ["majority", "fast_thresh"] + model_names
        accs_all = [baselines["majority"], baselines["fast_thresh"]] + \
                   [all_results[n]["eval"]["accuracy"] for n in model_names]
        bar_colors = ["#bdc3c7", "#bdc3c7"] + [colors_map.get(n, "gray") for n in model_names]
        bars = ax.barh(names_all, accs_all, color=bar_colors, alpha=0.8, height=0.6)
        for bar, acc in zip(bars, accs_all):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{acc:.3f}", va="center", fontsize=9)
        ax.set_xlabel("Test Accuracy"); ax.set_title("Accuracy Comparison")
        ax.set_xlim(0, 1)

        # (0,1) ROC curves
        ax = axes[0, 1]
        for name in model_names:
            ev = all_results[name]["eval"]
            try:
                fpr, tpr, _ = roc_curve(ev["labels"], ev["probs"][:, 1])
                ax.plot(fpr, tpr, color=colors_map.get(name, "gray"), lw=1.5,
                        label=f"{name} AUC={ev['auc']:.3f}")
            except ValueError:
                pass
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curves"); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # (1,0) Confusion matrices side by side
        ax = axes[1, 0]; ax.axis("off")
        cm_txt = "Confusion Matrices\n" + "=" * 50 + "\n\n"
        for name in model_names:
            ev = all_results[name]["eval"]
            cm = confusion_matrix(ev["labels"], ev["predictions"])
            cm_txt += f"{name}:\n"
            cm_txt += f"              Pred Easy  Pred Hard\n"
            cm_txt += f"  Actual Easy   {cm[0,0]:>5d}      {cm[0,1]:>5d}\n"
            cm_txt += f"  Actual Hard   {cm[1,0]:>5d}      {cm[1,1]:>5d}\n\n"
        ax.text(0.05, 0.95, cm_txt, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", fontfamily="monospace")

        # (1,1) Classification reports
        ax = axes[1, 1]; ax.axis("off")
        # Pick best model for detailed report
        best_name = max(model_names, key=lambda n: all_results[n]["eval"]["accuracy"])
        ev = all_results[best_name]["eval"]
        rpt = classification_report(ev["labels"], ev["predictions"],
                                     target_names=["easy", "hard"])
        ax.text(0.05, 0.95,
                f"Best: {best_name}\n{'='*40}\n\n{rpt}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                fontfamily="monospace")

        fig.suptitle("Page 2: Test Results", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150); plt.close(fig)

        # ── Page 3: Uncertainty quality (best model) ─────────────
        ev = all_results[best_name]["eval"]
        epi = ev["epistemic"]; ale = ev["aleatoric"]
        correct = ev["correct"]; labels = ev["labels"]
        p_hard = ev["probs"][:, 1]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (0,0) Epistemic: correct vs incorrect
        ax = axes[0, 0]
        if correct.sum() > 0:
            ax.hist(epi[correct], bins=30, alpha=0.6, color="#2ecc71",
                    label=f"Correct (n={correct.sum()})", density=True)
        if (~correct).sum() > 0:
            ax.hist(epi[~correct], bins=30, alpha=0.6, color="#e74c3c",
                    label=f"Incorrect (n={(~correct).sum()})", density=True)
        ax.set_xlabel("Epistemic (K/S)"); ax.set_ylabel("Density")
        ax.set_title(f"Epistemic Uncertainty ({best_name})"); ax.legend(fontsize=8)

        # (0,1) Aleatoric: correct vs incorrect
        ax = axes[0, 1]
        if correct.sum() > 0:
            ax.hist(ale[correct], bins=30, alpha=0.6, color="#2ecc71",
                    label="Correct", density=True)
        if (~correct).sum() > 0:
            ax.hist(ale[~correct], bins=30, alpha=0.6, color="#e74c3c",
                    label="Incorrect", density=True)
        ax.set_xlabel("Aleatoric (Entropy)"); ax.set_ylabel("Density")
        ax.set_title("Aleatoric Uncertainty"); ax.legend(fontsize=8)

        # (1,0) P(hard) vs epistemic
        ax = axes[1, 0]
        sc = np.where(correct, "#2ecc71", "#e74c3c")
        ax.scatter(p_hard, epi, c=sc, alpha=0.4, s=15, edgecolors="none")
        ax.set_xlabel("P(hard)"); ax.set_ylabel("Epistemic")
        ax.set_title("P(hard) vs Uncertainty (green=correct)")

        # (1,1) Uncertainty comparison across models
        ax = axes[1, 1]
        bar_data = []
        bar_labels_list = []
        bar_c = []
        for name in model_names:
            ev2 = all_results[name]["eval"]
            bar_labels_list.extend([f"{name}\ncorrect", f"{name}\nwrong"])
            bar_data.extend([ev2["epi_correct"], ev2["epi_wrong"]])
            bar_c.extend(["#2ecc71", "#e74c3c"])
        ax.bar(range(len(bar_data)), bar_data, color=bar_c, alpha=0.8)
        ax.set_xticks(range(len(bar_data)))
        ax.set_xticklabels(bar_labels_list, fontsize=7)
        ax.set_ylabel("Mean Epistemic Uncertainty")
        ax.set_title("Uncertainty: Correct vs Wrong (all models)")

        fig.suptitle("Page 3: Uncertainty Quality", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150); plt.close(fig)

        # ── Page 4: Calibration + per-seq + summary ──────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (0,0) Calibration (best model)
        ax = axes[0, 0]
        bins = np.linspace(0, 1, 11)
        bc, ba, bn_list = [], [], []
        for i in range(len(bins) - 1):
            m = (p_hard >= bins[i]) & (p_hard < bins[i + 1])
            if m.sum() > 0:
                bc.append(p_hard[m].mean())
                ba.append(labels[m].mean())
                bn_list.append(m.sum())
        if bc:
            ax.plot(bc, ba, "o-", color="#e74c3c", markersize=8, label="Model")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
            for x, y, n in zip(bc, ba, bn_list):
                ax.annotate(f"n={n}", (x, y), fontsize=6, ha="center", va="bottom")
        ax.set_xlabel("Predicted P(hard)"); ax.set_ylabel("Actual Fraction Hard")
        ax.set_title(f"Calibration ({best_name})"); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3); ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)

        # (0,1) Decision regions
        ax = axes[0, 1]
        ax.scatter(p_hard, epi, c=sc, alpha=0.4, s=15, edgecolors="none")
        ax.axvline(0.5, color="blue", ls="--", lw=1, label="P(hard)=0.5")
        epi_med = np.median(epi)
        ax.axhline(epi_med, color="orange", ls="--", lw=1,
                    label=f"Median unc={epi_med:.3f}")
        ax.set_xlabel("P(hard)"); ax.set_ylabel("Epistemic")
        ax.set_title("Decision Regions for MPC"); ax.legend(fontsize=7)

        # (1,0) Per-sequence accuracy (best model)
        ax = axes[1, 0]
        ev = all_results[best_name]["eval"]
        if ev["seq_acc"]:
            sorted_s = sorted(ev["seq_acc"].items(), key=lambda x: x[1])
            sn = [s[0] for s in sorted_s]
            sa = [s[1] for s in sorted_s]
            sc2 = ["#e74c3c" if a < 0.5 else "#f39c12" if a < 0.7 else "#2ecc71" for a in sa]
            ax.barh(range(len(sn)), sa, color=sc2, alpha=0.8, height=0.7)
            ax.set_yticks(range(len(sn)))
            ax.set_yticklabels(sn, fontsize=6)
            ax.set_xlabel("Accuracy"); ax.set_title(f"Per-Sequence ({best_name})")
            ax.axvline(0.5, color="gray", ls="--", lw=0.8)

        # (1,1) Summary
        ax = axes[1, 1]; ax.axis("off")
        summary = f"SUMMARY\n{'='*45}\n\n"
        summary += f"{'Model':<16s} {'Acc':>6s} {'AUC':>6s} {'Params':>8s} {'Unc?':>5s}\n"
        summary += "-" * 45 + "\n"
        for name in model_names:
            ev2 = all_results[name]["eval"]
            unc_works = "YES" if ev2["epi_wrong"] > ev2["epi_correct"] else "NO"
            summary += (f"{name:<16s} {ev2['accuracy']:>6.3f} {ev2['auc']:>6.3f} "
                       f"{all_results[name]['n_params']:>8,d} {unc_works:>5s}\n")
        summary += "-" * 45 + "\n"
        summary += f"{'majority':<16s} {baselines['majority']:>6.3f}\n"
        summary += f"{'fast_thresh':<16s} {baselines['fast_thresh']:>6.3f}\n\n"

        best_ev = all_results[best_name]["eval"]
        if best_ev["accuracy"] > baselines["fast_thresh"]:
            summary += f"VERDICT: {best_name} BEATS all baselines.\n"
        elif best_ev["accuracy"] > baselines["majority"] + 0.05:
            summary += f"VERDICT: {best_name} beats majority, close to FAST.\n"
        else:
            summary += f"VERDICT: Marginal — FAST threshold is competitive.\n"

        if best_ev["epi_wrong"] > best_ev["epi_correct"]:
            summary += "Uncertainty IS informative — MPC can use it.\n"
        else:
            summary += "Uncertainty NOT informative.\n"

        ax.text(0.05, 0.95, summary, transform=ax.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace")

        fig.suptitle("Page 4: Calibration & Summary", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150); plt.close(fig)

    print(f"  PDF saved -> {PDF_PATH}", flush=True)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)

    df = load_and_merge()
    train_seqs, val_seqs, test_seqs = split_sequences(df)
    print(f"  Split: {len(train_seqs)} / {len(val_seqs)} / {len(test_seqs)} sequences", flush=True)

    # Build windows
    train_windows = build_windows(df, train_seqs, stride=TRAIN_STRIDE)
    val_windows = build_windows(df, val_seqs, stride=WINDOW_SIZE)
    test_windows = build_windows(df, test_seqs, stride=WINDOW_SIZE)
    print(f"  Windows: {len(train_windows)} train / {len(val_windows)} val / "
          f"{len(test_windows)} test", flush=True)

    # Threshold
    all_train_fn = [w["mean_fn_nano"] for w in train_windows]
    median_thresh = np.median(all_train_fn)
    cost_thresh = 3.0
    med_bal = np.mean([1 if fn >= median_thresh else 0 for fn in all_train_fn])
    cost_bal = np.mean([1 if fn >= cost_thresh else 0 for fn in all_train_fn])
    print(f"  Threshold: median({median_thresh:.2f})={100*med_bal:.1f}% hard, "
          f"cost(3.0)={100*cost_bal:.1f}% hard", flush=True)

    if abs(med_bal - 0.5) <= abs(cost_bal - 0.5):
        threshold, threshold_type = median_thresh, "median"
    else:
        threshold, threshold_type = cost_thresh, "cost"
    print(f"  Using: {threshold_type} (threshold={threshold:.2f})\n", flush=True)

    for w in train_windows + val_windows + test_windows:
        w["label"] = 0 if w["mean_fn_nano"] < threshold else 1

    # Normalize
    scaler = normalize_features(train_windows, val_windows, test_windows)

    # ── Run all 3 models ──────────────────────────────────────────
    model_configs = [
        (DirichletSmallLSTM, SeqDataset,   "A: Small LSTM (1×32, dropout=0.5)"),
        (DirichletMLPMean,   MeanDataset,   "B: MLP on mean-pooled features"),
        (DirichletMLPStats,  StatsDataset,  "C: MLP on (mean, std, slope)"),
    ]

    all_results = {}

    for ModelClass, DatasetClass, desc in model_configs:
        model = ModelClass()
        name = model.name
        n_params = sum(p.numel() for p in model.parameters())
        model_path = OUT_DIR / f"dirichlet_best_{name}.pt"

        print(f"{'='*60}")
        print(f"  {desc}")
        print(f"  {n_params:,} parameters → {model_path.name}")
        print(f"{'='*60}", flush=True)

        train_ds = DatasetClass(train_windows)
        val_ds = DatasetClass(val_windows)
        test_ds = DatasetClass(test_windows)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        hist = train_one(model, train_loader, val_loader, len(train_ds), len(val_ds), model_path)

        model2 = ModelClass()
        ev = evaluate_one(model2, test_loader, test_windows, model_path)

        print(f"  Accuracy: {ev['accuracy']:.4f}  AUC: {ev['auc']:.3f}")
        print(f"  Epi correct: {ev['epi_correct']:.4f}  wrong: {ev['epi_wrong']:.4f}\n", flush=True)

        all_results[name] = {"history": hist, "eval": ev, "n_params": n_params}

    # Baselines
    first_name = list(all_results.keys())[0]
    baselines = compute_baselines(train_windows, test_windows,
                                  all_results[first_name]["eval"]["labels"])
    print(f"\nBaselines: majority={baselines['majority']:.3f}, "
          f"fast_thresh={baselines['fast_thresh']:.3f}", flush=True)

    # Save best model's predictions to CSV
    best_name = max(all_results.keys(), key=lambda n: all_results[n]["eval"]["accuracy"])
    ev = all_results[best_name]["eval"]
    rows = []
    for i, w in enumerate(test_windows):
        rows.append({
            "sequence": w["sequence"], "window_idx": w["window_idx"],
            "mean_fn_nano": w["mean_fn_nano"], "label": ev["labels"][i],
            "prediction": ev["predictions"][i],
            "alpha_0": ev["alphas"][i, 0], "alpha_1": ev["alphas"][i, 1],
            "p_easy": ev["probs"][i, 0], "p_hard": ev["probs"][i, 1],
            "epistemic": ev["epistemic"][i], "aleatoric": ev["aleatoric"][i],
            "correct": ev["correct"][i], "model": best_name,
        })
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    print(f"  CSV saved -> {CSV_PATH}", flush=True)

    # PDF
    print("\nGenerating PDF...", flush=True)
    generate_pdf(all_results, baselines, threshold, threshold_type,
                 train_windows, val_windows, test_windows,
                 train_seqs, val_seqs, test_seqs)

    print(f"\nDone in {time.time() - t0:.0f}s.", flush=True)


if __name__ == "__main__":
    main()
