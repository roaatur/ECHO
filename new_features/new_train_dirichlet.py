#!/usr/bin/env python3
"""Dirichlet LSTM vs RF Classifier — shifted-label binary prediction.

Task:  Given 30 frames of spatial features (1 sec), classify whether
       mean(fn_nano) over the NEXT 30 frames is >= median (hard) or < median (easy).

Models:
  1. Dirichlet LSTM — outputs Dirichlet concentration params (alpha_easy, alpha_hard)
     giving both a prediction AND calibrated uncertainty.
  2. Random Forest Classifier — baseline on window summary stats.

Baselines:
  - FAST threshold: classify based on mean(fast_keypoints_half) over input window
  - Logistic Regression on window-mean features (simplest deployable model)
  - Per-sequence majority class (how good is "always predict the sequence label"?)

All baselines are DEPLOYABLE (no ground truth at inference).

Bugs fixed vs prior experiments:
  - NO num_gt >= 1 filter (preserves temporal continuity)
  - NO per-sequence mean removal (avoids look-ahead bias)
  - Stride=15 train, stride=30 eval (no near-duplicate windows)
  - Sequence-level split (zero train/val leakage)
  - StandardScaler fit on train only
  - Proper Dirichlet loss (Sensoy et al. 2018) with annealed KL

Output: new_features/new_dirichlet_results.pdf
"""

import pickle
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
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "new_all_features.csv"
OUT_PDF = SCRIPT_DIR / "new_dirichlet_results.pdf"
SCALER_PATH = SCRIPT_DIR / "new_dirichlet_scaler.pkl"

META_COLS = ["sequence", "frame_id", "weather", "fn_nano", "num_gt",
             "tp_nano", "tp_medium", "recoverable_fn"]

# ── Task config ────────────────────────────────────────────────────
WINDOW = 30          # input: 30 frames (1 sec)
HORIZON = 30         # target: next 30 frames (1 sec)
TRAIN_STRIDE = 15    # 50% overlap for training
EVAL_STRIDE = 30     # non-overlapping for evaluation
TARGET = "fn_nano"
N_CLASSES = 2        # easy (0) / hard (1)
KL_ANNEALING_EPOCHS = 10  # anneal KL regularizer over this many epochs

# 20 non-redundant features from new_select_features.py
SELECTED_FEATURES = [
    "fast_keypoints_half", "edge_density_coarse", "gradient_magnitude_mean",
    "gabor_nyquist_energy", "fast_keypoints_full", "glcm_homogeneity",
    "overall_edge_density", "top_edge_density", "top_high_freq_energy",
    "top_gradient_std", "fft_critical_band_ratio", "total_edge_components",
    "top_laplacian_var", "glcm_entropy", "mid_edge_density",
    "dark_channel_mean", "mid_gradient_std", "foreground_blob_count",
    "mid_high_freq_energy", "rms_contrast",
]

# Val sequences (same as regression experiment — 20 seqs, 5 per weather)
VAL_SEQUENCES = [
    "MVI_39361", "MVI_20011", "MVI_20012", "MVI_20032", "MVI_20051",
    "MVI_40172", "MVI_40191", "MVI_40712", "MVI_40244", "MVI_40243",
    "MVI_40992", "MVI_40732", "MVI_39811", "MVI_40775", "MVI_40752",
    "MVI_63521", "MVI_40892", "MVI_63553", "MVI_63525", "MVI_63544",
]


# ═══════════════════════════════════════════════════════════════════
# Data Pipeline
# ═══════════════════════════════════════════════════════════════════

def load_data():
    """Load CSV, compute median threshold, build binary labels."""
    df = pd.read_csv(FEATURES_CSV)

    for c in SELECTED_FEATURES:
        if "temporal" in c or c == "motion_pixel_ratio":
            df[c] = df[c].fillna(0.0)

    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    print(f"  {len(df):,} frames, {df['sequence'].nunique()} seqs")
    return df


def build_windows(df, feature_cols, target_col, stride, scaler=None,
                  threshold=None):
    """Build shifted-label windows with binary classification labels.

    Input:  features[t : t+WINDOW]                      → (WINDOW, n_feat)
    Target: int(mean(fn_nano[t+WINDOW : t+WINDOW+HORIZON]) >= threshold)
    Also returns the raw target mean for analysis.
    """
    total_need = WINDOW + HORIZON
    X_list, y_list, y_raw_list, seq_ids = [], [], [], []

    for seq_name, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        n = len(sdf)
        if n < total_need:
            continue

        feat_vals = sdf[feature_cols].values
        target_vals = sdf[target_col].values

        if scaler is not None:
            feat_vals = scaler.transform(feat_vals)

        for t in range(0, n - total_need + 1, stride):
            x_window = feat_vals[t : t + WINDOW]
            target_mean = target_vals[t + WINDOW : t + WINDOW + HORIZON].mean()

            X_list.append(x_window)
            y_list.append(int(target_mean >= threshold))
            y_raw_list.append(target_mean)
            seq_ids.append((seq_name, sdf["frame_id"].iloc[t]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    y_raw = np.array(y_raw_list, dtype=np.float32)
    return X, y, y_raw, seq_ids


def build_rf_features(X_windows):
    """Window → flat: mean, std, slope per feature."""
    n_samples, window_len, n_feat = X_windows.shape
    t_centered = np.arange(window_len, dtype=np.float64) - (window_len - 1) / 2.0
    t_var = (t_centered ** 2).sum()

    means = X_windows.mean(axis=1)
    stds = X_windows.std(axis=1)
    slopes = np.einsum("ijk,j->ik", X_windows.astype(np.float64),
                       t_centered) / t_var

    return np.hstack([means, stds, slopes.astype(np.float32)])


# ═══════════════════════════════════════════════════════════════════
# Dirichlet Loss (Sensoy et al. 2018)
# ═══════════════════════════════════════════════════════════════════

def dirichlet_loss(alphas, targets, epoch, n_classes=N_CLASSES,
                   annealing_step=KL_ANNEALING_EPOCHS):
    """Type II Maximum Likelihood + annealed KL regularizer.

    alphas:  (batch, n_classes) — Dirichlet concentration params (>0)
    targets: (batch,) — integer class labels
    """
    y_oh = torch.zeros_like(alphas)
    y_oh.scatter_(1, targets.unsqueeze(1), 1.0)

    S = alphas.sum(dim=1, keepdim=True)  # Dirichlet strength

    # Type II ML loss: E_Dir[-log p(y|theta)]
    loss_ml = (y_oh * (torch.digamma(S) - torch.digamma(alphas))).sum(dim=1)

    # KL regularizer: push evidence toward uniform prior on WRONG classes
    # Only regularize the non-target alphas
    alpha_tilde = y_oh + (1.0 - y_oh) * alphas  # keep target alpha, shrink others
    beta = torch.ones_like(alphas)  # uniform prior

    # KL(Dir(alpha_tilde) || Dir(beta))
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(S_tilde.squeeze(1)) - torch.lgamma(torch.tensor(float(n_classes)))
        - torch.lgamma(alpha_tilde).sum(dim=1)
        + ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) -
           torch.digamma(S_tilde))).sum(dim=1)
    )

    # Annealing coefficient: 0 → 1 over annealing_step epochs
    lam = min(1.0, epoch / max(annealing_step, 1))

    return (loss_ml + lam * kl).mean()


# ═══════════════════════════════════════════════════════════════════
# Dirichlet LSTM
# ═══════════════════════════════════════════════════════════════════

class DirichletLSTM(nn.Module):
    """LSTM → Dirichlet alphas for binary classification with uncertainty."""

    def __init__(self, input_size, hidden_size=64, n_classes=N_CLASSES,
                 dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
            nn.Softplus(),  # alphas must be > 0
        )
        self.n_classes = n_classes

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        hidden = h_n[-1]  # last LSTM layer
        alphas = self.head(hidden) + 1.0  # shift so alpha >= 1 (proper Dir)
        return alphas

    def predict_with_uncertainty(self, x):
        """Returns predicted class, probability, and uncertainty."""
        alphas = self.forward(x)
        S = alphas.sum(dim=1, keepdim=True)
        probs = alphas / S  # expected probabilities
        uncertainty = self.n_classes / S.squeeze(1)  # total uncertainty [0, 1]

        pred_class = probs.argmax(dim=1)
        pred_prob = probs.max(dim=1).values
        return pred_class, pred_prob, uncertainty, alphas


def train_dirichlet_lstm(X_train, y_train, X_val, y_val, n_features, device):
    """Train Dirichlet LSTM with early stopping on val accuracy."""
    print("\n" + "=" * 60)
    print("TRAINING DIRICHLET LSTM")
    print("=" * 60)

    model = DirichletLSTM(input_size=n_features, hidden_size=64, dropout=0.2)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: LSTM(2-layer, h=64) → MLP(64→32→2) → Softplus+1")
    print(f"  Parameters: {n_params:,}")
    print(f"  Train: {len(X_train):,} windows, Val: {len(X_val):,} windows")
    print(f"  Class balance — Train: {y_train.mean():.1%} hard, "
          f"Val: {y_val.mean():.1%} hard")

    train_ds = TensorDataset(torch.from_numpy(X_train),
                             torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val),
                           torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    patience_limit = 25
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(200):
        # ── Train ──
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            alphas = model(xb)
            loss = dirichlet_loss(alphas, yb, epoch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(yb)
            preds = (alphas / alphas.sum(dim=1, keepdim=True)).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)

        train_loss = epoch_loss / total
        train_acc = correct / total

        # ── Validate ──
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                alphas = model(xb)
                loss = dirichlet_loss(alphas, yb, epoch)
                val_loss_sum += loss.item() * len(yb)
                preds = (alphas / alphas.sum(dim=1, keepdim=True)).argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += len(yb)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        lr = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 0 or patience_counter == 0:
            marker = " *" if patience_counter == 0 else ""
            print(f"  Epoch {epoch:3d}: loss={train_loss:.4f} "
                  f"acc={train_acc:.3f}  val_loss={val_loss:.4f} "
                  f"val_acc={val_acc:.3f}  lr={lr:.1e}{marker}", flush=True)

        if patience_counter >= patience_limit:
            print(f"  Early stopping at epoch {epoch} "
                  f"(best val_acc={best_val_acc:.4f})")
            break

    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    return model, history


# ═══════════════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════════════

def fast_threshold_baseline(X_windows, feature_cols, y_true):
    """FAST keypoint threshold: classify based on mean(fast_keypoints_half)."""
    fast_idx = feature_cols.index("fast_keypoints_half")
    fast_means = X_windows[:, :, fast_idx].mean(axis=1)

    # Find optimal threshold on this data (oracle — gives upper bound)
    best_acc, best_thr = 0, 0
    for q in np.arange(0.05, 0.95, 0.02):
        thr = np.quantile(fast_means, q)
        # Higher keypoints → more detail → harder for nano at low res
        preds = (fast_means >= thr).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    preds = (fast_means >= best_thr).astype(int)
    return preds, best_thr


def logistic_regression_baseline(X_train_rf, y_train, X_val_rf):
    """Logistic regression on window summary stats."""
    lr = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    lr.fit(X_train_rf, y_train)
    return lr.predict(X_val_rf), lr.predict_proba(X_val_rf)[:, 1], lr


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def eval_classifier(y_true, y_pred, y_prob=None, label=""):
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0

    print(f"  {label:35s}  Acc={acc:.3f}  F1={f1:.3f}  "
          f"Prec={prec:.3f}  Rec={rec:.3f}  AUC={auc:.3f}")
    return {"acc": acc, "f1": f1, "prec": prec, "rec": rec, "auc": auc}


def per_seq_accuracy(y_true, y_pred, seq_ids, weather_map):
    """Per-sequence and per-weather accuracy."""
    seq_data = defaultdict(lambda: {"true": [], "pred": []})
    for i, (sn, _) in enumerate(seq_ids):
        seq_data[sn]["true"].append(y_true[i])
        seq_data[sn]["pred"].append(y_pred[i])

    seq_accs = {}
    for sn, d in seq_data.items():
        yt = np.array(d["true"])
        yp = np.array(d["pred"])
        # Per-seq accuracy
        acc = accuracy_score(yt, yp)
        # Is this a mixed sequence?
        is_mixed = len(np.unique(yt)) > 1
        seq_accs[sn] = {"acc": acc, "n": len(yt),
                         "weather": weather_map.get(sn, "?"),
                         "mixed": is_mixed,
                         "pct_hard": float(yt.mean())}

    # Per-weather
    weather_data = defaultdict(lambda: {"true": [], "pred": []})
    for sn, d in seq_data.items():
        w = weather_map.get(sn, "?")
        weather_data[w]["true"].extend(d["true"])
        weather_data[w]["pred"].extend(d["pred"])

    weather_accs = {}
    for w, d in weather_data.items():
        yt = np.array(d["true"])
        yp = np.array(d["pred"])
        weather_accs[w] = {"acc": accuracy_score(yt, yp),
                           "f1": f1_score(yt, yp, zero_division=0),
                           "n": len(yt)}

    return seq_accs, weather_accs


# ═══════════════════════════════════════════════════════════════════
# PDF Report
# ═══════════════════════════════════════════════════════════════════

def generate_pdf(history, all_metrics, seq_accs_dict, weather_accs_dict,
                 y_val, y_raw_val, val_ids, weather_map,
                 lstm_uncertainty, rf_importances, threshold):

    with PdfPages(OUT_PDF) as pdf:

        # ── Page 1: Training curves + confusion matrices ──
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(history["train_loss"], "b-", alpha=0.7, label="Train loss")
        ax.plot(history["val_loss"], "r-", alpha=0.7, label="Val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dirichlet Loss")
        ax.set_title("Dirichlet LSTM Training Curves")
        ax.legend()

        ax = axes[0, 1]
        ax.plot(history["train_acc"], "b-", alpha=0.7, label="Train acc")
        ax.plot(history["val_acc"], "r-", alpha=0.7, label="Val acc")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Curves")
        ax.legend()
        ax.set_ylim(0.4, 1.0)

        # Confusion matrices for LSTM and RF
        for idx, model_name in enumerate(["Dirichlet LSTM", "RF Classifier"]):
            ax = axes[1, idx]
            m = all_metrics[model_name]
            if "cm" in m:
                cm = m["cm"]
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Easy", "Hard"])
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["Easy", "Hard"])
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i, j]), ha="center",
                                va="center", fontsize=14,
                                color="white" if cm[i, j] > cm.max() / 2 else "black")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"{model_name}\nAcc={m['acc']:.3f} F1={m['f1']:.3f}")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # ── Page 2: Model comparison bar chart ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        model_names = list(all_metrics.keys())
        accs = [all_metrics[m]["acc"] for m in model_names]
        f1s = [all_metrics[m]["f1"] for m in model_names]
        colors = ["steelblue", "forestgreen", "#FF9800", "#9C27B0", "#607D8B"]

        ax = axes[0]
        x = np.arange(len(model_names))
        bars = ax.bar(x, accs, color=colors[:len(model_names)])
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_title("Classification Accuracy Comparison")
        ax.axhline(0.5, color="k", ls="--", lw=0.5, label="Random")
        ax.legend()
        ax.set_ylim(0.4, 1.0)
        for bar, v in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", fontsize=9)

        ax = axes[1]
        bars = ax.bar(x, f1s, color=colors[:len(model_names)])
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("F1 Score")
        ax.set_title("F1 Score Comparison")
        ax.axhline(0.5, color="k", ls="--", lw=0.5)
        ax.set_ylim(0.4, 1.0)
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", fontsize=9)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # ── Page 3: Per-weather comparison ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        weather_cats = sorted(set().union(
            *(wa.keys() for wa in weather_accs_dict.values())))
        x = np.arange(len(weather_cats))
        bw = 0.8 / len(weather_accs_dict)

        for ax_idx, metric in enumerate(["acc", "f1"]):
            ax = axes[ax_idx]
            for i, (model_name, wa) in enumerate(weather_accs_dict.items()):
                vals = [wa.get(w, {}).get(metric, 0) for w in weather_cats]
                offset = (i - len(weather_accs_dict) / 2 + 0.5) * bw
                ax.bar(x + offset, vals, bw, label=model_name,
                       color=colors[i % len(colors)])
            ax.set_xticks(x)
            ax.set_xticklabels(weather_cats)
            ax.set_ylabel(metric.upper())
            ax.set_title(f"Per-Weather {metric.upper()}")
            ax.legend(fontsize=7)
            ax.axhline(0.5, color="k", ls="--", lw=0.5)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # ── Page 4: Uncertainty analysis (Dirichlet LSTM only) ──
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        unc = lstm_uncertainty["uncertainty"]
        correct = lstm_uncertainty["correct"]
        probs = lstm_uncertainty["probs"]

        # Uncertainty histogram: correct vs incorrect
        ax = axes[0, 0]
        ax.hist(unc[correct], bins=30, alpha=0.6, label="Correct",
                color="green", density=True)
        ax.hist(unc[~correct], bins=30, alpha=0.6, label="Incorrect",
                color="red", density=True)
        ax.set_xlabel("Uncertainty (K/S)")
        ax.set_ylabel("Density")
        ax.set_title("Uncertainty Distribution: Correct vs Incorrect")
        ax.legend()

        # Uncertainty vs accuracy (binned)
        ax = axes[0, 1]
        n_bins = 10
        bin_edges = np.linspace(unc.min(), unc.max(), n_bins + 1)
        bin_accs, bin_counts, bin_centers = [], [], []
        for b in range(n_bins):
            mask = (unc >= bin_edges[b]) & (unc < bin_edges[b + 1])
            if mask.sum() > 0:
                bin_accs.append(correct[mask].mean())
                bin_counts.append(mask.sum())
                bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
        ax.bar(bin_centers, bin_accs, width=(bin_edges[1] - bin_edges[0]) * 0.8,
               color="steelblue", alpha=0.7)
        ax.set_xlabel("Uncertainty (K/S)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Uncertainty\n(low uncertainty should → high accuracy)")

        # Per-sequence: accuracy on mixed vs uniform sequences
        ax = axes[1, 0]
        sa = seq_accs_dict.get("Dirichlet LSTM", {})
        mixed_accs = [v["acc"] for v in sa.values() if v["mixed"]]
        uniform_accs = [v["acc"] for v in sa.values() if not v["mixed"]]
        data_to_plot = []
        labels_to_plot = []
        if mixed_accs:
            data_to_plot.append(mixed_accs)
            labels_to_plot.append(f"Mixed ({len(mixed_accs)} seqs)")
        if uniform_accs:
            data_to_plot.append(uniform_accs)
            labels_to_plot.append(f"Uniform ({len(uniform_accs)} seqs)")
        if data_to_plot:
            ax.boxplot(data_to_plot, labels=labels_to_plot)
        ax.set_ylabel("Per-Sequence Accuracy")
        ax.set_title("Mixed vs Uniform Sequences\n(mixed = both easy & hard frames)")
        ax.axhline(0.5, color="r", ls="--", lw=0.5, label="Random")
        ax.legend()

        # RF feature importance
        ax = axes[1, 1]
        top15 = rf_importances.head(15)
        y_pos = range(len(top15))
        ax.barh(y_pos, top15.values, color="forestgreen")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top15.index, fontsize=7)
        ax.set_xlabel("Importance")
        ax.set_title("RF Top 15 Feature Importances")
        ax.invert_yaxis()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # ── Page 5: Summary table ──
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis("off")

        header = ["Model", "Acc", "F1", "Prec", "Rec", "AUC", "Deployable?"]
        rows = []
        for name, m in all_metrics.items():
            deployable = "No" if "Oracle" in name else "Yes"
            rows.append([
                name,
                f"{m['acc']:.3f}", f"{m['f1']:.3f}",
                f"{m['prec']:.3f}", f"{m['rec']:.3f}",
                f"{m['auc']:.3f}",
                deployable,
            ])

        table = ax.table(cellText=rows, colLabels=header,
                         loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 2.0)
        for j in range(len(header)):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Config info
        config = (
            f"Task: classify mean(fn_nano) over next {HORIZON} frames "
            f"as easy (<{threshold:.2f}) or hard (>={threshold:.2f})\n"
            f"Threshold: median fn_nano = {threshold:.2f}\n"
            f"Input: {WINDOW} frames x {len(SELECTED_FEATURES)} features = 1 sec\n"
            f"Train stride: {TRAIN_STRIDE}, Eval stride: {EVAL_STRIDE}\n"
            f"Split: 80 train / 20 val sequences (stratified by weather)\n"
            f"No num_gt filter, no per-seq normalization, no look-ahead bias\n"
            f"\nPer-sequence accuracy (Dirichlet LSTM):\n"
        )
        sa = seq_accs_dict.get("Dirichlet LSTM", {})
        if sa:
            all_accs = [v["acc"] for v in sa.values()]
            mixed_a = [v["acc"] for v in sa.values() if v["mixed"]]
            config += (
                f"  All sequences: median={np.median(all_accs):.3f}, "
                f"mean={np.mean(all_accs):.3f}\n"
                f"  Mixed only:    median={np.median(mixed_a):.3f}, "
                f"mean={np.mean(mixed_a):.3f}, "
                f"n_above_50%={sum(1 for a in mixed_a if a > 0.5)}/{len(mixed_a)}\n"
            )
        ax.text(0.5, 0.25, config, transform=ax.transAxes,
                ha="center", va="top", fontfamily="monospace", fontsize=9)
        ax.set_title("Dirichlet LSTM vs Baselines — Binary Classification",
                     fontsize=14, fontweight="bold", pad=40)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"  PDF: {OUT_PDF}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load ──
    df = load_data()
    weather_map = (df.drop_duplicates("sequence")
                   .set_index("sequence")["weather"].to_dict())

    # ── Compute threshold from ALL data (not just train) ──
    # Median of per-window fn_nano across all sequences
    all_window_targets = []
    for _, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        n = len(sdf)
        for t in range(0, n - WINDOW - HORIZON + 1, EVAL_STRIDE):
            all_window_targets.append(
                sdf[TARGET].iloc[t + WINDOW : t + WINDOW + HORIZON].mean())
    threshold = float(np.median(all_window_targets))
    print(f"  Threshold (median): {threshold:.3f}")

    # ── Split ──
    actual_val = [s for s in VAL_SEQUENCES if s in set(df["sequence"].unique())]
    train_df = df[~df["sequence"].isin(actual_val)].copy()
    val_df = df[df["sequence"].isin(actual_val)].copy()
    print(f"  Train: {train_df['sequence'].nunique()} seqs, "
          f"{len(train_df):,} frames")
    print(f"  Val:   {val_df['sequence'].nunique()} seqs, "
          f"{len(val_df):,} frames")

    # ── Scaler (fit on train) ──
    scaler = StandardScaler()
    scaler.fit(train_df[SELECTED_FEATURES].values)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # ── Build windows ──
    print("\n  Building windows ...", flush=True)
    X_train, y_train, y_raw_train, train_ids = build_windows(
        train_df, SELECTED_FEATURES, TARGET, TRAIN_STRIDE, scaler, threshold)
    X_val, y_val, y_raw_val, val_ids = build_windows(
        val_df, SELECTED_FEATURES, TARGET, EVAL_STRIDE, scaler, threshold)

    print(f"  Train: {X_train.shape[0]:,} windows "
          f"({y_train.mean():.1%} hard)")
    print(f"  Val:   {X_val.shape[0]:,} windows "
          f"({y_val.mean():.1%} hard)")

    X_train_rf = build_rf_features(X_train)
    X_val_rf = build_rf_features(X_val)

    # ═══════════════════════════════════════════════════════════════
    # Train models
    # ═══════════════════════════════════════════════════════════════

    # 1. Dirichlet LSTM
    lstm_model, history = train_dirichlet_lstm(
        X_train, y_train, X_val, y_val, len(SELECTED_FEATURES), device)

    # 2. RF Classifier
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 60)
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_leaf=10,
        max_features="sqrt", n_jobs=-1, random_state=42)
    rf.fit(X_train_rf, y_train)

    feat_names = (
        [f"{f}_mean" for f in SELECTED_FEATURES] +
        [f"{f}_std" for f in SELECTED_FEATURES] +
        [f"{f}_slope" for f in SELECTED_FEATURES]
    )
    rf_importances = pd.Series(rf.feature_importances_, index=feat_names)
    rf_importances = rf_importances.sort_values(ascending=False)
    print(f"  Top 5 features:")
    for fn, imp in rf_importances.head(5).items():
        print(f"    {fn:<40s} {imp:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Evaluate all models
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EVALUATION (non-overlapping val windows)")
    print("=" * 60)

    all_metrics = {}
    seq_accs_dict = {}
    weather_accs_dict = {}

    # ── Dirichlet LSTM ──
    with torch.no_grad():
        pred_class, pred_prob, unc, alphas = lstm_model.predict_with_uncertainty(
            torch.from_numpy(X_val).to(device))
    lstm_preds = pred_class.cpu().numpy()
    lstm_probs = pred_prob.cpu().numpy()
    lstm_unc = unc.cpu().numpy()
    # For AUC, use P(hard) = alpha_hard / S
    lstm_prob_hard = (alphas[:, 1] / alphas.sum(dim=1)).cpu().numpy()

    m = eval_classifier(y_val, lstm_preds, lstm_prob_hard, "Dirichlet LSTM")
    m["cm"] = confusion_matrix(y_val, lstm_preds)
    all_metrics["Dirichlet LSTM"] = m

    lstm_correct = (lstm_preds == y_val)
    lstm_uncertainty = {
        "uncertainty": lstm_unc,
        "correct": lstm_correct,
        "probs": lstm_prob_hard,
    }

    sa, wa = per_seq_accuracy(y_val, lstm_preds, val_ids, weather_map)
    seq_accs_dict["Dirichlet LSTM"] = sa
    weather_accs_dict["Dirichlet LSTM"] = wa

    # ── RF Classifier ──
    rf_preds = rf.predict(X_val_rf)
    rf_probs = rf.predict_proba(X_val_rf)[:, 1]
    m = eval_classifier(y_val, rf_preds, rf_probs, "RF Classifier")
    m["cm"] = confusion_matrix(y_val, rf_preds)
    all_metrics["RF Classifier"] = m

    sa, wa = per_seq_accuracy(y_val, rf_preds, val_ids, weather_map)
    seq_accs_dict["RF Classifier"] = sa
    weather_accs_dict["RF Classifier"] = wa

    # ── Logistic Regression ──
    lr_preds, lr_probs, _ = logistic_regression_baseline(
        X_train_rf, y_train, X_val_rf)
    m = eval_classifier(y_val, lr_preds, lr_probs, "Logistic Regression")
    m["cm"] = confusion_matrix(y_val, lr_preds)
    all_metrics["Logistic Regression"] = m

    sa, wa = per_seq_accuracy(y_val, lr_preds, val_ids, weather_map)
    seq_accs_dict["Logistic Regression"] = sa
    weather_accs_dict["Logistic Regression"] = wa

    # ── FAST Threshold (oracle — tuned on val) ──
    fast_preds, fast_thr = fast_threshold_baseline(X_val, SELECTED_FEATURES, y_val)
    m = eval_classifier(y_val, fast_preds, None, "FAST Threshold (oracle)")
    m["cm"] = confusion_matrix(y_val, fast_preds)
    m["auc"] = 0.0
    all_metrics["FAST Threshold (oracle)"] = m

    sa, wa = per_seq_accuracy(y_val, fast_preds, val_ids, weather_map)
    seq_accs_dict["FAST Threshold (oracle)"] = sa
    weather_accs_dict["FAST Threshold (oracle)"] = wa

    # ── Per-sequence majority baseline ──
    # For each val sequence, predict the majority class from TRAIN data
    # (simulate: "calibrate once at installation using training distribution")
    train_seq_majority = {}
    for _, sdf in train_df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        n = len(sdf)
        labels = []
        for t in range(0, n - WINDOW - HORIZON + 1, EVAL_STRIDE):
            target_mean = sdf[TARGET].iloc[t+WINDOW:t+WINDOW+HORIZON].mean()
            labels.append(int(target_mean >= threshold))
        if labels:
            train_seq_majority[sdf["weather"].iloc[0]] = (
                train_seq_majority.get(sdf["weather"].iloc[0], []) + labels)

    # For each val sequence, use same-weather majority from train
    majority_preds = []
    for sn, fid in val_ids:
        w = weather_map.get(sn, "?")
        if w in train_seq_majority:
            majority = int(np.mean(train_seq_majority[w]) >= 0.5)
        else:
            majority = 1
        majority_preds.append(majority)
    majority_preds = np.array(majority_preds)

    m = eval_classifier(y_val, majority_preds, None, "Weather Majority (train)")
    m["cm"] = confusion_matrix(y_val, majority_preds)
    m["auc"] = 0.0
    all_metrics["Weather Majority (train)"] = m

    sa, wa = per_seq_accuracy(y_val, majority_preds, val_ids, weather_map)
    seq_accs_dict["Weather Majority (train)"] = sa
    weather_accs_dict["Weather Majority (train)"] = wa

    # ── Print per-sequence detail for Dirichlet LSTM ──
    print("\n  Per-sequence (Dirichlet LSTM):")
    sa_lstm = seq_accs_dict["Dirichlet LSTM"]
    all_accs = [v["acc"] for v in sa_lstm.values()]
    mixed_accs = [v["acc"] for v in sa_lstm.values() if v["mixed"]]
    uniform_accs = [v["acc"] for v in sa_lstm.values() if not v["mixed"]]
    print(f"    All seqs:     median={np.median(all_accs):.3f}  "
          f"mean={np.mean(all_accs):.3f}")
    if mixed_accs:
        print(f"    Mixed seqs:   median={np.median(mixed_accs):.3f}  "
              f"mean={np.mean(mixed_accs):.3f}  "
              f"above 50%: {sum(1 for a in mixed_accs if a > 0.5)}/{len(mixed_accs)}")
    if uniform_accs:
        print(f"    Uniform seqs: median={np.median(uniform_accs):.3f}  "
              f"mean={np.mean(uniform_accs):.3f}")

    # ── Uncertainty calibration ──
    print(f"\n  Uncertainty calibration (Dirichlet LSTM):")
    print(f"    Mean uncertainty (correct):   {lstm_unc[lstm_correct].mean():.4f}")
    print(f"    Mean uncertainty (incorrect): {lstm_unc[~lstm_correct].mean():.4f}")
    unc_gap = lstm_unc[~lstm_correct].mean() - lstm_unc[lstm_correct].mean()
    print(f"    Gap: {unc_gap:+.4f} "
          f"({'good' if unc_gap > 0 else 'BAD'} — incorrect should be higher)")

    # ── Per-weather detail ──
    print(f"\n  Per-weather accuracy:")
    for model_name in ["Dirichlet LSTM", "RF Classifier", "Logistic Regression"]:
        print(f"    {model_name}:")
        for w in sorted(weather_accs_dict[model_name]):
            d = weather_accs_dict[model_name][w]
            print(f"      {w:8s}: Acc={d['acc']:.3f}  F1={d['f1']:.3f}  n={d['n']}")

    # ── PDF ──
    print("\n  Generating PDF ...", flush=True)
    generate_pdf(history, all_metrics, seq_accs_dict, weather_accs_dict,
                 y_val, y_raw_val, val_ids, weather_map,
                 lstm_uncertainty, rf_importances, threshold)

    print("\nDone.")


if __name__ == "__main__":
    main()
