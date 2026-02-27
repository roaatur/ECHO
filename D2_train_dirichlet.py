#!/usr/bin/env python3
"""D²-City Step 6 — Dirichlet LSTM vs RF Classifier binary prediction.

Ports new_features/new_train_dirichlet.py to D²-City.

Task:  Given WINDOW=30 sampled frames (~3.6 s) of spatial features,
       classify whether mean(fn_nano) over the NEXT 30 sampled frames
       is >= median (hard) or < median (easy).

Models:
  1. Dirichlet LSTM — outputs Dirichlet concentration params (α_easy, α_hard),
     giving both a prediction AND calibrated uncertainty.
  2. Random Forest Classifier — baseline on window summary stats.

Baselines:
  - FAST threshold: classify via mean(fast_keypoints_half) over input window
    (uses best threshold fit on *train* data — fully deployable)
  - Logistic Regression on window-mean features (simplest deployable model)
  - Global Majority: predict whichever class is more common in train windows

D²-City adaptations vs DETRAC version:
  - No weather metadata → clip-based analysis only; no per-weather breakdown
  - Val split: last 20 clips alphabetically (same deterministic split as
    D2_train_models.py — keeps both experiments comparable)
  - Selected features loaded from D2_analysis_summary.csv at runtime;
    falls back to top-20 by |ρ| if CSV not found
  - FAST baseline gracefully skips if fast_keypoints_half not selected
  - Page 3 shows per-clip accuracy distribution instead of per-weather bars

All baselines are DEPLOYABLE (no ground truth at inference).

Output: D2_dirichlet_results.pdf, D2_dirichlet_scaler.pkl
"""

import pickle
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D2_common import SCRIPT_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_CSV = SCRIPT_DIR / "D2_all_features.csv"
ANALYSIS_CSV = SCRIPT_DIR / "D2_analysis_summary.csv"
OUT_PDF      = SCRIPT_DIR / "D2_dirichlet_results.pdf"
SCALER_PATH  = SCRIPT_DIR / "D2_dirichlet_scaler.pkl"

META_COLS = ["clip_id", "frame_idx", "fn_nano", "num_gt_coco",
             "tp_nano", "tp_medium", "recoverable_fn"]

# ── Task config ───────────────────────────────────────────────────────────────
WINDOW       = 30   # input: 30 sampled frames ≈ 3.6 s at 8.3 fps
HORIZON      = 30   # target: next 30 sampled frames
TRAIN_STRIDE = 15   # 50% overlap for training
EVAL_STRIDE  = 30   # non-overlapping for evaluation
TARGET       = "fn_nano"
N_CLASSES    = 2    # easy (0) / hard (1)
KL_ANNEALING_EPOCHS = 10   # ramp KL weight from 0 → 1 over this many epochs
N_FEATURES_FALLBACK = 20   # top-N if analysis CSV not available
N_VAL_CLIPS  = 20   # last 20 clips alphabetically → val set


# ── Feature selection ─────────────────────────────────────────────────────────

def get_selected_features(df):
    """Top-20 features from D2_analysis_summary.csv, or top-20 by |ρ| fallback."""
    if not ANALYSIS_CSV.exists():
        feat_cols = [c for c in df.columns if c not in META_COLS]
        print(f"  [WARN] {ANALYSIS_CSV} not found — using first "
              f"{N_FEATURES_FALLBACK} feature columns as fallback")
        return feat_cols[:N_FEATURES_FALLBACK]
    summary = pd.read_csv(ANALYSIS_CSV)
    feats = summary["feature"].head(N_FEATURES_FALLBACK).tolist()
    print(f"  Selected {len(feats)} features from {ANALYSIS_CSV.name}")
    return feats


# ── Data pipeline ─────────────────────────────────────────────────────────────

def load_data():
    """Load CSV, fill temporal NaNs."""
    df = pd.read_csv(FEATURES_CSV)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    for c in feat_cols:
        if "temporal" in c or c == "motion_pixel_ratio":
            df[c] = df[c].fillna(0.0)
    print(f"  {len(df):,} frames, {df['clip_id'].nunique()} clips, "
          f"{len(feat_cols)} feature columns")
    return df


def build_windows(df, feature_cols, target_col, clip_ids, stride,
                  scaler=None, threshold=None):
    """Build shifted-label classification windows.

    Input:  features[t : t+WINDOW]                             → (WINDOW, n_feat)
    Target: int(mean(fn_nano[t+WINDOW : t+WINDOW+HORIZON]) >= threshold)
    Also returns the raw target mean for analysis.
    """
    total_need = WINDOW + HORIZON
    X_list, y_list, y_raw_list, seq_ids = [], [], [], []

    for clip_id in clip_ids:
        cdf = df[df["clip_id"] == clip_id].sort_values("frame_idx").reset_index(drop=True)
        n   = len(cdf)
        if n < total_need:
            continue

        feat_vals   = cdf[feature_cols].values.astype(np.float32)
        target_vals = cdf[target_col].values

        if scaler is not None:
            feat_vals = scaler.transform(feat_vals).astype(np.float32)

        for t in range(0, n - total_need + 1, stride):
            target_mean = target_vals[t + WINDOW : t + WINDOW + HORIZON].mean()
            X_list.append(feat_vals[t : t + WINDOW])
            y_list.append(int(target_mean >= threshold))
            y_raw_list.append(target_mean)
            seq_ids.append((clip_id, cdf["frame_idx"].iloc[t]))

    X      = np.array(X_list, dtype=np.float32)
    y      = np.array(y_list, dtype=np.int64)
    y_raw  = np.array(y_raw_list, dtype=np.float32)
    return X, y, y_raw, seq_ids


def build_rf_features(X_windows):
    """Window → flat: [mean, std, slope] per feature = 3 × n_feat."""
    n, w, f   = X_windows.shape
    t_c       = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
    t_var     = (t_c ** 2).sum()
    means     = X_windows.mean(axis=1)
    stds      = X_windows.std(axis=1)
    slopes    = np.einsum("ijk,j->ik", X_windows.astype(np.float64), t_c) / t_var
    return np.hstack([means, stds, slopes.astype(np.float32)])


# ── Dirichlet Loss (Sensoy et al. 2018) ──────────────────────────────────────

def dirichlet_loss(alphas, targets, epoch, n_classes=N_CLASSES,
                   annealing_step=KL_ANNEALING_EPOCHS):
    """Type-II Maximum Likelihood + annealed KL regularizer.

    alphas:  (batch, n_classes) — Dirichlet concentration params (>0)
    targets: (batch,) — integer class labels
    """
    y_oh = torch.zeros_like(alphas)
    y_oh.scatter_(1, targets.unsqueeze(1), 1.0)

    S = alphas.sum(dim=1, keepdim=True)

    # Type-II ML: E_Dir[-log p(y|θ)]
    loss_ml = (y_oh * (torch.digamma(S) - torch.digamma(alphas))).sum(dim=1)

    # KL: shrink evidence toward uniform prior on wrong classes
    alpha_tilde = y_oh + (1.0 - y_oh) * alphas
    beta        = torch.ones_like(alphas)   # uniform prior (unused explicitly)
    S_tilde     = alpha_tilde.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(S_tilde.squeeze(1))
        - torch.lgamma(torch.tensor(float(n_classes), device=alphas.device))
        - torch.lgamma(alpha_tilde).sum(dim=1)
        + ((alpha_tilde - 1.0) * (
            torch.digamma(alpha_tilde) - torch.digamma(S_tilde)
        )).sum(dim=1)
    )

    # Anneal KL weight 0 → 1 over annealing_step epochs
    lam = min(1.0, epoch / max(annealing_step, 1))
    return (loss_ml + lam * kl).mean()


# ── Dirichlet LSTM ────────────────────────────────────────────────────────────

class DirichletLSTM(nn.Module):
    """2-layer LSTM → Dirichlet concentration params for binary classification."""

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
            nn.Softplus(),        # ensure output > 0
        )
        self.n_classes = n_classes

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        hidden = h_n[-1]          # last LSTM layer hidden state
        alphas = self.head(hidden) + 1.0   # shift → α ≥ 1 (proper Dirichlet)
        return alphas

    def predict_with_uncertainty(self, x):
        """Returns (pred_class, pred_prob, uncertainty, alphas)."""
        alphas  = self.forward(x)
        S       = alphas.sum(dim=1, keepdim=True)
        probs   = alphas / S
        uncertainty = self.n_classes / S.squeeze(1)   # ∈ (0, 1]
        pred_class  = probs.argmax(dim=1)
        pred_prob   = probs.max(dim=1).values
        return pred_class, pred_prob, uncertainty, alphas


def train_dirichlet_lstm(X_train, y_train, X_val, y_val, n_features, device):
    """Train Dirichlet LSTM with early stopping on val accuracy."""
    print("\n" + "=" * 60)
    print("TRAINING DIRICHLET LSTM")
    print("=" * 60)

    model = DirichletLSTM(input_size=n_features, hidden_size=64, dropout=0.2)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: LSTM(2-layer, h=64) → MLP(64→32→{N_CLASSES}) → Softplus+1")
    print(f"  Parameters:   {n_params:,}")
    print(f"  Train: {len(X_train):,} windows   Val: {len(X_val):,} windows")
    print(f"  Class balance — Train: {y_train.mean():.1%} hard  "
          f"Val: {y_val.mean():.1%} hard")

    train_ds     = TensorDataset(torch.from_numpy(X_train),
                                  torch.from_numpy(y_train))
    val_ds       = TensorDataset(torch.from_numpy(X_val),
                                  torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=128)

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6)

    best_val_acc    = 0.0
    best_state      = None
    patience_counter = 0
    patience_limit  = 25
    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}

    for epoch in range(200):
        # ── Train ──
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            alphas = model(xb)
            loss   = dirichlet_loss(alphas, yb, epoch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(yb)
            preds   = (alphas / alphas.sum(dim=1, keepdim=True)).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += len(yb)

        train_loss = epoch_loss / total
        train_acc  = correct / total

        # ── Validate ──
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                alphas = model(xb)
                loss   = dirichlet_loss(alphas, yb, epoch)
                val_loss_sum += loss.item() * len(yb)
                preds    = (alphas / alphas.sum(dim=1, keepdim=True)).argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total   += len(yb)

        val_loss = val_loss_sum / val_total
        val_acc  = val_correct  / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_state       = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            lr = optimizer.param_groups[0]["lr"]
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


# ── Baselines ─────────────────────────────────────────────────────────────────

def fast_threshold_baseline(X_windows, feature_cols, y_true, train_X=None,
                             train_y=None):
    """FAST keypoint threshold: classify via mean(fast_keypoints_half) per window.

    Threshold is fit on *train* data (deployable).  If fast_keypoints_half is
    not in feature_cols, falls back to the feature with the highest mean
    variance (a crude proxy).  Returns (preds, threshold_value, feat_used).
    """
    feat_name = "fast_keypoints_half"
    if feat_name in feature_cols:
        feat_idx = feature_cols.index(feat_name)
    else:
        # Fallback: highest-variance feature in the train set
        if train_X is not None:
            feat_idx = int(train_X[:, :, :].std(axis=(0, 1)).argmax())
        else:
            feat_idx = 0
        feat_name = feature_cols[feat_idx]
        print(f"  [WARN] fast_keypoints_half not in selected features — "
              f"FAST baseline uses '{feat_name}' instead")

    # Fit threshold on train windows (or val if no train provided)
    fit_X = train_X if train_X is not None else X_windows
    fit_y = train_y if train_y is not None else y_true

    feat_means = fit_X[:, :, feat_idx].mean(axis=1)
    best_acc, best_thr = 0.0, float(feat_means.median() if hasattr(feat_means, "median") else np.median(feat_means))
    for q in np.arange(0.05, 0.95, 0.02):
        thr   = np.quantile(feat_means, q)
        preds = (feat_means >= thr).astype(int)
        acc   = accuracy_score(fit_y, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    # Apply to val
    val_means = X_windows[:, :, feat_idx].mean(axis=1)
    preds     = (val_means >= best_thr).astype(int)
    return preds, best_thr, feat_name


def logistic_regression_baseline(X_train_rf, y_train, X_val_rf):
    """Logistic regression on window summary stats."""
    lr = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    lr.fit(X_train_rf, y_train)
    return lr.predict(X_val_rf), lr.predict_proba(X_val_rf)[:, 1], lr


# ── Evaluation helpers ────────────────────────────────────────────────────────

def eval_classifier(y_true, y_pred, y_prob=None, label=""):
    """Compute and print classification metrics; return dict."""
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if (y_prob is not None
                                               and len(np.unique(y_true)) > 1) else 0.0
    print(f"  {label:35s}  Acc={acc:.3f}  F1={f1:.3f}  "
          f"Prec={prec:.3f}  Rec={rec:.3f}  AUC={auc:.3f}")
    return {"acc": acc, "f1": f1, "prec": prec, "rec": rec, "auc": auc}


def per_clip_accuracy(y_true, y_pred, seq_ids):
    """Per-clip accuracy; returns dict clip_id → {acc, n, mixed, pct_hard}."""
    clip_data = defaultdict(lambda: {"true": [], "pred": []})
    for i, (cid, _) in enumerate(seq_ids):
        clip_data[cid]["true"].append(y_true[i])
        clip_data[cid]["pred"].append(y_pred[i])

    clip_accs = {}
    for cid, d in clip_data.items():
        yt = np.array(d["true"])
        yp = np.array(d["pred"])
        clip_accs[cid] = {
            "acc":      float(accuracy_score(yt, yp)),
            "f1":       float(f1_score(yt, yp, zero_division=0)),
            "n":        len(yt),
            "mixed":    len(np.unique(yt)) > 1,
            "pct_hard": float(yt.mean()),
        }
    return clip_accs


# ── PDF Report ────────────────────────────────────────────────────────────────

def generate_pdf(history, all_metrics, clip_accs_dict,
                 y_val, y_raw_val, val_ids,
                 lstm_uncertainty, rf_importances, threshold,
                 selected_features):

    with PdfPages(OUT_PDF) as pdf:

        # ── Page 1: Training curves + confusion matrices ──────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(history["train_loss"], "b-", alpha=0.7, label="Train loss")
        ax.plot(history["val_loss"],   "r-", alpha=0.7, label="Val loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Dirichlet Loss")
        ax.set_title("Dirichlet LSTM Training Curves"); ax.legend()

        ax = axes[0, 1]
        ax.plot(history["train_acc"], "b-", alpha=0.7, label="Train acc")
        ax.plot(history["val_acc"],   "r-", alpha=0.7, label="Val acc")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Curves"); ax.legend()
        ax.set_ylim(0.4, 1.0)

        for idx, model_name in enumerate(["Dirichlet LSTM", "RF Classifier"]):
            ax = axes[1, idx]
            m  = all_metrics[model_name]
            if "cm" in m:
                cm = m["cm"]
                ax.imshow(cm, cmap="Blues")
                ax.set_xticks([0, 1]); ax.set_xticklabels(["Easy", "Hard"])
                ax.set_yticks([0, 1]); ax.set_yticklabels(["Easy", "Hard"])
                for i in range(2):
                    for j in range(2):
                        color = "white" if cm[i, j] > cm.max() / 2 else "black"
                        ax.text(j, i, str(cm[i, j]), ha="center",
                                va="center", fontsize=14, color=color)
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                ax.set_title(f"{model_name}\nAcc={m['acc']:.3f}  F1={m['f1']:.3f}")

        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # ── Page 2: Model comparison bar chart ───────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        model_names = list(all_metrics.keys())
        accs  = [all_metrics[m]["acc"] for m in model_names]
        f1s   = [all_metrics[m]["f1"]  for m in model_names]
        colors = ["steelblue", "forestgreen", "#FF9800", "#9C27B0", "#607D8B"]

        for ax_idx, (vals, ylabel, title) in enumerate([
            (accs, "Accuracy",  "Classification Accuracy"),
            (f1s,  "F1 Score",  "F1 Score"),
        ]):
            ax = axes[ax_idx]
            x  = np.arange(len(model_names))
            bars = ax.bar(x, vals, color=colors[:len(model_names)])
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel(ylabel); ax.set_title(f"D²-City {title} Comparison")
            ax.axhline(0.5, color="k", ls="--", lw=0.8, label="Chance (0.5)")
            ax.legend(fontsize=8); ax.set_ylim(0.35, 1.0)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.3f}", ha="center", fontsize=9)

        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # ── Page 3: Per-clip accuracy distribution ────────────────────────────
        # Shows per-clip accuracy spread for LSTM and RF (no weather metadata)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        main_models = ["Dirichlet LSTM", "RF Classifier", "Logistic Regression"]
        clip_accs_data   = [[v["acc"] for v in clip_accs_dict[m].values()]
                            for m in main_models if m in clip_accs_dict]
        clip_accs_labels = [m for m in main_models if m in clip_accs_dict]

        ax = axes[0]
        if clip_accs_data:
            ax.boxplot(clip_accs_data, labels=clip_accs_labels, patch_artist=True,
                       boxprops=dict(facecolor="lightsteelblue"),
                       medianprops=dict(color="navy", linewidth=2))
        ax.axhline(0.5, color="r", ls="--", lw=0.8, label="Chance (0.5)")
        ax.set_ylabel("Per-Clip Accuracy")
        ax.set_title("Per-Clip Accuracy Distribution\n(D²-City, no weather labels)")
        ax.legend(fontsize=8)
        ax.set_ylim(0.0, 1.05)

        # Mixed vs uniform clips (Dirichlet LSTM)
        ax = axes[1]
        sa_lstm    = clip_accs_dict.get("Dirichlet LSTM", {})
        mixed_acc  = [v["acc"] for v in sa_lstm.values() if v["mixed"]]
        uniform_acc = [v["acc"] for v in sa_lstm.values() if not v["mixed"]]
        data_bp, labels_bp = [], []
        if mixed_acc:
            data_bp.append(mixed_acc)
            labels_bp.append(f"Mixed ({len(mixed_acc)} clips)")
        if uniform_acc:
            data_bp.append(uniform_acc)
            labels_bp.append(f"Uniform ({len(uniform_acc)} clips)")
        if data_bp:
            ax.boxplot(data_bp, labels=labels_bp, patch_artist=True,
                       boxprops=dict(facecolor="lightgreen"),
                       medianprops=dict(color="darkgreen", linewidth=2))
        ax.axhline(0.5, color="r", ls="--", lw=0.8, label="Chance (0.5)")
        ax.set_ylabel("Per-Clip Accuracy")
        ax.set_title("Dirichlet LSTM — Mixed vs Uniform Clips\n"
                     "(mixed = clip has both easy & hard windows)")
        ax.legend(fontsize=8); ax.set_ylim(0.0, 1.05)

        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # ── Page 4: Uncertainty analysis ─────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        unc     = lstm_uncertainty["uncertainty"]
        correct = lstm_uncertainty["correct"]

        ax = axes[0, 0]
        ax.hist(unc[correct],  bins=30, alpha=0.6, label="Correct",
                color="green", density=True)
        ax.hist(unc[~correct], bins=30, alpha=0.6, label="Incorrect",
                color="red",   density=True)
        ax.set_xlabel("Uncertainty (K/S)"); ax.set_ylabel("Density")
        ax.set_title("Uncertainty Distribution: Correct vs Incorrect")
        ax.legend()

        ax = axes[0, 1]
        n_bins     = 10
        bin_edges  = np.linspace(unc.min(), unc.max(), n_bins + 1)
        bin_accs, bin_centers = [], []
        for b in range(n_bins):
            mask = (unc >= bin_edges[b]) & (unc < bin_edges[b + 1])
            if mask.sum() > 0:
                bin_accs.append(correct[mask].mean())
                bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
        ax.bar(bin_centers, bin_accs,
               width=(bin_edges[1] - bin_edges[0]) * 0.8,
               color="steelblue", alpha=0.7)
        ax.set_xlabel("Uncertainty (K/S)"); ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Uncertainty\n"
                     "(low uncertainty should → high accuracy)")

        # Scatter: uncertainty vs raw fn_nano target
        ax = axes[1, 0]
        ax.scatter(y_raw_val, unc, alpha=0.3, s=5, color="steelblue")
        ax.set_xlabel("Raw fn_nano target (next window mean)")
        ax.set_ylabel("Uncertainty (K/S)")
        ax.set_title("Uncertainty vs Difficulty\n"
                     "(high fn_nano = harder scene)")
        ax.axvline(x=float(np.median(y_raw_val)), color="r", ls="--",
                   lw=0.8, label=f"Threshold={float(np.median(y_raw_val)):.2f}")
        ax.legend(fontsize=8)

        # RF feature importance top-15
        ax = axes[1, 1]
        top15 = rf_importances.head(15)
        y_pos = range(len(top15))
        ax.barh(y_pos, top15.values, color="forestgreen")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top15.index, fontsize=7)
        ax.set_xlabel("Importance")
        ax.set_title("RF Top-15 Feature Importances")
        ax.invert_yaxis()

        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # ── Page 5: Summary table ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis("off")

        header = ["Model", "Acc", "F1", "Prec", "Rec", "AUC", "Deployable?"]
        rows   = []
        for name, m in all_metrics.items():
            deployable = "No" if "oracle" in name.lower() else "Yes"
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

        sa_lstm = clip_accs_dict.get("Dirichlet LSTM", {})
        all_accs   = [v["acc"] for v in sa_lstm.values()]
        mixed_accs = [v["acc"] for v in sa_lstm.values() if v["mixed"]]
        per_clip_str = ""
        if all_accs:
            per_clip_str = (
                f"\nPer-clip accuracy (Dirichlet LSTM):\n"
                f"  All clips:    median={np.median(all_accs):.3f}  "
                f"mean={np.mean(all_accs):.3f}  n={len(all_accs)}\n"
            )
            if mixed_accs:
                per_clip_str += (
                    f"  Mixed clips:  median={np.median(mixed_accs):.3f}  "
                    f"mean={np.mean(mixed_accs):.3f}  n={len(mixed_accs)}  "
                    f"above 50%: {sum(1 for a in mixed_accs if a > 0.5)}"
                    f"/{len(mixed_accs)}\n"
                )

        config = (
            f"Dataset     : D²-City validation (dashcam, moving camera)\n"
            f"Task        : classify mean(fn_nano) over next {HORIZON} sampled frames\n"
            f"              as easy (<{threshold:.2f}) or hard (≥{threshold:.2f})\n"
            f"Threshold   : median fn_nano across all dataset windows = {threshold:.3f}\n"
            f"Input       : {WINDOW} sampled frames × {len(selected_features)} features "
            f"≈ {WINDOW * 3 / 25:.1f} s at 8.3 fps\n"
            f"Train stride: {TRAIN_STRIDE}   Eval stride: {EVAL_STRIDE}\n"
            f"Split       : last {N_VAL_CLIPS} clips alphabetically → val; "
            f"remainder → train\n"
            f"No num_gt filter, no per-clip normalisation, no look-ahead bias\n"
            f"{per_clip_str}"
        )
        ax.text(0.5, 0.28, config, transform=ax.transAxes,
                ha="center", va="top", fontfamily="monospace", fontsize=9)
        ax.set_title("D²-City Dirichlet LSTM vs Baselines — Binary Classification",
                     fontsize=13, fontweight="bold", pad=40)
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

    print(f"  PDF: {OUT_PDF}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("=" * 70)
    print("D²-CITY DIRICHLET LSTM: BINARY HARD/EASY CLASSIFICATION")
    print("=" * 70)

    if not FEATURES_CSV.exists():
        print(f"\n  ERROR: {FEATURES_CSV} not found. "
              f"Run D2_extract_features.py first.")
        return

    # ── Load ──
    df = load_data()
    selected_features = get_selected_features(df)

    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        raise ValueError(f"Features not in CSV: {missing}")

    # ── Val/Train split (last N_VAL_CLIPS alphabetically) ──
    all_clips  = sorted(df["clip_id"].unique())
    val_clips  = all_clips[-N_VAL_CLIPS:]
    train_clips = [c for c in all_clips if c not in set(val_clips)]

    train_df = df[df["clip_id"].isin(train_clips)].copy()
    val_df   = df[df["clip_id"].isin(val_clips)].copy()
    print(f"\n  Train: {len(train_clips)} clips, {len(train_df):,} frames")
    print(f"  Val:   {len(val_clips)} clips, {len(val_df):,} frames")

    # ── Threshold: median of all-dataset window targets ──
    all_window_targets = []
    for cid in all_clips:
        cdf = df[df["clip_id"] == cid].sort_values("frame_idx").reset_index(drop=True)
        n   = len(cdf)
        for t in range(0, n - WINDOW - HORIZON + 1, EVAL_STRIDE):
            all_window_targets.append(
                cdf[TARGET].iloc[t + WINDOW : t + WINDOW + HORIZON].mean())
    threshold = float(np.median(all_window_targets))
    print(f"\n  Threshold (median fn_nano across all windows): {threshold:.3f}")

    # ── Scaler fit on train ──
    scaler = StandardScaler()
    scaler.fit(train_df[selected_features].values)
    with open(SCALER_PATH, "wb") as fh:
        pickle.dump(scaler, fh)

    # ── Build windows ──
    print("\n  Building windows ...", flush=True)
    X_train, y_train, y_raw_train, train_ids = build_windows(
        df, selected_features, TARGET, train_clips,
        TRAIN_STRIDE, scaler, threshold)
    X_val, y_val, y_raw_val, val_ids = build_windows(
        df, selected_features, TARGET, val_clips,
        EVAL_STRIDE, scaler, threshold)

    print(f"  Train: {X_train.shape[0]:,} windows ({y_train.mean():.1%} hard)")
    print(f"  Val:   {X_val.shape[0]:,} windows  ({y_val.mean():.1%} hard)")

    X_train_rf = build_rf_features(X_train)
    X_val_rf   = build_rf_features(X_val)

    # ══════════════════════════════════════════════════════════════════════
    # Train models
    # ══════════════════════════════════════════════════════════════════════

    # 1. Dirichlet LSTM
    lstm_model, history = train_dirichlet_lstm(
        X_train, y_train, X_val, y_val, len(selected_features), device)

    # 2. RF Classifier
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 60)
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_leaf=10,
        max_features="sqrt", n_jobs=-1, random_state=42)
    rf.fit(X_train_rf, y_train)

    feat_names = (
        [f"{f}_mean"  for f in selected_features] +
        [f"{f}_std"   for f in selected_features] +
        [f"{f}_slope" for f in selected_features]
    )
    rf_importances = (pd.Series(rf.feature_importances_, index=feat_names)
                      .sort_values(ascending=False))
    print("  Top 5 RF features:")
    for fn, imp in rf_importances.head(5).items():
        print(f"    {fn:<45s} {imp:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Evaluate all models
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EVALUATION (non-overlapping val windows)")
    print("=" * 60)

    all_metrics    = {}
    clip_accs_dict = {}

    # ── Dirichlet LSTM ──
    with torch.no_grad():
        pred_class, pred_prob, unc, alphas = lstm_model.predict_with_uncertainty(
            torch.from_numpy(X_val).to(device))
    lstm_preds     = pred_class.cpu().numpy()
    lstm_unc       = unc.cpu().numpy()
    lstm_prob_hard = (alphas[:, 1] / alphas.sum(dim=1)).cpu().numpy()

    m = eval_classifier(y_val, lstm_preds, lstm_prob_hard, "Dirichlet LSTM")
    m["cm"] = confusion_matrix(y_val, lstm_preds)
    all_metrics["Dirichlet LSTM"] = m
    clip_accs_dict["Dirichlet LSTM"] = per_clip_accuracy(y_val, lstm_preds, val_ids)

    lstm_correct    = (lstm_preds == y_val)
    lstm_uncertainty = {"uncertainty": lstm_unc, "correct": lstm_correct,
                        "probs": lstm_prob_hard}

    # ── RF Classifier ──
    rf_preds = rf.predict(X_val_rf)
    rf_probs = rf.predict_proba(X_val_rf)[:, 1]
    m = eval_classifier(y_val, rf_preds, rf_probs, "RF Classifier")
    m["cm"] = confusion_matrix(y_val, rf_preds)
    all_metrics["RF Classifier"] = m
    clip_accs_dict["RF Classifier"] = per_clip_accuracy(y_val, rf_preds, val_ids)

    # ── Logistic Regression ──
    lr_preds, lr_probs, _ = logistic_regression_baseline(
        X_train_rf, y_train, X_val_rf)
    m = eval_classifier(y_val, lr_preds, lr_probs, "Logistic Regression")
    m["cm"] = confusion_matrix(y_val, lr_preds)
    all_metrics["Logistic Regression"] = m
    clip_accs_dict["Logistic Regression"] = per_clip_accuracy(
        y_val, lr_preds, val_ids)

    # ── FAST threshold (train-fit, deployable) ──
    fast_preds, fast_thr, fast_feat = fast_threshold_baseline(
        X_val, selected_features, y_val,
        train_X=X_train, train_y=y_train)
    m = eval_classifier(y_val, fast_preds, None,
                        f"FAST Threshold ({fast_feat[:20]})")
    m["cm"]  = confusion_matrix(y_val, fast_preds)
    m["auc"] = 0.0
    all_metrics[f"FAST Threshold ({fast_feat[:20]})"] = m
    clip_accs_dict[f"FAST Threshold ({fast_feat[:20]})"] = per_clip_accuracy(
        y_val, fast_preds, val_ids)
    print(f"  FAST baseline threshold: {fast_thr:.4f} on '{fast_feat}'")

    # ── Global Majority (train-fit) ──
    # Compute majority class from ALL train windows; predict it for every val window
    # (D²-City has no weather → single global majority)
    train_majority = int(y_train.mean() >= 0.5)
    majority_preds = np.full(len(y_val), train_majority, dtype=np.int64)
    m = eval_classifier(y_val, majority_preds, None, "Global Majority (train)")
    m["cm"]  = confusion_matrix(y_val, majority_preds)
    m["auc"] = 0.0
    all_metrics["Global Majority (train)"] = m
    clip_accs_dict["Global Majority (train)"] = per_clip_accuracy(
        y_val, majority_preds, val_ids)
    print(f"  Global majority class: {'hard' if train_majority else 'easy'} "
          f"(train hard fraction = {y_train.mean():.1%})")

    # ── Per-clip detail (Dirichlet LSTM) ──
    sa_lstm  = clip_accs_dict["Dirichlet LSTM"]
    all_accs = [v["acc"] for v in sa_lstm.values()]
    mixed_a  = [v["acc"] for v in sa_lstm.values() if v["mixed"]]
    uni_a    = [v["acc"] for v in sa_lstm.values() if not v["mixed"]]
    print(f"\n  Per-clip accuracy (Dirichlet LSTM):")
    print(f"    All clips:     median={np.median(all_accs):.3f}  "
          f"mean={np.mean(all_accs):.3f}  n={len(all_accs)}")
    if mixed_a:
        print(f"    Mixed clips:   median={np.median(mixed_a):.3f}  "
              f"mean={np.mean(mixed_a):.3f}  "
              f"above 50%: {sum(1 for a in mixed_a if a > 0.5)}/{len(mixed_a)}")
    if uni_a:
        print(f"    Uniform clips: median={np.median(uni_a):.3f}  "
              f"mean={np.mean(uni_a):.3f}  n={len(uni_a)}")

    # ── Uncertainty calibration ──
    print(f"\n  Uncertainty calibration (Dirichlet LSTM):")
    unc_corr = lstm_unc[lstm_correct]
    unc_incorr = lstm_unc[~lstm_correct]
    if len(unc_corr) > 0 and len(unc_incorr) > 0:
        unc_gap = unc_incorr.mean() - unc_corr.mean()
        print(f"    Mean uncertainty correct:   {unc_corr.mean():.4f}")
        print(f"    Mean uncertainty incorrect: {unc_incorr.mean():.4f}")
        print(f"    Gap: {unc_gap:+.4f} "
              f"({'good' if unc_gap > 0 else 'BAD — incorrect should be higher'})")
    else:
        print("    (insufficient samples to compute gap)")

    # ── PDF ──
    print("\n  Generating PDF ...", flush=True)
    generate_pdf(
        history, all_metrics, clip_accs_dict,
        y_val, y_raw_val, val_ids,
        lstm_uncertainty, rf_importances, threshold,
        selected_features,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
