#!/usr/bin/env python3
"""Evidential LSTM trained on 15 spatial features to predict miss count (fn).

Architecture identical to Chapter 4 / 2lstms experiments.
Input: 15 spatial features × 20-frame context → 10-frame fn horizon + 1-step num_gt.
Target: fn (miss count per frame from nano@320).

Output: spatial_experiments_lstm_results.pdf (4 pages)
"""

import pickle
from collections import defaultdict
from pathlib import Path

import lightning as L
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── Paths & constants ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

DATA_CSV         = SCRIPT_DIR / "spatial_experiments_all_features.csv"
SCALER_PATH      = SCRIPT_DIR / "spatial_experiments_lstm_scaler.pkl"
DENSITY_MAX_PATH = SCRIPT_DIR / "spatial_experiments_lstm_density_max.pkl"
CKPT_DIR         = SCRIPT_DIR
LOG_DIR          = SCRIPT_DIR / "spatial_experiments_lstm_logs"
PDF_PATH         = SCRIPT_DIR / "spatial_experiments_lstm_results.pdf"

T_CONTEXT = 20
T_HORIZON = 10
TARGET_COL = "fn"

# 15 spatial features (selected by spatial_experiments_feature_selection.py)
FEATURES = [
    "top_laplacian_variance", "top_small_component_count", "top_gradient_magnitude_std",
    "mid_laplacian_variance", "mid_small_component_count", "mid_gradient_magnitude_std",
    "bot_laplacian_variance", "bot_small_component_count", "bot_gradient_magnitude_std",
    "global_total_small_components", "global_overall_edge_density",
    "global_FAST_keypoint_count", "global_mean_brightness",
    "global_dark_channel_mean", "global_temporal_diff_mean",
]

# Excluded: MVI_39211 — degenerate sequence (extreme fn, dominates loss)
EXCLUDE_SEQUENCES = ["MVI_39211"]

# 14 val sequences (MVI_39211 excluded)
VAL_SEQUENCES = [
    # sunny
    "MVI_39361", "MVI_20011", "MVI_41073",
    # cloudy
    "MVI_40172", "MVI_40191", "MVI_40712", "MVI_40244",
    # night
    "MVI_40992", "MVI_40732", "MVI_39811", "MVI_40775",
    # rainy
    "MVI_63521", "MVI_40892", "MVI_63553",
]

PLOT_SEQUENCES = [
    "MVI_39361", "MVI_41073",     # sunny
    "MVI_40172", "MVI_40191",     # cloudy
    "MVI_40992", "MVI_40732",     # night
    "MVI_63521", "MVI_40892",     # rainy
]

# Previous experiment results for comparison (all targeted recall_gap or recall)
PREV_RESULTS = {
    "Image-only (14 feat)":    {"target": "recall_gap", "r2": -0.115, "mae": 0.144, "rho": 0.084},
    "Both models (32 feat)":   {"target": "recall_gap", "r2":  0.165, "mae": 0.123, "rho": 0.530},
    "LSTM_nano (23 feat)":     {"target": "recall_gap", "r2": -0.017, "mae": 0.137, "rho": 0.357},
    "LSTM_medium (23 feat)":   {"target": "recall_gap", "r2": -0.098, "mae": 0.144, "rho": 0.215},
    "Nano recall (23 feat)":   {"target": "nano_recall", "r2": -0.021, "mae": 0.191, "rho": 0.265},
}


# ── Evidential loss ───────────────────────────────────────────────
def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2 * beta * (1 + v)
    nll = (
        0.5 * torch.log(torch.tensor(np.pi, device=v.device, dtype=v.dtype) / v)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + twoBlambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return torch.mean(nll) if reduce else nll


def NIG_Reg(y, gamma, v, alpha, beta, reduce=True):
    error = torch.abs(y - gamma)
    evi = 2 * v + alpha
    reg = error * evi
    return torch.mean(reg) if reduce else reg


def EvidentialLoss(y_true, gamma, v, alpha, beta, coeff=1.0):
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * loss_reg


# ── Model ──────────────────────────────────────────────────────────
class MultiTaskLSTM(L.LightningModule):
    """Evidential LSTM: fn prediction (10-step) + num_gt auxiliary (1-step)."""

    def __init__(self, input_size=15, hidden_size=64, horizon=10, lr=1e-3,
                 coeff=0.1, lambda_density=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True)
        self.head_fn = nn.Linear(hidden_size, horizon * 4)    # evidential
        self.head_density = nn.Linear(hidden_size, 1)          # 1-step

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        hidden = h_n.squeeze(0)

        fn_out = self.head_fn(hidden).view(-1, self.hparams.horizon, 4)
        gamma = F.softplus(fn_out[..., 0])     # fn ≥ 0
        v = F.softplus(fn_out[..., 1])
        alpha = F.softplus(fn_out[..., 2]) + 1
        beta = F.softplus(fn_out[..., 3])

        density = F.softplus(self.head_density(hidden)).squeeze(-1)

        return gamma, v, alpha, beta, density

    def training_step(self, batch, batch_idx):
        x, y_fn, y_density = batch
        gamma, v, alpha, beta, density = self(x)

        loss_fn = EvidentialLoss(y_fn, gamma, v, alpha, beta,
                                 coeff=self.hparams.coeff)
        loss_density = F.mse_loss(density, y_density)
        loss = loss_fn + self.hparams.lambda_density * loss_density

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_fn_loss", loss_fn)
        self.log("train_density_loss", loss_density)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_fn, y_density = batch
        gamma, v, alpha, beta, density = self(x)

        loss_fn = EvidentialLoss(y_fn, gamma, v, alpha, beta,
                                 coeff=self.hparams.coeff)
        loss_density = F.mse_loss(density, y_density)
        loss = loss_fn + self.hparams.lambda_density * loss_density

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_fn_loss", loss_fn)
        self.log("val_density_loss", loss_density)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def get_uncertainty(self, v, alpha, beta):
        aleatoric = beta / (alpha - 1)
        epistemic = beta / (v * (alpha - 1))
        return aleatoric, epistemic


# ── Data pipeline ──────────────────────────────────────────────────
def _forward_smooth(s, window=10):
    """Forward-looking rolling mean: smoothed[k] = mean(raw[k : k+window])."""
    vals = s.values
    out = np.empty_like(vals, dtype=np.float64)
    n = len(vals)
    for k in range(n):
        end = min(k + window, n)
        out[k] = np.mean(vals[k:end])
    return pd.Series(out, index=s.index)


def create_dataset():
    """Load features CSV, smooth fn target, filter."""
    df = pd.read_csv(DATA_CSV)
    print(f"Data loaded: {len(df)} rows, {df['sequence'].nunique()} sequences")
    print(f"Features: {len(FEATURES)}")
    print(f"Weather: {df['weather'].value_counts().to_dict()}")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Exclude degenerate sequences
    df = df[~df["sequence"].isin(EXCLUDE_SEQUENCES)].reset_index(drop=True)

    # Forward-smooth fn target (10-frame lookahead)
    df[TARGET_COL] = df.groupby("sequence")[TARGET_COL].transform(
        lambda s: _forward_smooth(s, window=10)
    )

    df = df[df["num_gt"] >= 1]
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    print(f"After filtering: {len(df)} rows, {df['sequence'].nunique()} sequences")
    return df


def build_sliding_windows(df, feature_cols, target_col, density_col="num_gt"):
    """Build sliding windows with stride = T_CONTEXT."""
    X_list, y_fn_list, y_density_list, seq_ids = [], [], [], []
    total_window = T_CONTEXT + T_HORIZON

    for seq_name, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        features = sdf[feature_cols].values
        fn_targets = sdf[target_col].values
        density_targets = sdf[density_col].values

        n = len(sdf)
        for i in range(0, n - total_window + 1, T_CONTEXT):
            X_list.append(features[i : i + T_CONTEXT])
            y_fn_list.append(fn_targets[i + T_CONTEXT : i + T_CONTEXT + T_HORIZON])
            y_density_list.append(density_targets[i + T_CONTEXT])
            seq_ids.append((seq_name, i))

    X = np.array(X_list, dtype=np.float32)
    y_fn = np.array(y_fn_list, dtype=np.float32)
    y_density = np.array(y_density_list, dtype=np.float32)
    return X, y_fn, y_density, seq_ids


def compute_metrics(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    print(f"  {label:40s}  MAE={mae:.4f}  R2={r2:.4f}  rho={rho:.4f}")
    return {"mae": mae, "r2": r2, "rho": rho}


# ── Main ───────────────────────────────────────────────────────────
def main():
    df = create_dataset()

    # ── Val split ──
    train_mask = ~df["sequence"].isin(VAL_SEQUENCES)
    train_df = df[train_mask].copy()
    val_df = df[df["sequence"].isin(VAL_SEQUENCES)].copy()

    print(f"\nTrain: {train_df['sequence'].nunique()} sequences, {len(train_df)} frames")
    print(f"Val:   {val_df['sequence'].nunique()} sequences, {len(val_df)} frames")

    actual_val = set(val_df["sequence"].unique())
    expected_val = set(VAL_SEQUENCES)
    if actual_val != expected_val:
        print(f"WARNING: Missing val sequences: {expected_val - actual_val}")

    seq_weather = df.drop_duplicates("sequence").set_index("sequence")["weather"].to_dict()

    # ── Normalize features (fit on train only) ──
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES])
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved -> {SCALER_PATH}")

    df[FEATURES] = scaler.transform(df[FEATURES])

    # ── Normalize num_gt (fit on train only) ──
    density_max = float(train_df["num_gt"].max())
    with open(DENSITY_MAX_PATH, "wb") as f:
        pickle.dump(density_max, f)
    print(f"Density max = {density_max}")

    df["num_gt_norm"] = df["num_gt"] / density_max

    train_df = df[~df["sequence"].isin(VAL_SEQUENCES)].copy()
    val_df = df[df["sequence"].isin(VAL_SEQUENCES)].copy()

    # ── Sliding windows ──
    X_train, y_fn_train, y_den_train, train_ids = build_sliding_windows(
        train_df, FEATURES, TARGET_COL)
    X_val, y_fn_val, y_den_val, val_ids = build_sliding_windows(
        val_df, FEATURES, TARGET_COL)

    _, _, y_den_train_norm, _ = build_sliding_windows(
        train_df, FEATURES, TARGET_COL, density_col="num_gt_norm")
    _, _, y_den_val_norm, _ = build_sliding_windows(
        val_df, FEATURES, TARGET_COL, density_col="num_gt_norm")

    print(f"Windows -- train: {X_train.shape[0]}, val: {X_val.shape[0]}")
    print(f"X shape: {X_train.shape}, y_fn: {y_fn_train.shape}, "
          f"y_den: {y_den_train_norm.shape}")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_fn_train),
        torch.from_numpy(y_den_train_norm),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_fn_val),
        torch.from_numpy(y_den_val_norm),
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # ── Train ──
    model = MultiTaskLSTM(
        input_size=len(FEATURES),
        hidden_size=64,
        horizon=T_HORIZON,
        lr=1e-3,
        coeff=0.1,
        lambda_density=0.1,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, mode="min", verbose=True),
        ModelCheckpoint(
            dirpath=str(CKPT_DIR),
            filename="spatial_experiments_lstm_best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
        ),
    ]

    csv_logger = CSVLogger(save_dir=str(SCRIPT_DIR),
                           name="spatial_experiments_lstm_logs", version=0)

    trainer = L.Trainer(
        max_epochs=150,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        logger=csv_logger,
    )
    trainer.fit(model, train_loader, val_loader)

    best_path = callbacks[1].best_model_path
    if best_path:
        print(f"\nLoading best checkpoint: {best_path}")
        model = MultiTaskLSTM.load_from_checkpoint(best_path)
    print(f"Best val_loss: {callbacks[1].best_model_score:.5f}")

    # ── Evaluation ─────────────────────────────────────────────────
    device = next(model.parameters()).device
    model.eval()

    print("\n" + "=" * 70)
    print("QUANTITATIVE EVALUATION")
    print("=" * 70)

    all_fn_metrics = {}
    all_den_metrics = {}
    val_seq_data = {}

    for seq_name in VAL_SEQUENCES:
        indices = [i for i, (s, _) in enumerate(val_ids) if s == seq_name]
        if not indices:
            continue

        X_seq = torch.from_numpy(X_val[indices]).to(device)
        y_fn_seq = y_fn_val[indices]
        y_den_seq = y_den_val[indices]

        with torch.no_grad():
            gamma, v, alpha, beta, density = model(X_seq)
            fn_preds = gamma.cpu().numpy()
            den_preds_norm = density.cpu().numpy()
            aleatoric, epistemic = model.get_uncertainty(v, alpha, beta)
            epistemic = epistemic.cpu().numpy()

        den_preds = den_preds_norm * density_max

        weather = seq_weather.get(seq_name, "unknown")
        print(f"\n  --- {seq_name} ({weather}) ---")
        fn_m = compute_metrics(y_fn_seq.flatten(), fn_preds.flatten(),
                               label=f"{seq_name} fn (miss count)")
        den_m = compute_metrics(y_den_seq.flatten(), den_preds.flatten(),
                                label=f"{seq_name} num_gt")
        all_fn_metrics[seq_name] = fn_m
        all_den_metrics[seq_name] = den_m

        fn_pred_at = defaultdict(list)
        fn_epis_at = defaultdict(list)
        fn_true_at = {}
        den_pred_at = defaultdict(list)
        den_true_at = {}

        for w, idx in enumerate(indices):
            window_start = idx - indices[0]
            for h in range(T_HORIZON):
                frame = window_start + T_CONTEXT + h
                fn_pred_at[frame].append(fn_preds[w, h])
                fn_epis_at[frame].append(epistemic[w, h])
                fn_true_at[frame] = y_fn_seq[w, h]

            den_frame = window_start + T_CONTEXT
            den_pred_at[den_frame].append(den_preds[w])
            den_true_at[den_frame] = y_den_seq[w]

        frames_fn = sorted(fn_pred_at.keys())
        frames_den = sorted(den_pred_at.keys())
        val_seq_data[seq_name] = {
            "frames_fn": frames_fn,
            "fn_true": np.array([fn_true_at[f] for f in frames_fn]),
            "fn_pred": np.array([np.mean(fn_pred_at[f]) for f in frames_fn]),
            "fn_epis": np.array([np.mean(fn_epis_at[f]) for f in frames_fn]),
            "frames_den": frames_den,
            "den_true": np.array([den_true_at[f] for f in frames_den]),
            "den_pred": np.array([np.mean(den_pred_at[f]) for f in frames_den]),
        }

    # Overall val metrics
    print("\n" + "-" * 70)
    all_fn_true, all_fn_pred = [], []
    all_den_true, all_den_pred = [], []
    for seq_name in VAL_SEQUENCES:
        indices = [i for i, (s, _) in enumerate(val_ids) if s == seq_name]
        if not indices:
            continue
        X_seq = torch.from_numpy(X_val[indices]).to(device)
        with torch.no_grad():
            gamma, _, _, _, density = model(X_seq)
        all_fn_true.append(y_fn_val[indices].flatten())
        all_fn_pred.append(gamma.cpu().numpy().flatten())
        all_den_true.append(y_den_val[indices].flatten())
        all_den_pred.append((density.cpu().numpy() * density_max).flatten())

    all_fn_true = np.concatenate(all_fn_true)
    all_fn_pred = np.concatenate(all_fn_pred)
    all_den_true = np.concatenate(all_den_true)
    all_den_pred = np.concatenate(all_den_pred)

    overall_fn = compute_metrics(all_fn_true, all_fn_pred, label="OVERALL fn (miss count)")
    overall_den = compute_metrics(all_den_true, all_den_pred, label="OVERALL num_gt")
    all_fn_metrics["OVERALL"] = overall_fn
    all_den_metrics["OVERALL"] = overall_den

    baseline_fn_pred = np.full_like(all_fn_true, all_fn_true.mean())
    baseline_den_pred = np.full_like(all_den_true, all_den_true.mean())
    baseline_fn = compute_metrics(all_fn_true, baseline_fn_pred,
                                  label="BASELINE fn (mean)")
    baseline_den = compute_metrics(all_den_true, baseline_den_pred,
                                   label="BASELINE num_gt (mean)")

    # Per-weather R2
    print("\n" + "=" * 70)
    print("PER-WEATHER FN R2")
    print("=" * 70)
    weather_r2 = {}
    for weather_cat in ["sunny", "cloudy", "night", "rainy"]:
        w_seqs = [s for s in VAL_SEQUENCES if seq_weather.get(s) == weather_cat]
        w_fn_true, w_fn_pred = [], []
        for seq_name in w_seqs:
            indices = [i for i, (s, _) in enumerate(val_ids) if s == seq_name]
            if not indices:
                continue
            X_seq = torch.from_numpy(X_val[indices]).to(device)
            with torch.no_grad():
                gamma, _, _, _, _ = model(X_seq)
            w_fn_true.append(y_fn_val[indices].flatten())
            w_fn_pred.append(gamma.cpu().numpy().flatten())
        if w_fn_true:
            wt = np.concatenate(w_fn_true)
            wp = np.concatenate(w_fn_pred)
            w_r2 = r2_score(wt, wp)
            w_mae = mean_absolute_error(wt, wp)
            w_rho, _ = spearmanr(wt, wp)
            weather_r2[weather_cat] = {"r2": w_r2, "mae": w_mae, "rho": w_rho,
                                       "n": len(wt)}
            print(f"  {weather_cat:8s}  R2={w_r2:.4f}  MAE={w_mae:.4f}  "
                  f"rho={w_rho:.4f}  n={len(wt)}")

    print(f"\n  OVERALL fn R2  = {overall_fn['r2']:.4f}")
    print(f"  OVERALL fn MAE = {overall_fn['mae']:.4f}")
    print(f"  OVERALL fn rho = {overall_fn['rho']:.4f}")
    print(f"  Baseline MAE   = {baseline_fn['mae']:.4f}")

    # ── PDF ────────────────────────────────────────────────────────
    with PdfPages(str(PDF_PATH)) as pdf:

        # Page 1: 8 val sequences (top 2 per weather)
        plot_seqs = [s for s in PLOT_SEQUENCES if s in val_seq_data]
        n_rows = len(plot_seqs)
        fig, axes = plt.subplots(n_rows, 2, figsize=(18, 4 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for row, seq_name in enumerate(plot_seqs):
            d = val_seq_data[seq_name]
            weather = seq_weather.get(seq_name, "unknown")
            fm = all_fn_metrics[seq_name]
            dm = all_den_metrics[seq_name]

            ax = axes[row, 0]
            ax.plot(d["frames_fn"], d["fn_true"], "b-", alpha=0.7, lw=0.8,
                    label="Actual")
            ax.plot(d["frames_fn"], d["fn_pred"], "r-", alpha=0.7, lw=0.8,
                    label="Predicted")
            ax.fill_between(
                d["frames_fn"],
                d["fn_pred"] - np.sqrt(np.clip(d["fn_epis"], 0, None)),
                d["fn_pred"] + np.sqrt(np.clip(d["fn_epis"], 0, None)),
                alpha=0.25, color="orange", label="Epistemic",
            )
            ax.set_ylabel("Miss Count (fn)")
            ax.set_title(f"{seq_name} [{weather}] fn "
                         f"(MAE={fm['mae']:.3f}, R2={fm['r2']:.3f})")
            ax.legend(fontsize=7, loc="upper right")

            ax = axes[row, 1]
            ax.plot(d["frames_den"], d["den_true"], "b-", alpha=0.7, lw=0.8,
                    label="Actual")
            ax.plot(d["frames_den"], d["den_pred"], "r-", alpha=0.7, lw=0.8,
                    label="Predicted")
            ax.set_ylabel("num_gt")
            ax.set_title(f"{seq_name} [{weather}] num_gt "
                         f"(MAE={dm['mae']:.3f}, R2={dm['r2']:.3f})")
            ax.legend(fontsize=7, loc="upper right")

            if row == n_rows - 1:
                axes[row, 0].set_xlabel("Frame index (local)")
                axes[row, 1].set_xlabel("Frame index (local)")

        fig.suptitle("Spatial LSTM (15 features) -- Miss Count Prediction "
                     "-- Top 2 Sequences per Weather",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # Page 2: Diagnostics (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        log_csv = LOG_DIR / "version_0" / "metrics.csv"
        if log_csv.exists():
            logs = pd.read_csv(log_csv)
            ax = axes[0, 0]
            for col, label, color in [
                ("train_loss", "Train total", "blue"),
                ("val_loss", "Val total", "red"),
                ("train_fn_loss", "Train fn", "blue"),
                ("val_fn_loss", "Val fn", "red"),
                ("train_density_loss", "Train density", "blue"),
                ("val_density_loss", "Val density", "red"),
            ]:
                if col in logs.columns:
                    data = logs[["epoch", col]].dropna()
                    if len(data) > 0:
                        epoch_mean = data.groupby("epoch")[col].mean()
                        ls = ("-" if "total" in label
                              else ("--" if "fn" in label else ":"))
                        ax.plot(epoch_mean.index, epoch_mean.values,
                                color=color, ls=ls, alpha=0.7, label=label)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training Curves")
            ax.legend(fontsize=7)
        else:
            axes[0, 0].text(0.5, 0.5, "No log CSV found", ha="center")

        ax = axes[0, 1]
        ax.scatter(all_fn_true, all_fn_pred, alpha=0.1, s=4)
        lims = [min(all_fn_true.min(), all_fn_pred.min()),
                max(all_fn_true.max(), all_fn_pred.max())]
        ax.plot(lims, lims, "k--", lw=0.5)
        ax.set_xlabel("Actual fn (miss count)")
        ax.set_ylabel("Predicted fn")
        ax.set_title(f"Miss Count (R2={overall_fn['r2']:.3f})")

        ax = axes[1, 0]
        ax.scatter(all_den_true, all_den_pred, alpha=0.1, s=4)
        lims = [min(all_den_true.min(), all_den_pred.min()),
                max(all_den_true.max(), all_den_pred.max())]
        ax.plot(lims, lims, "k--", lw=0.5)
        ax.set_xlabel("Actual num_gt")
        ax.set_ylabel("Predicted num_gt")
        ax.set_title(f"Num GT (R2={overall_den['r2']:.3f})")

        ax = axes[1, 1]
        seq_names_plot = [s for s in VAL_SEQUENCES if s in all_fn_metrics]
        x = np.arange(len(seq_names_plot))
        bw = 0.35
        fn_r2 = [all_fn_metrics[s]["r2"] for s in seq_names_plot]
        den_r2 = [all_den_metrics[s]["r2"] for s in seq_names_plot]
        ax.bar(x - bw/2, fn_r2, bw, label="fn R2", color="steelblue")
        ax.bar(x + bw/2, den_r2, bw, label="num_gt R2", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(seq_names_plot, rotation=90, ha="center", fontsize=6)
        ax.set_ylabel("R2")
        ax.set_title("Per-Sequence R2")
        ax.legend(fontsize=7)
        ax.axhline(0, color="k", lw=0.5, ls="--")

        fig.suptitle("Page 2: Diagnostics", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # Page 3: Metrics table
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 10))

        ax1.axis("off")
        rows = []
        for seq in VAL_SEQUENCES + ["OVERALL"]:
            fm = all_fn_metrics.get(seq)
            dm = all_den_metrics.get(seq)
            if fm and dm:
                w_label = seq_weather.get(seq, "")
                rows.append([
                    seq, w_label,
                    f"{fm['mae']:.3f}", f"{fm['r2']:.4f}", f"{fm['rho']:.4f}",
                    f"{dm['mae']:.3f}", f"{dm['r2']:.4f}", f"{dm['rho']:.4f}",
                ])
        rows.append([
            "BASELINE", "(mean)",
            f"{baseline_fn['mae']:.3f}", f"{baseline_fn['r2']:.4f}",
            f"{baseline_fn['rho']:.4f}",
            f"{baseline_den['mae']:.3f}", f"{baseline_den['r2']:.4f}",
            f"{baseline_den['rho']:.4f}",
        ])

        col_labels = ["Sequence", "Weather",
                       "fn MAE", "fn R2", "fn rho",
                       "GT MAE", "GT R2", "GT rho"]
        table = ax1.table(cellText=rows, colLabels=col_labels,
                          loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.4)

        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")

        ax1.set_title("Spatial LSTM -- Validation Metrics (15 spatial features, target=fn)",
                       fontsize=12, fontweight="bold", pad=20)

        fig.suptitle("Page 3: Per-Sequence Metrics", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # Page 4: Cross-experiment comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12),
                                         gridspec_kw={"height_ratios": [2, 1]})

        ax1.axis("off")
        comp_rows = []
        for name, res in PREV_RESULTS.items():
            comp_rows.append([
                name, res["target"],
                f"{res['r2']:.4f}", f"{res['mae']:.4f}", f"{res['rho']:.4f}",
            ])
        comp_rows.append([
            "Spatial (15 feat)", "miss_count (fn)",
            f"{overall_fn['r2']:.4f}", f"{overall_fn['mae']:.4f}",
            f"{overall_fn['rho']:.4f}",
        ])

        comp_labels = ["Experiment", "Target", "R2", "MAE", "rho"]
        comp_table = ax1.table(cellText=comp_rows, colLabels=comp_labels,
                               loc="center", cellLoc="center")
        comp_table.auto_set_font_size(False)
        comp_table.set_fontsize(9)
        comp_table.scale(1.0, 2.0)

        for j in range(len(comp_labels)):
            comp_table[0, j].set_facecolor("#C44472")
            comp_table[0, j].set_text_props(color="white", fontweight="bold")
        # Highlight last row (spatial)
        last_row = len(comp_rows)
        for j in range(len(comp_labels)):
            comp_table[last_row, j].set_facecolor("#E8F5E9")

        ax1.set_title("Cross-Experiment Comparison (all LSTM experiments on DETRAC)",
                       fontsize=12, fontweight="bold", pad=20)

        # Per-weather R2 bar chart
        weather_cats = ["sunny", "cloudy", "night", "rainy"]
        x = np.arange(len(weather_cats))
        bw = 0.25
        spatial_r2 = [weather_r2.get(w, {}).get("r2", 0) for w in weather_cats]

        ax2.bar(x, spatial_r2, bw, label="Spatial (15 feat, fn)", color="steelblue")
        ax2.set_xticks(x)
        ax2.set_xticklabels(weather_cats, fontsize=10)
        ax2.set_ylabel("R2")
        ax2.set_title("Per-Weather fn R2", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.axhline(0, color="k", lw=0.5, ls="--")

        fig.suptitle("Page 4: Cross-Experiment Comparison", fontsize=14,
                     fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"\nPDF saved -> {PDF_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
