#!/usr/bin/env python3
"""LSTM vs Random Forest — shifted-label prediction on DETRAC.

Task:  Given 30 frames of spatial features (1 sec), predict the mean
       fn_nano over the NEXT 30 frames (1 sec lookahead).

Design choices (all bugs from prior experiments fixed):
  - NO num_gt >= 1 filter        → preserves temporal continuity
  - NO per-sequence mean removal → avoids look-ahead bias
  - Stride=15 for train          → 50% overlap, enough diversity
  - Stride=30 for val/test       → non-overlapping, unbiased metrics
  - Sequence-level split         → zero train/val/test leakage
  - StandardScaler fit on train  → no future information leak
  - Frame-ID gap check           → verified temporal continuity

Output: new_features/new_train_results.pdf
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "new_all_features.csv"
OUT_PDF = SCRIPT_DIR / "new_train_results.pdf"
SCALER_PATH = SCRIPT_DIR / "new_scaler.pkl"

META_COLS = ["sequence", "frame_id", "weather", "fn_nano", "num_gt",
             "tp_nano", "tp_medium", "recoverable_fn"]

# ── Task configuration ─────────────────────────────────────────────
WINDOW = 30          # input window:  30 frames = 1 sec
HORIZON = 30         # target window: next 30 frames = 1 sec lookahead
TRAIN_STRIDE = 15    # 50% overlap for training
EVAL_STRIDE = 30     # non-overlapping for evaluation

TARGET = "fn_nano"

# 20 non-redundant features from new_select_features.py
SELECTED_FEATURES = [
    "fast_keypoints_half",
    "edge_density_coarse",
    "gradient_magnitude_mean",
    "gabor_nyquist_energy",
    "fast_keypoints_full",
    "glcm_homogeneity",
    "overall_edge_density",
    "top_edge_density",
    "top_high_freq_energy",
    "top_gradient_std",
    "fft_critical_band_ratio",
    "total_edge_components",
    "top_laplacian_var",
    "glcm_entropy",
    "mid_edge_density",
    "dark_channel_mean",
    "mid_gradient_std",
    "foreground_blob_count",
    "mid_high_freq_energy",
    "rms_contrast",
]

# ── Train/val split (sequence-level, stratified by weather) ────────
# 80/20 split: ~80 train, ~20 val — hand-picked for weather balance
VAL_SEQUENCES = [
    # sunny (5 sequences)
    "MVI_39361", "MVI_20011", "MVI_20012", "MVI_20032", "MVI_20051",
    # cloudy (5 sequences)
    "MVI_40172", "MVI_40191", "MVI_40712", "MVI_40244", "MVI_40243",
    # night (5 sequences)
    "MVI_40992", "MVI_40732", "MVI_39811", "MVI_40775", "MVI_40752",
    # rainy (5 sequences)
    "MVI_63521", "MVI_40892", "MVI_63553", "MVI_63525", "MVI_63544",
]


# ═══════════════════════════════════════════════════════════════════
# Data Pipeline
# ═══════════════════════════════════════════════════════════════════

def load_and_prepare():
    """Load CSV, fill NaN, verify temporal continuity."""
    print("Loading data ...", flush=True)
    df = pd.read_csv(FEATURES_CSV)

    # Fill NaN in temporal features (first frame of each sequence → 0)
    for c in SELECTED_FEATURES:
        if "temporal" in c or c == "motion_pixel_ratio":
            df[c] = df[c].fillna(0.0)

    # Verify selected features exist
    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Check temporal continuity: frame_id should increment by 1 within each seq
    gap_count = 0
    for seq_name, sdf in df.groupby("sequence"):
        fids = sdf.sort_values("frame_id")["frame_id"].values
        diffs = np.diff(fids)
        n_gaps = int((diffs != 1).sum())
        if n_gaps > 0:
            gap_count += n_gaps
    if gap_count > 0:
        print(f"  WARNING: {gap_count} frame-ID gaps detected across sequences")
    else:
        print("  Temporal continuity verified: no frame-ID gaps")

    # Filter val sequences that actually exist in data
    available_seqs = set(df["sequence"].unique())
    actual_val = [s for s in VAL_SEQUENCES if s in available_seqs]
    if len(actual_val) < len(VAL_SEQUENCES):
        missing_val = set(VAL_SEQUENCES) - available_seqs
        print(f"  Note: {len(missing_val)} val sequences not in data: {missing_val}")

    print(f"  {len(df):,} frames, {df['sequence'].nunique()} seqs, "
          f"{len(SELECTED_FEATURES)} features")
    return df, actual_val


def build_windows(df, feature_cols, target_col, stride, scaler=None):
    """Build shifted-label sliding windows.

    Input:  features[t : t+WINDOW]           → shape (WINDOW, n_features)
    Target: mean(fn_nano[t+WINDOW : t+WINDOW+HORIZON]) → scalar

    Returns: X (n_windows, WINDOW, n_features), y (n_windows,),
             seq_ids list of (seq_name, start_frame_idx)
    """
    total_need = WINDOW + HORIZON
    X_list, y_list, seq_ids = [], [], []

    for seq_name, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        n = len(sdf)
        if n < total_need:
            continue

        feat_vals = sdf[feature_cols].values  # (n, n_features)
        target_vals = sdf[target_col].values  # (n,)

        # Apply scaler if provided (already fit on train)
        if scaler is not None:
            feat_vals = scaler.transform(feat_vals)

        for t in range(0, n - total_need + 1, stride):
            x_window = feat_vals[t : t + WINDOW]                # (30, n_feat)
            y_target = target_vals[t + WINDOW : t + WINDOW + HORIZON].mean()  # scalar

            X_list.append(x_window)
            y_list.append(y_target)
            seq_ids.append((seq_name, sdf["frame_id"].iloc[t]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, seq_ids


def build_rf_features(X_windows):
    """Convert LSTM-style windows into flat RF features.

    For each feature, compute: mean, std, linear slope over the window.
    → 3 × n_features flat vector per sample.
    """
    n_samples, window_len, n_feat = X_windows.shape
    t_centered = np.arange(window_len, dtype=np.float64) - (window_len - 1) / 2.0
    t_var = (t_centered ** 2).sum()  # constant for all samples

    means = X_windows.mean(axis=1)                    # (n, n_feat)
    stds = X_windows.std(axis=1)                      # (n, n_feat)

    # Vectorized linear slope: slope = sum(t_centered * x) / sum(t_centered^2)
    # X_windows: (n, 30, n_feat), t_centered: (30,)
    slopes = np.einsum("ijk,j->ik", X_windows.astype(np.float64),
                       t_centered) / t_var            # (n, n_feat)

    return np.hstack([means, stds, slopes.astype(np.float32)])


# ═══════════════════════════════════════════════════════════════════
# LSTM Model
# ═══════════════════════════════════════════════════════════════════

class ForecastLSTM(nn.Module):
    """Simple LSTM: 30 frames of features → scalar fn prediction.

    No evidential head — just MSE. Keeps comparison with RF fair.
    """
    def __init__(self, input_size, hidden_size=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, 30, n_features)
        _, (h_n, _) = self.lstm(x)
        # h_n: (2, batch, hidden) — use last layer
        hidden = h_n[-1]  # (batch, hidden)
        return self.head(hidden).squeeze(-1)  # (batch,)


def train_lstm(X_train, y_train, X_val, y_val, n_features, device):
    """Train LSTM with early stopping."""
    print("\n" + "=" * 60)
    print("TRAINING LSTM")
    print("=" * 60)

    model = ForecastLSTM(input_size=n_features, hidden_size=64, dropout=0.2)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: LSTM(2-layer, h=64) → MLP(64→32→1)")
    print(f"  Parameters: {n_params:,}")
    print(f"  Train windows: {len(X_train):,} (stride={TRAIN_STRIDE})")
    print(f"  Val windows:   {len(X_val):,} (stride={EVAL_STRIDE})")

    train_ds = TensorDataset(torch.from_numpy(X_train),
                             torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val),
                           torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=128)

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience_limit = 25
    train_losses, val_losses = [], []

    for epoch in range(200):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss_sum += criterion(pred, yb).item() * len(yb)
                n_val += len(yb)
        val_loss = val_loss_sum / n_val

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        lr = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 0 or patience_counter == 0:
            marker = " *" if patience_counter == 0 else ""
            print(f"  Epoch {epoch:3d}: train={train_loss:.5f}  "
                  f"val={val_loss:.5f}  lr={lr:.1e}{marker}", flush=True)

        if patience_counter >= patience_limit:
            print(f"  Early stopping at epoch {epoch} "
                  f"(best val={best_val_loss:.5f})")
            break

    # Load best model
    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    return model, train_losses, val_losses


def train_random_forest(X_train_rf, y_train, X_val_rf, y_val):
    """Train Random Forest on summary-statistics features."""
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    n_rf_features = X_train_rf.shape[1]
    print(f"  Features: {n_rf_features} (mean + std + slope × {len(SELECTED_FEATURES)})")
    print(f"  Train samples: {len(X_train_rf):,} (stride={TRAIN_STRIDE})")
    print(f"  Val samples:   {len(X_val_rf):,} (stride={EVAL_STRIDE})")

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    print("  Training ...", flush=True)
    rf.fit(X_train_rf, y_train)

    # Feature importances
    feat_names = (
        [f"{f}_mean" for f in SELECTED_FEATURES] +
        [f"{f}_std" for f in SELECTED_FEATURES] +
        [f"{f}_slope" for f in SELECTED_FEATURES]
    )
    importances = pd.Series(rf.feature_importances_, index=feat_names)
    importances = importances.sort_values(ascending=False)

    print(f"\n  Top 10 RF features by importance:")
    for fname, imp in importances.head(10).items():
        print(f"    {fname:<40s} {imp:.4f}")

    return rf, importances


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_model(y_true, y_pred, label=""):
    """Compute R², MAE, Spearman ρ."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rho, p = spearmanr(y_true, y_pred)
    if np.isnan(rho):
        rho = 0.0
    print(f"  {label:45s}  R²={r2:.4f}  MAE={mae:.3f}  ρ={rho:.4f}")
    return {"r2": r2, "mae": mae, "rho": rho}


def per_sequence_eval(y_true, y_pred, seq_ids, weather_map, label=""):
    """Evaluate per-sequence and per-weather."""
    # Group predictions by sequence
    seq_data = defaultdict(lambda: {"true": [], "pred": []})
    for i, (seq_name, _) in enumerate(seq_ids):
        seq_data[seq_name]["true"].append(y_true[i])
        seq_data[seq_name]["pred"].append(y_pred[i])

    seq_metrics = {}
    for seq_name, data in sorted(seq_data.items()):
        yt = np.array(data["true"])
        yp = np.array(data["pred"])
        if len(yt) < 3 or yt.std() < 1e-10:
            continue
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        rho, _ = spearmanr(yt, yp)
        weather = weather_map.get(seq_name, "?")
        seq_metrics[seq_name] = {"r2": r2, "mae": mae, "rho": rho,
                                 "weather": weather, "n": len(yt)}

    # Per-weather aggregation
    weather_metrics = defaultdict(lambda: {"true": [], "pred": []})
    for seq_name, data in seq_data.items():
        w = weather_map.get(seq_name, "?")
        weather_metrics[w]["true"].extend(data["true"])
        weather_metrics[w]["pred"].extend(data["pred"])

    print(f"\n  Per-weather ({label}):")
    weather_results = {}
    for w in sorted(weather_metrics):
        yt = np.array(weather_metrics[w]["true"])
        yp = np.array(weather_metrics[w]["pred"])
        if len(yt) < 3:
            continue
        r2 = r2_score(yt, yp)
        rho, _ = spearmanr(yt, yp)
        weather_results[w] = {"r2": r2, "rho": rho, "n": len(yt)}
        print(f"    {w:8s}  R²={r2:.4f}  ρ={rho:.4f}  n={len(yt)}")

    return seq_metrics, weather_results


# ═══════════════════════════════════════════════════════════════════
# PDF Report
# ═══════════════════════════════════════════════════════════════════

def generate_pdf(lstm_metrics, rf_metrics, baseline_metrics,
                 lstm_seq, rf_seq, lstm_weather, rf_weather,
                 train_losses, val_losses, rf_importances,
                 y_val_true, lstm_preds, rf_preds, val_seq_ids,
                 weather_map):
    """Generate comparison PDF."""

    with PdfPages(OUT_PDF) as pdf:

        # ── Page 1: Training curves + scatter plots ──
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Training curves
        ax = axes[0, 0]
        epochs = range(len(train_losses))
        ax.plot(epochs, train_losses, "b-", alpha=0.7, label="Train MSE")
        ax.plot(epochs, val_losses, "r-", alpha=0.7, label="Val MSE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("LSTM Training Curves")
        ax.legend()
        ax.set_yscale("log")

        # LSTM scatter
        ax = axes[0, 1]
        ax.scatter(y_val_true, lstm_preds, alpha=0.3, s=8, c="steelblue")
        lims = [min(y_val_true.min(), lstm_preds.min()),
                max(y_val_true.max(), lstm_preds.max())]
        ax.plot(lims, lims, "k--", lw=0.5)
        ax.set_xlabel("Actual mean(fn_nano)")
        ax.set_ylabel("Predicted")
        ax.set_title(f"LSTM: R²={lstm_metrics['r2']:.4f}, ρ={lstm_metrics['rho']:.4f}")

        # RF scatter
        ax = axes[1, 0]
        ax.scatter(y_val_true, rf_preds, alpha=0.3, s=8, c="forestgreen")
        ax.plot(lims, lims, "k--", lw=0.5)
        ax.set_xlabel("Actual mean(fn_nano)")
        ax.set_ylabel("Predicted")
        ax.set_title(f"RF: R²={rf_metrics['r2']:.4f}, ρ={rf_metrics['rho']:.4f}")

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

        # ── Page 2: Per-weather comparison ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        weather_cats = sorted(set(lstm_weather) | set(rf_weather))
        x = np.arange(len(weather_cats))
        bw = 0.35

        # R²
        ax = axes[0]
        lstm_r2 = [lstm_weather.get(w, {}).get("r2", 0) for w in weather_cats]
        rf_r2 = [rf_weather.get(w, {}).get("r2", 0) for w in weather_cats]
        ax.bar(x - bw/2, lstm_r2, bw, label="LSTM", color="steelblue")
        ax.bar(x + bw/2, rf_r2, bw, label="RF", color="forestgreen")
        ax.set_xticks(x)
        ax.set_xticklabels(weather_cats)
        ax.set_ylabel("R²")
        ax.set_title("Per-Weather R²")
        ax.legend()
        ax.axhline(0, color="k", lw=0.5, ls="--")

        # ρ
        ax = axes[1]
        lstm_rho = [lstm_weather.get(w, {}).get("rho", 0) for w in weather_cats]
        rf_rho = [rf_weather.get(w, {}).get("rho", 0) for w in weather_cats]
        ax.bar(x - bw/2, lstm_rho, bw, label="LSTM", color="steelblue")
        ax.bar(x + bw/2, rf_rho, bw, label="RF", color="forestgreen")
        ax.set_xticks(x)
        ax.set_xticklabels(weather_cats)
        ax.set_ylabel("Spearman ρ")
        ax.set_title("Per-Weather Spearman ρ")
        ax.legend()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # ── Page 3: Per-sequence R² comparison ──
        fig, ax = plt.subplots(figsize=(16, 10))
        common_seqs = sorted(set(lstm_seq) & set(rf_seq),
                             key=lambda s: lstm_seq[s]["r2"], reverse=True)
        if len(common_seqs) > 25:
            common_seqs = common_seqs[:25]
        x = np.arange(len(common_seqs))
        bw = 0.35
        lr2 = [lstm_seq[s]["r2"] for s in common_seqs]
        rr2 = [rf_seq[s]["r2"] for s in common_seqs]
        labels = [f"{s} [{lstm_seq[s]['weather']}]" for s in common_seqs]
        ax.barh(x - bw/2, lr2, bw, label="LSTM", color="steelblue")
        ax.barh(x + bw/2, rr2, bw, label="RF", color="forestgreen")
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("R²")
        ax.set_title("Per-Sequence R² (val set, non-overlapping windows)")
        ax.legend()
        ax.axvline(0, color="k", lw=0.5)
        ax.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # ── Page 4: Example sequences (best 4) ──
        best_seqs = common_seqs[:4]
        fig, axes = plt.subplots(len(best_seqs), 1, figsize=(16, 4 * len(best_seqs)))
        if len(best_seqs) == 1:
            axes = [axes]

        # Build per-sequence prediction timeseries
        seq_ts = defaultdict(lambda: {"frame": [], "true": [],
                                       "lstm": [], "rf": []})
        for i, (sn, fid) in enumerate(val_seq_ids):
            seq_ts[sn]["frame"].append(fid)
            seq_ts[sn]["true"].append(y_val_true[i])
            seq_ts[sn]["lstm"].append(lstm_preds[i])
            seq_ts[sn]["rf"].append(rf_preds[i])

        for row, seq_name in enumerate(best_seqs):
            ax = axes[row]
            d = seq_ts[seq_name]
            # Sort by frame_id
            order = np.argsort(d["frame"])
            frames = np.array(d["frame"])[order]
            true_v = np.array(d["true"])[order]
            lstm_v = np.array(d["lstm"])[order]
            rf_v = np.array(d["rf"])[order]

            ax.plot(frames, true_v, "k-", lw=1.2, alpha=0.8, label="Actual")
            ax.plot(frames, lstm_v, "b-", lw=0.8, alpha=0.7, label="LSTM")
            ax.plot(frames, rf_v, "g-", lw=0.8, alpha=0.7, label="RF")
            w = weather_map.get(seq_name, "?")
            lr = lstm_seq.get(seq_name, {}).get("r2", 0)
            rr = rf_seq.get(seq_name, {}).get("r2", 0)
            ax.set_title(f"{seq_name} [{w}]  LSTM R²={lr:.3f}  RF R²={rr:.3f}",
                         fontsize=10)
            ax.set_ylabel("mean fn_nano (next 30f)")
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("Frame ID")
        fig.suptitle("Best Val Sequences — Actual vs Predicted", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig)
        plt.close()

        # ── Page 5: Summary table ──
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")

        rows = [
            ["", "R²", "MAE", "Spearman ρ"],
            ["LSTM (2-layer, h=64)", f"{lstm_metrics['r2']:.4f}",
             f"{lstm_metrics['mae']:.3f}", f"{lstm_metrics['rho']:.4f}"],
            ["Random Forest (500 trees)", f"{rf_metrics['r2']:.4f}",
             f"{rf_metrics['mae']:.3f}", f"{rf_metrics['rho']:.4f}"],
            ["Baseline (predict mean)", f"{baseline_metrics['r2']:.4f}",
             f"{baseline_metrics['mae']:.3f}", f"{baseline_metrics['rho']:.4f}"],
        ]

        table = ax.table(cellText=rows[1:], colLabels=rows[0],
                         loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 2.0)
        for j in range(4):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Add config info below table
        config_text = (
            f"Task: predict mean(fn_nano) over next {HORIZON} frames "
            f"from {WINDOW} frames of {len(SELECTED_FEATURES)} features\n"
            f"Train stride: {TRAIN_STRIDE}, Eval stride: {EVAL_STRIDE} "
            f"(non-overlapping)\n"
            f"Split: {100 - len(VAL_SEQUENCES)}+ train / "
            f"{len(VAL_SEQUENCES)} val sequences (stratified by weather)\n"
            f"Normalization: StandardScaler fit on train only\n"
            f"No num_gt filter (preserves temporal continuity)\n"
            f"No per-sequence mean removal (avoids look-ahead bias)"
        )
        ax.text(0.5, 0.25, config_text, transform=ax.transAxes,
                ha="center", va="top", fontfamily="monospace", fontsize=9)

        ax.set_title("LSTM vs Random Forest — Shifted-Label Prediction",
                     fontsize=14, fontweight="bold", pad=40)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"\n  PDF: {OUT_PDF}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    df, actual_val = load_and_prepare()
    weather_map = df.drop_duplicates("sequence").set_index("sequence")["weather"].to_dict()

    # ── Train/val split by sequence ──
    train_df = df[~df["sequence"].isin(actual_val)].copy()
    val_df = df[df["sequence"].isin(actual_val)].copy()

    n_train_seqs = train_df["sequence"].nunique()
    n_val_seqs = val_df["sequence"].nunique()
    print(f"\n  Train: {n_train_seqs} seqs, {len(train_df):,} frames")
    print(f"  Val:   {n_val_seqs} seqs, {len(val_df):,} frames")

    # ── Fit scaler on train only ──
    scaler = StandardScaler()
    scaler.fit(train_df[SELECTED_FEATURES].values)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved: {SCALER_PATH}")

    # ── Build windows ──
    print("\n  Building windows ...", flush=True)
    X_train, y_train, train_ids = build_windows(
        train_df, SELECTED_FEATURES, TARGET, stride=TRAIN_STRIDE, scaler=scaler)
    X_val, y_val, val_ids = build_windows(
        val_df, SELECTED_FEATURES, TARGET, stride=EVAL_STRIDE, scaler=scaler)

    print(f"  Train: {X_train.shape[0]:,} windows × {X_train.shape[1]} steps "
          f"× {X_train.shape[2]} features")
    print(f"  Val:   {X_val.shape[0]:,} windows × {X_val.shape[1]} steps "
          f"× {X_val.shape[2]} features")
    print(f"  Target range: [{y_train.min():.2f}, {y_train.max():.2f}], "
          f"mean={y_train.mean():.2f}")

    # ── Train LSTM ──
    lstm_model, train_losses, val_losses = train_lstm(
        X_train, y_train, X_val, y_val,
        n_features=len(SELECTED_FEATURES), device=device)

    # ── Train RF ──
    X_train_rf = build_rf_features(X_train)
    X_val_rf = build_rf_features(X_val)
    rf_model, rf_importances = train_random_forest(
        X_train_rf, y_train, X_val_rf, y_val)

    # ── Evaluate ──
    print("\n" + "=" * 60)
    print("EVALUATION (non-overlapping val windows, stride=30)")
    print("=" * 60)

    # LSTM predictions
    lstm_model.eval()
    with torch.no_grad():
        lstm_preds = lstm_model(
            torch.from_numpy(X_val).to(device)).cpu().numpy()

    # RF predictions
    rf_preds = rf_model.predict(X_val_rf)

    # Baseline: predict training mean
    baseline_pred = np.full_like(y_val, y_train.mean())

    print("\n  Overall:")
    lstm_metrics = evaluate_model(y_val, lstm_preds, "LSTM")
    rf_metrics = evaluate_model(y_val, rf_preds, "Random Forest")
    baseline_metrics = evaluate_model(y_val, baseline_pred, "Baseline (train mean)")

    # Per-sequence + per-weather
    lstm_seq, lstm_weather = per_sequence_eval(
        y_val, lstm_preds, val_ids, weather_map, "LSTM")
    rf_seq, rf_weather = per_sequence_eval(
        y_val, rf_preds, val_ids, weather_map, "RF")

    # ── PDF ──
    print("\n  Generating PDF ...", flush=True)
    generate_pdf(
        lstm_metrics, rf_metrics, baseline_metrics,
        lstm_seq, rf_seq, lstm_weather, rf_weather,
        train_losses, val_losses, rf_importances,
        y_val, lstm_preds, rf_preds, val_ids, weather_map)

    print("\nDone.")


if __name__ == "__main__":
    main()
