"""Deployment-realistic LSTM — predict nano's own recall (not gap).

23 features: 14 image + 9 nano detection.
Target: nano_recall (range 0-1), not recall_gap.

Architecture identical to 2lstms_train_nano.py.
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
MERGED_CSV = SCRIPT_DIR / "2lstms_nano_recall_data.csv"

SCALER_PATH      = SCRIPT_DIR / "2lstms_nano_recall_scaler.pkl"
DENSITY_MAX_PATH = SCRIPT_DIR / "2lstms_nano_recall_density_max.pkl"
CKPT_DIR         = SCRIPT_DIR
LOG_DIR          = SCRIPT_DIR / "2lstms_nano_recall_logs"
PDF_PATH         = SCRIPT_DIR / "2lstms_nano_recall_results.pdf"

T_CONTEXT = 20
T_HORIZON = 10

TARGET_COL = "nano_recall"

# 14 image features
IMAGE_FEATURES = [
    "gradient_magnitude_std", "brightness_skewness", "overexposed_ratio",
    "contour_count", "wavelet_detail_energy", "high_freq_energy_ratio",
    "mean_brightness", "spectral_rolloff", "brightness_kurtosis",
    "dark_channel_std", "color_channel_corr", "FAST_keypoint_count",
    "ssim_prev", "temporal_diff_mean",
]

# 9 nano detection features (no medium)
DETECTION_FEATURES = [
    "nano_num_detections", "nano_mean_confidence", "nano_min_confidence",
    "nano_confidence_std", "nano_mean_box_area", "nano_min_box_area",
    "nano_box_area_std", "nano_detection_density", "nano_edge_box_ratio",
]

FEATURES = IMAGE_FEATURES + DETECTION_FEATURES  # 23 total

# Same 15 val sequences as all other experiments
VAL_SEQUENCES = [
    # sunny
    "MVI_39361", "MVI_20011", "MVI_41073", "MVI_39211",
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

# Previous nano gap-target results for comparison
PREV_NANO_GAP = {"r2": -0.017, "mae": 0.137, "rho": 0.357}


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

    def __init__(self, input_size=23, hidden_size=64, horizon=10, lr=1e-3,
                 coeff=0.1, lambda_density=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True)
        self.head_recall = nn.Linear(hidden_size, horizon * 4)  # evidential
        self.head_density = nn.Linear(hidden_size, 1)            # 1-step

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        hidden = h_n.squeeze(0)

        recall_out = self.head_recall(hidden).view(-1, self.hparams.horizon, 4)
        gamma = recall_out[..., 0]
        v = F.softplus(recall_out[..., 1])
        alpha = F.softplus(recall_out[..., 2]) + 1
        beta = F.softplus(recall_out[..., 3])

        density = F.softplus(self.head_density(hidden)).squeeze(-1)

        return gamma, v, alpha, beta, density

    def training_step(self, batch, batch_idx):
        x, y_recall, y_density = batch
        gamma, v, alpha, beta, density = self(x)

        loss_recall = EvidentialLoss(y_recall, gamma, v, alpha, beta,
                                     coeff=self.hparams.coeff)
        loss_density = F.mse_loss(density, y_density)
        loss = loss_recall + self.hparams.lambda_density * loss_density

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recall_loss", loss_recall)
        self.log("train_density_loss", loss_density)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_recall, y_density = batch
        gamma, v, alpha, beta, density = self(x)

        loss_recall = EvidentialLoss(y_recall, gamma, v, alpha, beta,
                                     coeff=self.hparams.coeff)
        loss_density = F.mse_loss(density, y_density)
        loss = loss_recall + self.hparams.lambda_density * loss_density

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recall_loss", loss_recall)
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
    """Load merged CSV, smooth targets, filter."""
    df = pd.read_csv(MERGED_CSV)
    print(f"Data loaded: {len(df)} rows, {df['sequence'].nunique()} sequences")
    print(f"Features: {len(FEATURES)} ({len(IMAGE_FEATURES)} image + "
          f"{len(DETECTION_FEATURES)} detection)")
    print(f"Target: {TARGET_COL}")
    print(f"Weather: {df['weather'].value_counts().to_dict()}")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    print(f"  {TARGET_COL} range (raw): [{df[TARGET_COL].min():.4f}, "
          f"{df[TARGET_COL].max():.4f}], mean={df[TARGET_COL].mean():.4f}")

    # Forward-looking smoothing on nano_recall
    df[TARGET_COL] = df.groupby("sequence")[TARGET_COL].transform(
        lambda s: _forward_smooth(s, window=10)
    )

    print(f"  {TARGET_COL} range (smoothed): [{df[TARGET_COL].min():.4f}, "
          f"{df[TARGET_COL].max():.4f}], mean={df[TARGET_COL].mean():.4f}")

    df = df[df["num_gt"] >= 1]
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    print(f"After filtering: {len(df)} rows, {df['sequence'].nunique()} sequences")
    return df


def build_sliding_windows(df, feature_cols):
    """Build sliding windows with stride = T_CONTEXT."""
    X_list, y_recall_list, y_density_list, seq_ids = [], [], [], []
    total_window = T_CONTEXT + T_HORIZON

    for seq_name, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        features = sdf[feature_cols].values
        recall_targets = sdf[TARGET_COL].values
        density_targets = sdf["num_gt"].values

        n = len(sdf)
        for i in range(0, n - total_window + 1, T_CONTEXT):
            X_list.append(features[i : i + T_CONTEXT])
            y_recall_list.append(recall_targets[i + T_CONTEXT : i + T_CONTEXT + T_HORIZON])
            y_density_list.append(density_targets[i + T_CONTEXT])
            seq_ids.append((seq_name, i))

    X = np.array(X_list, dtype=np.float32)
    y_recall = np.array(y_recall_list, dtype=np.float32)
    y_density = np.array(y_density_list, dtype=np.float32)
    return X, y_recall, y_density, seq_ids


def compute_metrics(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    print(f"  {label:40s}  MAE={mae:.5f}  R2={r2:.4f}  rho={rho:.4f}")
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

    # ── Normalize features ──
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES])
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved -> {SCALER_PATH}")

    df[FEATURES] = scaler.transform(df[FEATURES])

    # ── Normalize num_gt ──
    density_max = float(train_df["num_gt"].max())
    with open(DENSITY_MAX_PATH, "wb") as f:
        pickle.dump(density_max, f)
    print(f"Density max = {density_max}")

    df["num_gt_norm"] = df["num_gt"] / density_max

    train_df = df[~df["sequence"].isin(VAL_SEQUENCES)].copy()
    val_df = df[df["sequence"].isin(VAL_SEQUENCES)].copy()

    # ── Sliding windows ──
    X_train, y_recall_train, y_den_train, train_ids = build_sliding_windows(
        train_df, FEATURES)
    X_val, y_recall_val, y_den_val, val_ids = build_sliding_windows(
        val_df, FEATURES)

    _, _, y_den_train_norm, _ = build_sliding_windows(
        train_df.assign(**{"num_gt": train_df["num_gt_norm"]}), FEATURES)
    _, _, y_den_val_norm, _ = build_sliding_windows(
        val_df.assign(**{"num_gt": val_df["num_gt_norm"]}), FEATURES)

    print(f"Windows -- train: {X_train.shape[0]}, val: {X_val.shape[0]}")
    print(f"X shape: {X_train.shape}, y_recall: {y_recall_train.shape}, "
          f"y_den: {y_den_train_norm.shape}")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_recall_train),
        torch.from_numpy(y_den_train_norm),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_recall_val),
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
            filename="2lstms_nano_recall_best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
        ),
    ]

    csv_logger = CSVLogger(save_dir=str(SCRIPT_DIR), name="2lstms_nano_recall_logs",
                           version=0)

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

    all_recall_metrics = {}
    all_den_metrics = {}
    val_seq_data = {}

    for seq_name in VAL_SEQUENCES:
        indices = [i for i, (s, _) in enumerate(val_ids) if s == seq_name]
        if not indices:
            continue

        X_seq = torch.from_numpy(X_val[indices]).to(device)
        y_recall_seq = y_recall_val[indices]
        y_den_seq = y_den_val[indices]

        with torch.no_grad():
            gamma, v, alpha, beta, density = model(X_seq)
            recall_preds = gamma.cpu().numpy()
            den_preds_norm = density.cpu().numpy()
            aleatoric, epistemic = model.get_uncertainty(v, alpha, beta)
            epistemic = epistemic.cpu().numpy()

        den_preds = den_preds_norm * density_max

        weather = seq_weather.get(seq_name, "unknown")
        print(f"\n  --- {seq_name} ({weather}) ---")
        recall_m = compute_metrics(y_recall_seq.flatten(), recall_preds.flatten(),
                                   label=f"{seq_name} nano_recall")
        den_m = compute_metrics(y_den_seq.flatten(), den_preds.flatten(),
                                label=f"{seq_name} num_gt")
        all_recall_metrics[seq_name] = recall_m
        all_den_metrics[seq_name] = den_m

        recall_pred_at = defaultdict(list)
        recall_epis_at = defaultdict(list)
        recall_true_at = {}
        den_pred_at = defaultdict(list)
        den_true_at = {}

        for w, idx in enumerate(indices):
            window_start = idx - indices[0]
            for h in range(T_HORIZON):
                frame = window_start + T_CONTEXT + h
                recall_pred_at[frame].append(recall_preds[w, h])
                recall_epis_at[frame].append(epistemic[w, h])
                recall_true_at[frame] = y_recall_seq[w, h]

            den_frame = window_start + T_CONTEXT
            den_pred_at[den_frame].append(den_preds[w])
            den_true_at[den_frame] = y_den_seq[w]

        frames_recall = sorted(recall_pred_at.keys())
        frames_den = sorted(den_pred_at.keys())
        val_seq_data[seq_name] = {
            "frames_recall": frames_recall,
            "recall_true": np.array([recall_true_at[f] for f in frames_recall]),
            "recall_pred": np.array([np.mean(recall_pred_at[f]) for f in frames_recall]),
            "recall_epis": np.array([np.mean(recall_epis_at[f]) for f in frames_recall]),
            "frames_den": frames_den,
            "den_true": np.array([den_true_at[f] for f in frames_den]),
            "den_pred": np.array([np.mean(den_pred_at[f]) for f in frames_den]),
        }

    # Overall val metrics
    print("\n" + "-" * 70)
    all_recall_true, all_recall_pred = [], []
    all_den_true, all_den_pred = [], []
    for seq_name in VAL_SEQUENCES:
        indices = [i for i, (s, _) in enumerate(val_ids) if s == seq_name]
        if not indices:
            continue
        X_seq = torch.from_numpy(X_val[indices]).to(device)
        with torch.no_grad():
            gamma, _, _, _, density = model(X_seq)
        all_recall_true.append(y_recall_val[indices].flatten())
        all_recall_pred.append(gamma.cpu().numpy().flatten())
        all_den_true.append(y_den_val[indices].flatten())
        all_den_pred.append((density.cpu().numpy() * density_max).flatten())

    all_recall_true = np.concatenate(all_recall_true)
    all_recall_pred = np.concatenate(all_recall_pred)
    all_den_true = np.concatenate(all_den_true)
    all_den_pred = np.concatenate(all_den_pred)

    overall_recall = compute_metrics(all_recall_true, all_recall_pred,
                                     label="OVERALL nano_recall")
    overall_den = compute_metrics(all_den_true, all_den_pred, label="OVERALL num_gt")
    all_recall_metrics["OVERALL"] = overall_recall
    all_den_metrics["OVERALL"] = overall_den

    baseline_recall_pred = np.full_like(all_recall_true, all_recall_true.mean())
    baseline_den_pred = np.full_like(all_den_true, all_den_true.mean())
    baseline_recall = compute_metrics(all_recall_true, baseline_recall_pred,
                                      label="BASELINE nano_recall (mean)")
    baseline_den = compute_metrics(all_den_true, baseline_den_pred,
                                   label="BASELINE num_gt (mean)")

    # Per-weather R2
    print("\n" + "=" * 70)
    print("PER-WEATHER NANO_RECALL R2")
    print("=" * 70)
    weather_r2 = {}
    for weather_cat in ["sunny", "cloudy", "night", "rainy"]:
        w_seqs = [s for s in VAL_SEQUENCES if seq_weather.get(s) == weather_cat]
        w_recall_true, w_recall_pred = [], []
        for seq_name in w_seqs:
            indices = [i for i, (s, _) in enumerate(val_ids) if s == seq_name]
            if not indices:
                continue
            X_seq = torch.from_numpy(X_val[indices]).to(device)
            with torch.no_grad():
                gamma, _, _, _, _ = model(X_seq)
            w_recall_true.append(y_recall_val[indices].flatten())
            w_recall_pred.append(gamma.cpu().numpy().flatten())
        if w_recall_true:
            wt = np.concatenate(w_recall_true)
            wp = np.concatenate(w_recall_pred)
            w_r2 = r2_score(wt, wp)
            w_mae = mean_absolute_error(wt, wp)
            weather_r2[weather_cat] = {"r2": w_r2, "mae": w_mae, "n": len(wt)}
            print(f"  {weather_cat:8s}  R2={w_r2:.4f}  MAE={w_mae:.5f}  n={len(wt)}")

    print(f"\n  LSTM_nano (gap target) R2    = {PREV_NANO_GAP['r2']}")
    print(f"  LSTM_nano (recall target) R2 = {overall_recall['r2']:.4f}")

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
            rm = all_recall_metrics[seq_name]
            dm = all_den_metrics[seq_name]

            ax = axes[row, 0]
            ax.plot(d["frames_recall"], d["recall_true"], "b-", alpha=0.7, lw=0.8,
                    label="Actual")
            ax.plot(d["frames_recall"], d["recall_pred"], "r-", alpha=0.7, lw=0.8,
                    label="Predicted")
            ax.fill_between(
                d["frames_recall"],
                d["recall_pred"] - np.sqrt(np.clip(d["recall_epis"], 0, None)),
                d["recall_pred"] + np.sqrt(np.clip(d["recall_epis"], 0, None)),
                alpha=0.25, color="orange", label="Epistemic",
            )
            ax.set_ylabel("Nano Recall")
            ax.set_title(f"{seq_name} [{weather}] nano_recall "
                         f"(MAE={rm['mae']:.4f}, R2={rm['r2']:.3f})")
            ax.legend(fontsize=7, loc="upper right")

            ax = axes[row, 1]
            ax.plot(d["frames_den"], d["den_true"], "b-", alpha=0.7, lw=0.8,
                    label="Actual")
            ax.plot(d["frames_den"], d["den_pred"], "r-", alpha=0.7, lw=0.8,
                    label="Predicted")
            ax.set_ylabel("num_gt")
            ax.set_title(f"{seq_name} [{weather}] num_gt "
                         f"(MAE={dm['mae']:.4f}, R2={dm['r2']:.3f})")
            ax.legend(fontsize=7, loc="upper right")

            if row == n_rows - 1:
                axes[row, 0].set_xlabel("Frame index (local)")
                axes[row, 1].set_xlabel("Frame index (local)")

        fig.suptitle("LSTM_nano recall target (23 feat: 14 image + 9 nano det) "
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
                ("train_recall_loss", "Train recall", "blue"),
                ("val_recall_loss", "Val recall", "red"),
                ("train_density_loss", "Train density", "blue"),
                ("val_density_loss", "Val density", "red"),
            ]:
                if col in logs.columns:
                    data = logs[["epoch", col]].dropna()
                    if len(data) > 0:
                        epoch_mean = data.groupby("epoch")[col].mean()
                        ls = ("-" if "total" in label
                              else ("--" if "recall" in label else ":"))
                        ax.plot(epoch_mean.index, epoch_mean.values,
                                color=color, ls=ls, alpha=0.7, label=label)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training Curves")
            ax.legend(fontsize=7)
        else:
            axes[0, 0].text(0.5, 0.5, "No log CSV found", ha="center")

        ax = axes[0, 1]
        ax.scatter(all_recall_true, all_recall_pred, alpha=0.1, s=4)
        lims = [min(all_recall_true.min(), all_recall_pred.min()),
                max(all_recall_true.max(), all_recall_pred.max())]
        ax.plot(lims, lims, "k--", lw=0.5)
        ax.set_xlabel("Actual nano_recall")
        ax.set_ylabel("Predicted nano_recall")
        ax.set_title(f"Nano Recall (R2={overall_recall['r2']:.3f})")

        ax = axes[1, 0]
        ax.scatter(all_den_true, all_den_pred, alpha=0.1, s=4)
        lims = [min(all_den_true.min(), all_den_pred.min()),
                max(all_den_true.max(), all_den_pred.max())]
        ax.plot(lims, lims, "k--", lw=0.5)
        ax.set_xlabel("Actual num_gt")
        ax.set_ylabel("Predicted num_gt")
        ax.set_title(f"Num GT (R2={overall_den['r2']:.3f})")

        ax = axes[1, 1]
        seq_names_plot = [s for s in VAL_SEQUENCES if s in all_recall_metrics]
        x = np.arange(len(seq_names_plot))
        w = 0.35
        recall_r2 = [all_recall_metrics[s]["r2"] for s in seq_names_plot]
        den_r2 = [all_den_metrics[s]["r2"] for s in seq_names_plot]
        ax.bar(x - w/2, recall_r2, w, label="nano_recall R2", color="steelblue")
        ax.bar(x + w/2, den_r2, w, label="num_gt R2", color="coral")
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

        # Page 3: Metrics table + comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14),
                                         gridspec_kw={"height_ratios": [3, 1]})

        ax1.axis("off")
        rows = []
        for seq in VAL_SEQUENCES + ["OVERALL"]:
            rm = all_recall_metrics.get(seq)
            dm = all_den_metrics.get(seq)
            if rm and dm:
                w_label = seq_weather.get(seq, "")
                rows.append([
                    seq, w_label,
                    f"{rm['mae']:.4f}", f"{rm['r2']:.4f}", f"{rm['rho']:.4f}",
                    f"{dm['mae']:.4f}", f"{dm['r2']:.4f}", f"{dm['rho']:.4f}",
                ])
        rows.append([
            "BASELINE", "(mean)",
            f"{baseline_recall['mae']:.4f}", f"{baseline_recall['r2']:.4f}",
            f"{baseline_recall['rho']:.4f}",
            f"{baseline_den['mae']:.4f}", f"{baseline_den['r2']:.4f}",
            f"{baseline_den['rho']:.4f}",
        ])

        col_labels = ["Sequence", "Weather",
                       "Recall MAE", "Recall R2", "Recall rho",
                       "GT MAE", "GT R2", "GT rho"]
        table = ax1.table(cellText=rows, colLabels=col_labels,
                          loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.4)

        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")

        ax1.set_title("LSTM_nano recall target -- Validation Metrics "
                       "(23 features: 14 image + 9 nano)",
                       fontsize=12, fontweight="bold", pad=20)

        # Bottom: gap-target vs recall-target comparison
        ax2.axis("off")
        comp_rows = [
            ["LSTM_nano (gap target)", "recall_gap",
             f"{PREV_NANO_GAP['r2']:.4f}",
             f"{PREV_NANO_GAP['mae']:.4f}",
             f"{PREV_NANO_GAP['rho']:.4f}"],
            ["LSTM_nano (recall target)", "nano_recall",
             f"{overall_recall['r2']:.4f}",
             f"{overall_recall['mae']:.4f}",
             f"{overall_recall['rho']:.4f}"],
        ]

        # Add per-weather rows for recall target
        for wcat in ["sunny", "cloudy", "night", "rainy"]:
            wr = weather_r2.get(wcat, {})
            comp_rows.append([
                f"  {wcat}", "nano_recall",
                f"{wr.get('r2', float('nan')):.4f}",
                f"{wr.get('mae', float('nan')):.5f}",
                str(wr.get("n", 0)),
            ])

        comp_labels = ["Experiment", "Target", "R2", "MAE", "rho / n"]
        comp_table = ax2.table(cellText=comp_rows, colLabels=comp_labels,
                               loc="center", cellLoc="center")
        comp_table.auto_set_font_size(False)
        comp_table.set_fontsize(9)
        comp_table.scale(0.8, 1.8)

        for j in range(len(comp_labels)):
            comp_table[0, j].set_facecolor("#C44472")
            comp_table[0, j].set_text_props(color="white", fontweight="bold")

        ax2.set_title("Gap Target vs Recall Target Comparison",
                       fontsize=11, fontweight="bold", pad=20)

        fig.suptitle("Page 3: Metrics & Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"\nPDF saved -> {PDF_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
