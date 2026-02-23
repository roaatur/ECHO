"""Single-task evidential LSTM — density head zeroed out (lambda_density=0).

Identical to js8 except the density loss is multiplied by 0, isolating the
effect of the auxiliary num_gt task on recall_gap prediction.
"""

import argparse
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
ROOT = Path(__file__).resolve().parent.parent          # yolo_runs_on_uavdt/
JAYS = ROOT / "jays"

# Features
FEAT_CSV_UAVDT = JAYS / "jays_uavdt_all50_features.csv"
FEAT_CSV_VISDRONE = JAYS / "jays_visdrone_features.csv"

# Results
RES_CSV_UAVDT = ROOT / "uavdt_all50_results.csv"
RES_CSV_VISDRONE = ROOT / "visdrone_classagnostic_results.csv"

# Outputs
SCALER_PATH = JAYS / "js9_feature_scaler.pkl"
DENSITY_MAX_PATH = JAYS / "js9_density_max.pkl"
CKPT_DIR = JAYS
LOG_DIR = JAYS / "js9_logs"
PDF_PATH = JAYS / "js9_single_task_results.pdf"

T_CONTEXT = 20
T_HORIZON = 10

STATIC_FEATURES = [
    "rms_contrast", "weber_contrast", "transmission_estimate", "saturation_std",
    "brightness_std", "mean_brightness", "mean_saturation", "local_entropy_std",
    "dark_channel_std", "dark_channel_mean", "overexposed_ratio", "underexposed_ratio",
    "high_freq_energy_ratio", "colorfulness",
]
TEMPORAL_FEATURES = [
    "optical_flow_mag_mean", "optical_flow_mag_std", "optical_flow_angle_var",
    "frame_difference_mean", "SSIM_prev_frame", "histogram_intersection",
    "brightness_delta", "edge_density_delta", "flow_coherence", "scene_cut_score",
]
FEATURES = STATIC_FEATURES + TEMPORAL_FEATURES  # 24 total

UAVDT_VAL = ["M0801", "M1101", "M1009", "M1401"]

# js5 results for comparison (from js5 run output — 20 seq training)
JS5_METRICS = {
    "M0801": {"gap_mae": 0.63888, "gap_r2": -4.9355},
    "M1101": {"gap_mae": 0.26743, "gap_r2": -6.5659},
    "M1009": {"gap_mae": 0.17757, "gap_r2": -29.7761},
    "M1401": {"gap_mae": 0.14914, "gap_r2": -2.5037},
}

# js8 results for comparison (from js8 run output — 123 seq, lambda=0.1)
JS8_METRICS = {
    "M0801": {"gap_mae": 0.62717, "gap_r2": -4.7302},
    "M1101": {"gap_mae": 0.18834, "gap_r2": -3.0723},
    "M1009": {"gap_mae": 0.06337, "gap_r2": -5.2823},
    "M1401": {"gap_mae": 0.09074, "gap_r2": -0.3892},
}


# ── Evidential loss (copied verbatim from js5/lstm.py) ────────────
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

    def __init__(self, input_size=24, hidden_size=64, horizon=10, lr=1e-3,
                 coeff=0.1, lambda_density=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True)
        self.head_gap = nn.Linear(hidden_size, horizon * 4)
        self.head_density = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        hidden = h_n.squeeze(0)

        gap_out = self.head_gap(hidden).view(-1, self.hparams.horizon, 4)
        gamma = gap_out[..., 0]
        v = F.softplus(gap_out[..., 1])
        alpha = F.softplus(gap_out[..., 2]) + 1
        beta = F.softplus(gap_out[..., 3])

        density = F.softplus(self.head_density(hidden))

        return gamma, v, alpha, beta, density

    def training_step(self, batch, batch_idx):
        x, y_gap, y_density = batch
        gamma, v, alpha, beta, density = self(x)

        loss_gap = EvidentialLoss(y_gap, gamma, v, alpha, beta,
                                  coeff=self.hparams.coeff)
        loss_density = F.mse_loss(density, y_density)
        loss = loss_gap + self.hparams.lambda_density * loss_density

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_gap_loss", loss_gap)
        self.log("train_density_loss", loss_density)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_gap, y_density = batch
        gamma, v, alpha, beta, density = self(x)

        loss_gap = EvidentialLoss(y_gap, gamma, v, alpha, beta,
                                  coeff=self.hparams.coeff)
        loss_density = F.mse_loss(density, y_density)
        loss = loss_gap + self.hparams.lambda_density * loss_density

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_gap_loss", loss_gap)
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
def create_dataset():
    """Load features + results from both UAVDT and VisDrone, merge, smooth, filter."""
    # Load and concatenate features
    feat_uavdt = pd.read_csv(FEAT_CSV_UAVDT)
    feat_visdrone = pd.read_csv(FEAT_CSV_VISDRONE)
    feat = pd.concat([feat_uavdt, feat_visdrone], ignore_index=True)
    print(f"Features loaded: UAVDT {len(feat_uavdt)} rows ({feat_uavdt['sequence'].nunique()} seq) "
          f"+ VisDrone {len(feat_visdrone)} rows ({feat_visdrone['sequence'].nunique()} seq)")
    print(f"Combined features: {len(feat)} rows, {feat['sequence'].nunique()} sequences")

    # Load and concatenate results
    res_uavdt = pd.read_csv(RES_CSV_UAVDT)
    res_visdrone = pd.read_csv(RES_CSV_VISDRONE)
    res = pd.concat([res_uavdt, res_visdrone], ignore_index=True)
    print(f"Results loaded: UAVDT {len(res_uavdt)} rows + VisDrone {len(res_visdrone)} rows")

    # Extract nano recall and num_gt
    nano = res[res["model"] == "yolo11n.pt"][["sequence", "frame_id", "recall", "num_gt"]].copy()
    nano = nano.rename(columns={"recall": "recall_nano"})

    # Extract medium recall
    medium = res[res["model"] == "yolo11m.pt"][["sequence", "frame_id", "recall"]].copy()
    medium = medium.rename(columns={"recall": "recall_medium"})

    # Merge nano + medium → recall_gap
    merged = nano.merge(medium, on=["sequence", "frame_id"])
    merged["recall_gap"] = merged["recall_medium"] - merged["recall_nano"]

    # Merge with features
    df = feat.merge(
        merged[["sequence", "frame_id", "recall_gap", "num_gt"]],
        on=["sequence", "frame_id"],
    )

    seqs_with_features = df["sequence"].nunique()
    print(f"Sequences after merge: {seqs_with_features}")
    print(f"  UAVDT: {df[~df['sequence'].str.startswith('uav')]['sequence'].nunique()}")
    print(f"  VisDrone: {df[df['sequence'].str.startswith('uav')]['sequence'].nunique()}")

    # Smooth recall_gap per sequence
    df["recall_gap"] = df.groupby("sequence")["recall_gap"].transform(
        lambda s: s.rolling(10, center=True, min_periods=1).mean()
    )

    # Filter
    df = df[df["num_gt"] >= 1]
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    print(f"After filtering: {len(df)} rows, {df['sequence'].nunique()} sequences")
    return df


def build_sliding_windows(df, feature_cols):
    """Build sliding windows for both targets."""
    X_list, y_gap_list, y_density_list, seq_ids = [], [], [], []
    total_window = T_CONTEXT + T_HORIZON

    for seq_name, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        features = sdf[feature_cols].values
        gap_targets = sdf["recall_gap"].values
        density_targets = sdf["num_gt"].values

        n = len(sdf)
        for i in range(n - total_window + 1):
            X_list.append(features[i : i + T_CONTEXT])
            y_gap_list.append(gap_targets[i + T_CONTEXT : i + T_CONTEXT + T_HORIZON])
            y_density_list.append(density_targets[i + T_CONTEXT : i + T_CONTEXT + T_HORIZON])
            seq_ids.append((seq_name, i))

    X = np.array(X_list, dtype=np.float32)
    y_gap = np.array(y_gap_list, dtype=np.float32)
    y_density = np.array(y_density_list, dtype=np.float32)
    return X, y_gap, y_density, seq_ids


def compute_metrics(y_true, y_pred, label=""):
    """Compute and print regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    print(f"  {label:40s}  MAE={mae:.5f}  R2={r2:.4f}  rho={rho:.4f}")
    return {"mae": mae, "r2": r2, "rho": rho}


# ── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_density", type=float, default=0.0)
    args = parser.parse_args()

    print(f"lambda_density = {args.lambda_density}")

    # ── Data ──
    df = create_dataset()

    # ── Val split: 4 UAVDT + every 5th VisDrone ──
    all_seqs = sorted(df["sequence"].unique())
    visdrone_seqs = sorted([s for s in all_seqs if s.startswith("uav")])
    visdrone_val = visdrone_seqs[::5]
    val_sequences = UAVDT_VAL + list(visdrone_val)

    train_mask = ~df["sequence"].isin(val_sequences)
    train_df = df[train_mask].copy()
    val_df = df[df["sequence"].isin(val_sequences)].copy()

    print(f"\nTrain sequences ({train_df['sequence'].nunique()}): "
          f"{sorted(train_df['sequence'].unique())[:10]}...")
    print(f"Val sequences ({val_df['sequence'].nunique()}):")
    print(f"  UAVDT val: {UAVDT_VAL}")
    print(f"  VisDrone val ({len(visdrone_val)}): {list(visdrone_val)}")

    # ── Normalize features ──
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES])
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {SCALER_PATH}")

    df[FEATURES] = scaler.transform(df[FEATURES])

    # ── Normalize num_gt ──
    density_max = float(train_df["num_gt"].max())
    with open(DENSITY_MAX_PATH, "wb") as f:
        pickle.dump(density_max, f)
    print(f"Density max = {density_max}, saved to {DENSITY_MAX_PATH}")

    df["num_gt_norm"] = df["num_gt"] / density_max

    # Re-split after normalization
    train_df = df[~df["sequence"].isin(val_sequences)].copy()
    val_df = df[df["sequence"].isin(val_sequences)].copy()

    # ── Sliding windows ──
    X_train, y_gap_train, y_den_train, train_ids = build_sliding_windows(
        train_df, FEATURES)
    X_val, y_gap_val, y_den_val, val_ids = build_sliding_windows(
        val_df, FEATURES)

    # Use normalized density for targets
    _, _, y_den_train_norm, _ = build_sliding_windows(
        train_df.assign(num_gt=train_df["num_gt_norm"]).rename(
            columns={"num_gt": "num_gt"}), FEATURES)
    _, _, y_den_val_norm, _ = build_sliding_windows(
        val_df.assign(num_gt=val_df["num_gt_norm"]).rename(
            columns={"num_gt": "num_gt"}), FEATURES)

    print(f"Train: {X_train.shape[0]} windows | Val: {X_val.shape[0]} windows")
    print(f"X shape: {X_train.shape}, y_gap: {y_gap_train.shape}, y_den: {y_den_train_norm.shape}")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_gap_train),
        torch.from_numpy(y_den_train_norm),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_gap_val),
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
        lambda_density=args.lambda_density,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, mode="min", verbose=True),
        ModelCheckpoint(
            dirpath=str(CKPT_DIR),
            filename="js9_single_task_best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
        ),
    ]

    csv_logger = CSVLogger(save_dir=str(JAYS), name="js9_logs", version=0)

    trainer = L.Trainer(
        max_epochs=150,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        logger=csv_logger,
    )
    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint
    best_path = callbacks[1].best_model_path
    if best_path:
        print(f"\nLoading best checkpoint: {best_path}")
        model = MultiTaskLSTM.load_from_checkpoint(best_path)
    print(f"Best val_loss: {callbacks[1].best_model_score:.5f}")

    # ── Evaluation ─────────────────────────────────────────────────
    device = next(model.parameters()).device
    model.eval()

    # Get weather info for UAVDT sequences
    res_uavdt = pd.read_csv(RES_CSV_UAVDT)
    seq_weather = res_uavdt.drop_duplicates("sequence").set_index("sequence")["weather"].to_dict()

    print("\n" + "=" * 70)
    print("QUANTITATIVE EVALUATION")
    print("=" * 70)

    all_gap_metrics = {}
    all_den_metrics = {}
    val_seq_data = {}

    for seq_name in val_sequences:
        indices = [i for i, (s, _) in enumerate(val_ids) if s == seq_name]
        if not indices:
            continue

        X_seq = torch.from_numpy(X_val[indices]).to(device)
        y_gap_seq = y_gap_val[indices]
        y_den_seq = y_den_val[indices]

        with torch.no_grad():
            gamma, v, alpha, beta, density = model(X_seq)
            gap_preds = gamma.cpu().numpy()
            den_preds_norm = density.cpu().numpy()
            aleatoric, epistemic = model.get_uncertainty(v, alpha, beta)
            epistemic = epistemic.cpu().numpy()

        den_preds = den_preds_norm * density_max

        weather = seq_weather.get(seq_name, "VisDrone")
        print(f"\n  --- {seq_name} ({weather}) ---")
        gap_m = compute_metrics(y_gap_seq.flatten(), gap_preds.flatten(),
                                label=f"{seq_name} recall_gap")
        den_m = compute_metrics(y_den_seq.flatten(), den_preds.flatten(),
                                label=f"{seq_name} num_gt")
        all_gap_metrics[seq_name] = gap_m
        all_den_metrics[seq_name] = den_m

        # Frame-averaged predictions for plotting
        gap_pred_at = defaultdict(list)
        gap_epis_at = defaultdict(list)
        gap_true_at = {}
        den_pred_at = defaultdict(list)
        den_true_at = {}

        for w, idx in enumerate(indices):
            window_start = idx - indices[0]
            for h in range(T_HORIZON):
                frame = window_start + T_CONTEXT + h
                gap_pred_at[frame].append(gap_preds[w, h])
                gap_epis_at[frame].append(epistemic[w, h])
                gap_true_at[frame] = y_gap_seq[w, h]
                den_pred_at[frame].append(den_preds[w, h])
                den_true_at[frame] = y_den_seq[w, h]

        frames = sorted(gap_pred_at.keys())
        val_seq_data[seq_name] = {
            "frames": frames,
            "gap_true": np.array([gap_true_at[f] for f in frames]),
            "gap_pred": np.array([np.mean(gap_pred_at[f]) for f in frames]),
            "gap_epis": np.array([np.mean(gap_epis_at[f]) for f in frames]),
            "den_true": np.array([den_true_at[f] for f in frames]),
            "den_pred": np.array([np.mean(den_pred_at[f]) for f in frames]),
        }

    # Overall val metrics
    print("\n" + "-" * 70)
    all_gap_true, all_gap_pred = [], []
    all_den_true, all_den_pred = [], []
    for seq_name in val_sequences:
        indices = [i for i, (s, _) in enumerate(val_ids) if s == seq_name]
        if not indices:
            continue
        X_seq = torch.from_numpy(X_val[indices]).to(device)
        with torch.no_grad():
            gamma, _, _, _, density = model(X_seq)
        all_gap_true.append(y_gap_val[indices].flatten())
        all_gap_pred.append(gamma.cpu().numpy().flatten())
        all_den_true.append(y_den_val[indices].flatten())
        all_den_pred.append((density.cpu().numpy() * density_max).flatten())

    all_gap_true = np.concatenate(all_gap_true)
    all_gap_pred = np.concatenate(all_gap_pred)
    all_den_true = np.concatenate(all_den_true)
    all_den_pred = np.concatenate(all_den_pred)

    overall_gap = compute_metrics(all_gap_true, all_gap_pred, label="OVERALL recall_gap")
    overall_den = compute_metrics(all_den_true, all_den_pred, label="OVERALL num_gt")
    all_gap_metrics["OVERALL"] = overall_gap
    all_den_metrics["OVERALL"] = overall_den

    # Baseline
    baseline_gap_pred = np.full_like(all_gap_true, all_gap_true.mean())
    baseline_den_pred = np.full_like(all_den_true, all_den_true.mean())
    baseline_gap = compute_metrics(all_gap_true, baseline_gap_pred,
                                   label="BASELINE recall_gap (mean)")
    baseline_den = compute_metrics(all_den_true, baseline_den_pred,
                                   label="BASELINE num_gt (mean)")

    # ── js5 vs js8 vs js9 comparison ─────────────────────────────
    print("\n" + "=" * 80)
    print("js5 (20 seq) vs js8 (lambda=0.1) vs js9 (lambda=0) — Gap only")
    print("=" * 80)
    print(f"  {'Sequence':<10} {'Metric':<10} {'js5':>10} {'js8':>10} {'js9':>10}  best")
    print("  " + "-" * 62)
    for seq in UAVDT_VAL:
        js5 = JS5_METRICS.get(seq, {})
        js8 = JS8_METRICS.get(seq, {})
        js9_gap = all_gap_metrics.get(seq, {})
        if js5 and js9_gap:
            for metric, key5, key8, key9 in [
                ("Gap MAE", "gap_mae", "gap_mae", "mae"),
                ("Gap R2", "gap_r2", "gap_r2", "r2"),
            ]:
                v5 = js5.get(key5, float("nan"))
                v8 = js8.get(key8, float("nan"))
                v9 = js9_gap.get(key9, float("nan"))
                vals = {"js5": v5, "js8": v8, "js9": v9}
                if "R2" in metric:
                    best = max(vals, key=vals.get)
                else:
                    best = min(vals, key=vals.get)
                print(f"  {seq:<10} {metric:<10} {v5:>10.4f} {v8:>10.4f} {v9:>10.4f}  {best}")
            print("  " + "-" * 62)

    # ── PDF ────────────────────────────────────────────────────────
    with PdfPages(str(PDF_PATH)) as pdf:
        # Page 1: UAVDT val sequences
        uavdt_val_present = [s for s in UAVDT_VAL if s in val_seq_data]
        if uavdt_val_present:
            n_rows = len(uavdt_val_present)
            fig, axes = plt.subplots(n_rows, 2, figsize=(18, 4 * n_rows))
            if n_rows == 1:
                axes = axes[np.newaxis, :]
            for row, seq_name in enumerate(uavdt_val_present):
                d = val_seq_data[seq_name]
                weather = seq_weather.get(seq_name, "unknown")
                gm = all_gap_metrics[seq_name]
                dm = all_den_metrics[seq_name]

                ax = axes[row, 0]
                ax.plot(d["frames"], d["gap_true"], "b-", alpha=0.7, lw=0.8, label="Actual")
                ax.plot(d["frames"], d["gap_pred"], "r-", alpha=0.7, lw=0.8, label="Predicted")
                ax.fill_between(
                    d["frames"],
                    d["gap_pred"] - np.sqrt(np.clip(d["gap_epis"], 0, None)),
                    d["gap_pred"] + np.sqrt(np.clip(d["gap_epis"], 0, None)),
                    alpha=0.25, color="orange", label="Epistemic",
                )
                ax.set_ylabel("Recall Gap")
                ax.set_title(f"{seq_name} [{weather}] recall_gap "
                             f"(MAE={gm['mae']:.4f}, R2={gm['r2']:.3f})")
                ax.legend(fontsize=7, loc="upper right")

                ax = axes[row, 1]
                ax.plot(d["frames"], d["den_true"], "b-", alpha=0.7, lw=0.8, label="Actual")
                ax.plot(d["frames"], d["den_pred"], "r-", alpha=0.7, lw=0.8, label="Predicted")
                ax.set_ylabel("num_gt")
                ax.set_title(f"{seq_name} [{weather}] num_gt "
                             f"(MAE={dm['mae']:.4f}, R2={dm['r2']:.3f})")
                ax.legend(fontsize=7, loc="upper right")

                if row == n_rows - 1:
                    axes[row, 0].set_xlabel("Frame index (local)")
                    axes[row, 1].set_xlabel("Frame index (local)")

            fig.suptitle("Page 1: UAVDT Validation Sequences", fontsize=14, fontweight="bold")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        # Page 2: First 4 VisDrone val sequences
        visdrone_val_present = [s for s in visdrone_val if s in val_seq_data][:4]
        if visdrone_val_present:
            n_rows = len(visdrone_val_present)
            fig, axes = plt.subplots(n_rows, 2, figsize=(18, 4 * n_rows))
            if n_rows == 1:
                axes = axes[np.newaxis, :]
            for row, seq_name in enumerate(visdrone_val_present):
                d = val_seq_data[seq_name]
                gm = all_gap_metrics[seq_name]
                dm = all_den_metrics[seq_name]

                ax = axes[row, 0]
                ax.plot(d["frames"], d["gap_true"], "b-", alpha=0.7, lw=0.8, label="Actual")
                ax.plot(d["frames"], d["gap_pred"], "r-", alpha=0.7, lw=0.8, label="Predicted")
                ax.fill_between(
                    d["frames"],
                    d["gap_pred"] - np.sqrt(np.clip(d["gap_epis"], 0, None)),
                    d["gap_pred"] + np.sqrt(np.clip(d["gap_epis"], 0, None)),
                    alpha=0.25, color="orange", label="Epistemic",
                )
                ax.set_ylabel("Recall Gap")
                ax.set_title(f"{seq_name} recall_gap "
                             f"(MAE={gm['mae']:.4f}, R2={gm['r2']:.3f})")
                ax.legend(fontsize=7, loc="upper right")

                ax = axes[row, 1]
                ax.plot(d["frames"], d["den_true"], "b-", alpha=0.7, lw=0.8, label="Actual")
                ax.plot(d["frames"], d["den_pred"], "r-", alpha=0.7, lw=0.8, label="Predicted")
                ax.set_ylabel("num_gt")
                ax.set_title(f"{seq_name} num_gt "
                             f"(MAE={dm['mae']:.4f}, R2={dm['r2']:.3f})")
                ax.legend(fontsize=7, loc="upper right")

                if row == n_rows - 1:
                    axes[row, 0].set_xlabel("Frame index (local)")
                    axes[row, 1].set_xlabel("Frame index (local)")

            fig.suptitle("Page 2: VisDrone Validation Sequences (sample)", fontsize=14, fontweight="bold")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        # Page 3: Diagnostics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top-left: loss curves
        log_csv = LOG_DIR / "version_0" / "metrics.csv"
        if log_csv.exists():
            logs = pd.read_csv(log_csv)
            ax = axes[0, 0]
            for col, label, color in [
                ("train_loss", "Train total", "blue"),
                ("val_loss", "Val total", "red"),
                ("train_gap_loss", "Train gap", "blue"),
                ("val_gap_loss", "Val gap", "red"),
                ("train_density_loss", "Train density", "blue"),
                ("val_density_loss", "Val density", "red"),
            ]:
                if col in logs.columns:
                    data = logs[["epoch", col]].dropna()
                    if len(data) > 0:
                        epoch_mean = data.groupby("epoch")[col].mean()
                        ls = "-" if "total" in label else ("--" if "gap" in label else ":")
                        ax.plot(epoch_mean.index, epoch_mean.values,
                                color=color, ls=ls, alpha=0.7, label=label)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training Curves")
            ax.legend(fontsize=7)
        else:
            axes[0, 0].text(0.5, 0.5, "No log CSV found", ha="center")

        # Top-right: scatter gap
        ax = axes[0, 1]
        ax.scatter(all_gap_true, all_gap_pred, alpha=0.1, s=4)
        lims = [min(all_gap_true.min(), all_gap_pred.min()),
                max(all_gap_true.max(), all_gap_pred.max())]
        ax.plot(lims, lims, "k--", lw=0.5)
        ax.set_xlabel("Actual recall_gap")
        ax.set_ylabel("Predicted recall_gap")
        ax.set_title(f"Recall Gap (R2={overall_gap['r2']:.3f})")

        # Bottom-left: scatter density
        ax = axes[1, 0]
        ax.scatter(all_den_true, all_den_pred, alpha=0.1, s=4)
        lims = [min(all_den_true.min(), all_den_pred.min()),
                max(all_den_true.max(), all_den_pred.max())]
        ax.plot(lims, lims, "k--", lw=0.5)
        ax.set_xlabel("Actual num_gt")
        ax.set_ylabel("Predicted num_gt")
        ax.set_title(f"Num GT (R2={overall_den['r2']:.3f})")

        # Bottom-right: per-sequence R² bar chart
        ax = axes[1, 1]
        seq_names_plot = [s for s in val_sequences if s in all_gap_metrics]
        x = np.arange(len(seq_names_plot))
        w = 0.35
        gap_r2 = [all_gap_metrics[s]["r2"] for s in seq_names_plot]
        den_r2 = [all_den_metrics[s]["r2"] for s in seq_names_plot]
        ax.bar(x - w/2, gap_r2, w, label="recall_gap R2", color="steelblue")
        ax.bar(x + w/2, den_r2, w, label="num_gt R2", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(seq_names_plot, rotation=90, ha="center", fontsize=6)
        ax.set_ylabel("R2")
        ax.set_title("Per-Sequence R2")
        ax.legend(fontsize=7)
        ax.axhline(0, color="k", lw=0.5, ls="--")

        fig.suptitle("Page 3: Diagnostics", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # Page 4: Metrics table + js5 comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14),
                                         gridspec_kw={"height_ratios": [3, 1]})

        # Top: full metrics table
        ax1.axis("off")
        rows = []
        seq_names_table = [s for s in val_sequences if s in all_gap_metrics]
        for seq in seq_names_table + ["OVERALL"]:
            gm = all_gap_metrics[seq]
            dm = all_den_metrics[seq]
            rows.append([
                seq,
                f"{gm['mae']:.4f}", f"{gm['r2']:.4f}", f"{gm['rho']:.4f}",
                f"{dm['mae']:.4f}", f"{dm['r2']:.4f}", f"{dm['rho']:.4f}",
            ])
        rows.append([
            "BASELINE (mean)",
            f"{baseline_gap['mae']:.4f}", f"{baseline_gap['r2']:.4f}", f"{baseline_gap['rho']:.4f}",
            f"{baseline_den['mae']:.4f}", f"{baseline_den['r2']:.4f}", f"{baseline_den['rho']:.4f}",
        ])

        col_labels = ["Sequence",
                       "Gap MAE", "Gap R2", "Gap rho",
                       "GT MAE", "GT R2", "GT rho"]
        table = ax1.table(cellText=rows, colLabels=col_labels,
                          loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.3)

        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")

        ax1.set_title("All Validation Metrics", fontsize=12, fontweight="bold", pad=20)

        # Bottom: js5 vs js8 vs js9 comparison (Gap only)
        ax2.axis("off")
        comp_rows = []
        for seq in UAVDT_VAL:
            js5 = JS5_METRICS.get(seq, {})
            js8 = JS8_METRICS.get(seq, {})
            js9_gap = all_gap_metrics.get(seq, {})
            if js5 and js9_gap:
                comp_rows.append([
                    seq,
                    f"{js5['gap_mae']:.4f}", f"{js8['gap_mae']:.4f}", f"{js9_gap['mae']:.4f}",
                    f"{js5['gap_r2']:.4f}", f"{js8['gap_r2']:.4f}", f"{js9_gap['r2']:.4f}",
                ])

        comp_labels = ["Sequence",
                        "js5 MAE", "js8 MAE", "js9 MAE",
                        "js5 R2", "js8 R2", "js9 R2"]
        comp_table = ax2.table(cellText=comp_rows, colLabels=comp_labels,
                               loc="center", cellLoc="center")
        comp_table.auto_set_font_size(False)
        comp_table.set_fontsize(8)
        comp_table.scale(1.0, 1.5)

        for j in range(len(comp_labels)):
            comp_table[0, j].set_facecolor("#C44472")
            comp_table[0, j].set_text_props(color="white", fontweight="bold")

        ax2.set_title("js5 (20seq) vs js8 (lambda=0.1) vs js9 (lambda=0) — Gap R2 & MAE",
                       fontsize=11, fontweight="bold", pad=20)

        fig.suptitle("Page 4: Metrics & Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"\nPDF saved to {PDF_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
