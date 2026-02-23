import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from collections import defaultdict
import pickle

#loss functions
def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    """
    Negative Log-Likelihood of Normal-Inverse-Gamma distribution
    From Amini's repo - converted to PyTorch
    """
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
    """
    Evidence Regularizer - penalizes high evidence when prediction is wrong
    """
    error = torch.abs(y - gamma)
    evi = 2 * v + alpha
    reg = error * evi

    return torch.mean(reg) if reduce else reg


def EvidentialLoss(y_true, gamma, v, alpha, beta, coeff=1.0):
    """
    Combined loss: NLL + coeff * Regularizer
    """
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)

    return loss_nll + coeff * loss_reg


#the model
class EvidentialLSTM(L.LightningModule):

    def __init__(self, input_size=22, hidden_size=128, num_layers=2,
                 dropout=0.2, horizon=10, lr=1e-3, coeff=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Two-layer head for better expressiveness
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon * 4),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        # Use last layer's hidden state
        hidden = h_n[-1]

        output = self.head(hidden)
        output = output.view(-1, self.horizon, 4)

        # Apply constraints for valid NIG parameters
        gamma = output[..., 0]                      # prediction mean
        v = F.softplus(output[..., 1])              # ν > 0
        alpha = F.softplus(output[..., 2]) + 1      # α > 1
        beta = F.softplus(output[..., 3])           # β > 0

        return gamma, v, alpha, beta

    def training_step(self, batch, batch_idx):
        x, y = batch
        gamma, v, alpha, beta = self(x)
        loss = EvidentialLoss(y, gamma, v, alpha, beta, coeff=self.hparams.coeff)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        gamma, v, alpha, beta = self(x)
        loss = EvidentialLoss(y, gamma, v, alpha, beta, coeff=self.hparams.coeff)
        mse = F.mse_loss(gamma, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mse', mse, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def get_uncertainty(self, v, alpha, beta):
        """Extract uncertainties from NIG parameters (Eq. 5 from Amini)"""
        aleatoric = beta / (alpha - 1)
        epistemic = beta / (v * (alpha - 1))
        return aleatoric, epistemic


# ── Configuration ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
FEAT_CSV = ROOT / "berry_frame_features.csv"
RES_CSV = ROOT / "cherry_results.csv"
CRAFTED_CSV = ROOT / "crafted_dataset.csv"
SCALER_PATH = ROOT / "feature_scaler.pkl"
CKPT_PATH = ROOT / "best_model.ckpt"

T_CONTEXT = 20
T_HORIZON = 10

RUNTIME_FEATURES = [
    "mean_brightness", "brightness_std", "dark_channel", "dynamic_range",
    "weber_contrast", "blur_estimate", "spatial_freq_low_ratio",
    "grad_mag_mean", "grad_mag_var", "histogram_bimodality",
    "backlight_ratio", "edge_density",
    "optical_flow_mag", "optical_flow_var", "interframe_brightness_change",
    "brightness_var_5f", "brightness_var_10f", "brightness_var_30f",
    "flow_mean_5f", "flow_mean_10f", "flow_mean_15f", "flow_mean_30f",
]

VAL_SEQUENCES = [
    "uav0000117_02622_v",
    "uav0000268_05773_v",
    "uav0000289_00001_v",
    "uav0000339_00001_v",
]


def create_crafted_dataset():
    if CRAFTED_CSV.exists():
        print(f"Loading cached {CRAFTED_CSV}")
        return pd.read_csv(CRAFTED_CSV)

    print("Building crafted dataset...")
    feat = pd.read_csv(FEAT_CSV)
    res = pd.read_csv(RES_CSV)

    # Extract nano recall and gt_count
    nano = res[res["model"] == "yolo11n.pt"][["sequence", "frame_id", "recall", "tp", "fn"]].copy()
    nano["gt_count"] = nano["tp"] + nano["fn"]
    nano = nano.rename(columns={"recall": "recall_nano"})

    # Extract medium recall
    medium = res[res["model"] == "yolo11m.pt"][["sequence", "frame_id", "recall"]].copy()
    medium = medium.rename(columns={"recall": "recall_medium"})

    # Merge nano and medium first to get recall_gap
    merged = nano[["sequence", "frame_id", "recall_nano", "gt_count"]].merge(
        medium, on=["sequence", "frame_id"]
    )
    merged["recall_gap"] = merged["recall_medium"] - merged["recall_nano"]

    # Then merge with features
    df = feat.merge(
        merged[["sequence", "frame_id", "recall_gap", "gt_count"]],
        on=["sequence", "frame_id"],
    )

    # Filter frames with enough ground-truth objects
    df = df[df["gt_count"] >= 3]

    # Keep only needed columns
    keep_cols = ["sequence", "frame_id", "recall_gap"] + RUNTIME_FEATURES
    df = df[keep_cols]

    # Drop rows where any runtime feature is NaN
    df = df.dropna(subset=RUNTIME_FEATURES).reset_index(drop=True)

    df.to_csv(CRAFTED_CSV, index=False)
    print(f"Saved {CRAFTED_CSV} ({len(df)} rows)")
    return df


def build_sliding_windows(df, feature_cols, target_col="recall_gap"):
    X_list, y_list, seq_ids = [], [], []
    total_window = T_CONTEXT + T_HORIZON

    for seq_name, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        features = sdf[feature_cols].values
        targets = sdf[target_col].values

        n = len(sdf)
        for i in range(n - total_window + 1):
            X_list.append(features[i : i + T_CONTEXT])
            y_list.append(targets[i + T_CONTEXT : i + T_CONTEXT + T_HORIZON])
            seq_ids.append((seq_name, i))

    X_all = np.array(X_list, dtype=np.float32)
    y_all = np.array(y_list, dtype=np.float32)
    return X_all, y_all, seq_ids


def compute_metrics(y_true, y_pred, label=""):
    """Compute and print regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:30s}  MSE={mse:.5f}  RMSE={rmse:.5f}  MAE={mae:.5f}  R2={r2:.4f}")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


# ── Data Pipeline ────────────────────────────────────────────────
if __name__ == "__main__":
    df = create_crafted_dataset()

    # Fit scaler on train sequences only, save for deployment
    train_mask = ~df["sequence"].isin(VAL_SEQUENCES)
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, RUNTIME_FEATURES])

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {SCALER_PATH}")

    df[RUNTIME_FEATURES] = scaler.transform(df[RUNTIME_FEATURES])

    # Build sliding windows per split
    train_df = df[train_mask].copy()
    val_df = df[df["sequence"].isin(VAL_SEQUENCES)].copy()

    X_train, y_train, train_seq_ids = build_sliding_windows(train_df, RUNTIME_FEATURES)
    X_val, y_val, val_seq_ids = build_sliding_windows(val_df, RUNTIME_FEATURES)

    print(f"Train: {X_train.shape[0]} windows | Val: {X_val.shape[0]} windows")
    print(f"X shape: {X_train.shape}, y shape: {y_train.shape}")

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # ── Train ────────────────────────────────────────────────────
    model = EvidentialLSTM(
        input_size=len(RUNTIME_FEATURES),
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        horizon=T_HORIZON,
        lr=1e-3,
        coeff=0.01,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, mode="min", verbose=True),
        ModelCheckpoint(
            dirpath=str(ROOT),
            filename="best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=150,
        callbacks=callbacks,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint
    best_path = callbacks[1].best_model_path
    if best_path:
        print(f"\nLoading best checkpoint: {best_path}")
        model = EvidentialLSTM.load_from_checkpoint(best_path)
    print(f"Best val_loss: {callbacks[1].best_model_score:.5f}")

    # ── Evaluation ───────────────────────────────────────────────
    device = next(model.parameters()).device
    model.eval()

    print("\n" + "=" * 70)
    print("QUANTITATIVE EVALUATION")
    print("=" * 70)

    all_metrics = {}
    fig, axes = plt.subplots(len(VAL_SEQUENCES), 2, figsize=(18, 4 * len(VAL_SEQUENCES)))

    for row, seq_name in enumerate(VAL_SEQUENCES):
        indices = [i for i, (s, _) in enumerate(val_seq_ids) if s == seq_name]
        if not indices:
            continue

        X_seq = torch.from_numpy(X_val[indices]).to(device)
        y_seq = y_val[indices]

        with torch.no_grad():
            gamma, v, alpha, beta = model(X_seq)
            preds = gamma.cpu().numpy()
            aleatoric, epistemic = model.get_uncertainty(v, alpha, beta)
            aleatoric = aleatoric.cpu().numpy()
            epistemic = epistemic.cpu().numpy()

        # Per-window metrics (no averaging tricks)
        seq_metrics = compute_metrics(
            y_seq.flatten(), preds.flatten(), label=seq_name
        )
        all_metrics[seq_name] = seq_metrics

        # Average overlapping predictions per frame for plotting
        pred_at = defaultdict(list)
        epis_at = defaultdict(list)
        alea_at = defaultdict(list)
        true_at = {}

        for w, idx in enumerate(indices):
            window_start = idx - indices[0]
            for h in range(T_HORIZON):
                frame = window_start + T_CONTEXT + h
                pred_at[frame].append(preds[w, h])
                epis_at[frame].append(epistemic[w, h])
                alea_at[frame].append(aleatoric[w, h])
                true_at[frame] = y_seq[w, h]

        frames = sorted(pred_at.keys())
        y_true = np.array([true_at[f] for f in frames])
        y_pred = np.array([np.mean(pred_at[f]) for f in frames])
        epis = np.array([np.mean(epis_at[f]) for f in frames])
        alea = np.array([np.mean(alea_at[f]) for f in frames])

        # Left: predictions vs actual with epistemic uncertainty band
        ax_pred = axes[row, 0]
        ax_pred.plot(frames, y_true, "b-", alpha=0.7, linewidth=0.8, label="Actual recall gap")
        ax_pred.plot(frames, y_pred, "r-", alpha=0.7, linewidth=0.8, label="Predicted")
        ax_pred.fill_between(
            frames,
            y_pred - np.sqrt(np.clip(epis, 0, None)),
            y_pred + np.sqrt(np.clip(epis, 0, None)),
            alpha=0.25, color="orange", label="Epistemic +/- sigma",
        )
        ax_pred.set_ylabel("Recall Gap")
        ax_pred.set_title(f"{seq_name}  (MAE={seq_metrics['mae']:.4f}, R2={seq_metrics['r2']:.3f})")
        ax_pred.legend(fontsize=7, loc="upper right")

        # Right: uncertainty breakdown
        ax_unc = axes[row, 1]
        ax_unc.plot(frames, alea, "-", alpha=0.7, label="Aleatoric")
        ax_unc.plot(frames, epis, "-", alpha=0.7, label="Epistemic")
        ax_unc.set_ylabel("Uncertainty")
        ax_unc.set_title(f"{seq_name} — Uncertainty")
        ax_unc.legend(fontsize=7, loc="upper right")

        if row == len(VAL_SEQUENCES) - 1:
            ax_pred.set_xlabel("Frame index (local)")
            ax_unc.set_xlabel("Frame index (local)")

    # Overall metrics across all val sequences
    print("-" * 70)
    all_true_list, all_pred_list = [], []
    for seq_name in VAL_SEQUENCES:
        indices = [i for i, (s, _) in enumerate(val_seq_ids) if s == seq_name]
        if indices:
            all_true_list.append(y_val[indices].flatten())
            X_seq = torch.from_numpy(X_val[indices]).to(device)
            with torch.no_grad():
                gamma, _, _, _ = model(X_seq)
                all_pred_list.append(gamma.cpu().numpy().flatten())

    overall = compute_metrics(
        np.concatenate(all_true_list), np.concatenate(all_pred_list), label="OVERALL"
    )
    all_metrics["OVERALL"] = overall

    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(ROOT / "lstm_metrics.csv")
    print(f"\nMetrics saved to {ROOT / 'lstm_metrics.csv'}")

    plt.tight_layout()
    plt.savefig(ROOT / "lstm_predictions.pdf", dpi=150, bbox_inches="tight")
    print(f"Plots saved to {ROOT / 'lstm_predictions.pdf'}")
    plt.close()
