import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.tuner import Tuner
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

#loss functions
def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    """
    Negative Log-Likelihood of Normal-Inverse-Gamma distribution
    From Amini's repo - converted to PyTorch
    """
    twoBlambda = 2 * beta * (1 + v)

    nll = (
        0.5 * torch.log(torch.tensor(np.pi) / v)
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

    def __init__(self, input_size=22, hidden_size=64, horizon=10, lr=1e-3, coeff=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Output 4 parameters per horizon step: γ, ν, α, β
        self.fc = nn.Linear(hidden_size, horizon * 4)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        hidden = h_n.squeeze(0)

        output = self.fc(hidden)
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
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        gamma, v, alpha, beta = self(x)
        loss = EvidentialLoss(y, gamma, v, alpha, beta, coeff=self.hparams.coeff)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

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


# ── Data Pipeline ────────────────────────────────────────────────
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

# ── Train ────────────────────────────────────────────────────────
model = EvidentialLSTM(input_size=22, horizon=T_HORIZON, coeff=0.01)
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, train_loader, val_loader)


# ── Evaluation ───────────────────────────────────────────────────
# Overlapping-window averaging below is a plotting convenience only.
# At deployment each MPC cycle produces one forward pass — no averaging.
model.eval()

fig, axes = plt.subplots(len(VAL_SEQUENCES), 2, figsize=(18, 4 * len(VAL_SEQUENCES)))

for row, seq_name in enumerate(VAL_SEQUENCES):
    # Gather all windows for this sequence
    indices = [i for i, (s, _) in enumerate(val_seq_ids) if s == seq_name]
    if not indices:
        continue

    X_seq = torch.from_numpy(X_val[indices])
    y_seq = y_val[indices]

    with torch.no_grad():
        gamma, v, alpha, beta = model(X_seq)
        preds = gamma.numpy()
        aleatoric, epistemic = model.get_uncertainty(v, alpha, beta)
        aleatoric = aleatoric.numpy()
        epistemic = epistemic.numpy()

    # Average overlapping predictions per frame (plot convenience only)
    from collections import defaultdict
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
        y_pred - np.sqrt(epis),
        y_pred + np.sqrt(epis),
        alpha=0.25, color="orange", label="Epistemic ±σ",
    )
    ax_pred.set_ylabel("Recall Gap")
    ax_pred.set_title(f"{seq_name} — Predictions")
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

plt.tight_layout()
plt.savefig(ROOT / "lstm_predictions.pdf", dpi=150, bbox_inches="tight")
plt.show()
