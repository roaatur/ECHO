"""Cross-experiment comparison: Image-only vs Both-models vs LSTM_nano vs LSTM_medium.

Loads both trained checkpoints, re-runs inference on the val set,
and produces a 4-page comparison PDF.
"""

import pickle
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
import torch.nn.functional as F
import lightning as L
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

NANO_CSV   = SCRIPT_DIR / "2lstms_nano_data.csv"
MEDIUM_CSV = SCRIPT_DIR / "2lstms_medium_data.csv"

NANO_CKPT       = SCRIPT_DIR / "2lstms_nano_best.ckpt"
NANO_SCALER     = SCRIPT_DIR / "2lstms_nano_scaler.pkl"
NANO_DMAX       = SCRIPT_DIR / "2lstms_nano_density_max.pkl"

MEDIUM_CKPT     = SCRIPT_DIR / "2lstms_medium_best.ckpt"
MEDIUM_SCALER   = SCRIPT_DIR / "2lstms_medium_scaler.pkl"
MEDIUM_DMAX     = SCRIPT_DIR / "2lstms_medium_density_max.pkl"

PDF_PATH = SCRIPT_DIR / "2lstms_comparison.pdf"

T_CONTEXT = 20
T_HORIZON = 10

IMAGE_FEATURES = [
    "gradient_magnitude_std", "brightness_skewness", "overexposed_ratio",
    "contour_count", "wavelet_detail_energy", "high_freq_energy_ratio",
    "mean_brightness", "spectral_rolloff", "brightness_kurtosis",
    "dark_channel_std", "color_channel_corr", "FAST_keypoint_count",
    "ssim_prev", "temporal_diff_mean",
]

DET_SUFFIXES = [
    "num_detections", "mean_confidence", "min_confidence", "confidence_std",
    "mean_box_area", "min_box_area", "box_area_std",
    "detection_density", "edge_box_ratio",
]

NANO_FEATURES   = IMAGE_FEATURES + [f"nano_{s}" for s in DET_SUFFIXES]    # 23
MEDIUM_FEATURES = IMAGE_FEATURES + [f"medium_{s}" for s in DET_SUFFIXES]  # 23

VAL_SEQUENCES = [
    "MVI_39361", "MVI_20011", "MVI_41073", "MVI_39211",
    "MVI_40172", "MVI_40191", "MVI_40712", "MVI_40244",
    "MVI_40992", "MVI_40732", "MVI_39811", "MVI_40775",
    "MVI_63521", "MVI_40892", "MVI_63553",
]

# Hardcoded previous results
PREV_IMAGE_ONLY = {"r2": -0.115, "mae": 0.144, "rho": 0.084}
PREV_BOTH_MODELS = {"r2": 0.165, "mae": 0.123, "rho": 0.530}


# ── Evidential loss (needed for checkpoint loading) ───────────────
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


class MultiTaskLSTM(L.LightningModule):

    def __init__(self, input_size=23, hidden_size=64, horizon=10, lr=1e-3,
                 coeff=0.1, lambda_density=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True)
        self.head_gap = nn.Linear(hidden_size, horizon * 4)
        self.head_density = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        hidden = h_n.squeeze(0)
        gap_out = self.head_gap(hidden).view(-1, self.hparams.horizon, 4)
        gamma = gap_out[..., 0]
        v = F.softplus(gap_out[..., 1])
        alpha = F.softplus(gap_out[..., 2]) + 1
        beta = F.softplus(gap_out[..., 3])
        density = F.softplus(self.head_density(hidden)).squeeze(-1)
        return gamma, v, alpha, beta, density

    def get_uncertainty(self, v, alpha, beta):
        aleatoric = beta / (alpha - 1)
        epistemic = beta / (v * (alpha - 1))
        return aleatoric, epistemic


# ── Data helpers ───────────────────────────────────────────────────
def _forward_smooth(s, window=10):
    vals = s.values
    out = np.empty_like(vals, dtype=np.float64)
    n = len(vals)
    for k in range(n):
        end = min(k + window, n)
        out[k] = np.mean(vals[k:end])
    return pd.Series(out, index=s.index)


def load_and_prepare(csv_path, feature_cols, scaler):
    """Load CSV, smooth targets, filter, scale features, return df."""
    df = pd.read_csv(csv_path)
    df["recall_gap"] = df.groupby("sequence")["recall_gap"].transform(
        lambda s: _forward_smooth(s, window=10)
    )
    df = df[df["num_gt"] >= 1].dropna(subset=feature_cols).reset_index(drop=True)
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


def build_sliding_windows(df, feature_cols):
    X_list, y_gap_list, y_density_list, seq_ids = [], [], [], []
    total_window = T_CONTEXT + T_HORIZON
    for seq_name, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        features = sdf[feature_cols].values
        gap_targets = sdf["recall_gap"].values
        density_targets = sdf["num_gt"].values
        n = len(sdf)
        for i in range(0, n - total_window + 1, T_CONTEXT):
            X_list.append(features[i : i + T_CONTEXT])
            y_gap_list.append(gap_targets[i + T_CONTEXT : i + T_CONTEXT + T_HORIZON])
            y_density_list.append(density_targets[i + T_CONTEXT])
            seq_ids.append((seq_name, i))
    return (np.array(X_list, dtype=np.float32),
            np.array(y_gap_list, dtype=np.float32),
            np.array(y_density_list, dtype=np.float32),
            seq_ids)


def evaluate_model(model, X_val, y_gap_val, val_ids, density_max, seq_weather):
    """Run inference and collect per-sequence + overall metrics."""
    device = next(model.parameters()).device
    model.eval()

    all_gap_metrics = {}
    val_seq_data = {}

    for seq_name in VAL_SEQUENCES:
        indices = [i for i, (s, _) in enumerate(val_ids) if s == seq_name]
        if not indices:
            continue

        X_seq = torch.from_numpy(X_val[indices]).to(device)
        y_gap_seq = y_gap_val[indices]

        with torch.no_grad():
            gamma, v, alpha, beta, _ = model(X_seq)
            gap_preds = gamma.cpu().numpy()
            _, epistemic = model.get_uncertainty(v, alpha, beta)
            epistemic = epistemic.cpu().numpy()

        mae = mean_absolute_error(y_gap_seq.flatten(), gap_preds.flatten())
        r2 = r2_score(y_gap_seq.flatten(), gap_preds.flatten())
        rho, _ = spearmanr(y_gap_seq.flatten(), gap_preds.flatten())
        all_gap_metrics[seq_name] = {"mae": mae, "r2": r2, "rho": rho}

        # Frame-averaged predictions for plotting
        gap_pred_at = defaultdict(list)
        gap_epis_at = defaultdict(list)
        gap_true_at = {}
        for w, idx in enumerate(indices):
            window_start = idx - indices[0]
            for h in range(T_HORIZON):
                frame = window_start + T_CONTEXT + h
                gap_pred_at[frame].append(gap_preds[w, h])
                gap_epis_at[frame].append(epistemic[w, h])
                gap_true_at[frame] = y_gap_seq[w, h]

        frames_gap = sorted(gap_pred_at.keys())
        val_seq_data[seq_name] = {
            "frames_gap": frames_gap,
            "gap_true": np.array([gap_true_at[f] for f in frames_gap]),
            "gap_pred": np.array([np.mean(gap_pred_at[f]) for f in frames_gap]),
            "gap_epis": np.array([np.mean(gap_epis_at[f]) for f in frames_gap]),
        }

    # Overall
    all_gap_true, all_gap_pred = [], []
    for seq_name in VAL_SEQUENCES:
        indices = [i for i, (s, _) in enumerate(val_ids) if s == seq_name]
        if not indices:
            continue
        X_seq = torch.from_numpy(X_val[indices]).to(device)
        with torch.no_grad():
            gamma, _, _, _, _ = model(X_seq)
        all_gap_true.append(y_gap_val[indices].flatten())
        all_gap_pred.append(gamma.cpu().numpy().flatten())

    all_gap_true = np.concatenate(all_gap_true)
    all_gap_pred = np.concatenate(all_gap_pred)

    overall_mae = mean_absolute_error(all_gap_true, all_gap_pred)
    overall_r2 = r2_score(all_gap_true, all_gap_pred)
    overall_rho, _ = spearmanr(all_gap_true, all_gap_pred)
    all_gap_metrics["OVERALL"] = {"mae": overall_mae, "r2": overall_r2,
                                   "rho": overall_rho}

    # Per-weather
    weather_r2 = {}
    for weather_cat in ["sunny", "cloudy", "night", "rainy"]:
        w_seqs = [s for s in VAL_SEQUENCES if seq_weather.get(s) == weather_cat]
        w_true, w_pred = [], []
        for seq_name in w_seqs:
            indices = [i for i, (s, _) in enumerate(val_ids) if s == seq_name]
            if not indices:
                continue
            X_seq = torch.from_numpy(X_val[indices]).to(device)
            with torch.no_grad():
                gamma, _, _, _, _ = model(X_seq)
            w_true.append(y_gap_val[indices].flatten())
            w_pred.append(gamma.cpu().numpy().flatten())
        if w_true:
            wt = np.concatenate(w_true)
            wp = np.concatenate(w_pred)
            weather_r2[weather_cat] = r2_score(wt, wp)

    return all_gap_metrics, weather_r2, val_seq_data


# ── Main ───────────────────────────────────────────────────────────
def main():
    # Load scalers and density_max
    with open(NANO_SCALER, "rb") as f:
        nano_scaler = pickle.load(f)
    with open(MEDIUM_SCALER, "rb") as f:
        medium_scaler = pickle.load(f)
    with open(NANO_DMAX, "rb") as f:
        nano_dmax = pickle.load(f)
    with open(MEDIUM_DMAX, "rb") as f:
        medium_dmax = pickle.load(f)

    # Load models
    nano_model = MultiTaskLSTM.load_from_checkpoint(str(NANO_CKPT))
    medium_model = MultiTaskLSTM.load_from_checkpoint(str(MEDIUM_CKPT))
    print(f"Nano model loaded from {NANO_CKPT}")
    print(f"Medium model loaded from {MEDIUM_CKPT}")

    # Prepare data
    nano_df = load_and_prepare(NANO_CSV, NANO_FEATURES, nano_scaler)
    medium_df = load_and_prepare(MEDIUM_CSV, MEDIUM_FEATURES, medium_scaler)

    seq_weather = nano_df.drop_duplicates("sequence").set_index("sequence")["weather"].to_dict()

    nano_val = nano_df[nano_df["sequence"].isin(VAL_SEQUENCES)].copy()
    medium_val = medium_df[medium_df["sequence"].isin(VAL_SEQUENCES)].copy()

    X_nano, y_gap_nano, _, nano_ids = build_sliding_windows(nano_val, NANO_FEATURES)
    X_medium, y_gap_medium, _, medium_ids = build_sliding_windows(medium_val, MEDIUM_FEATURES)

    print(f"Nano val windows:   {X_nano.shape[0]}")
    print(f"Medium val windows: {X_medium.shape[0]}")

    # Evaluate both models
    print("\n--- LSTM_nano evaluation ---")
    nano_metrics, nano_weather_r2, nano_seq_data = evaluate_model(
        nano_model, X_nano, y_gap_nano, nano_ids, nano_dmax, seq_weather)

    print("\n--- LSTM_medium evaluation ---")
    medium_metrics, medium_weather_r2, medium_seq_data = evaluate_model(
        medium_model, X_medium, y_gap_medium, medium_ids, medium_dmax, seq_weather)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, m in [("Image-only (14 feat)", PREV_IMAGE_ONLY),
                      ("Both models (32 feat)", PREV_BOTH_MODELS),
                      ("LSTM_nano (23 feat)", nano_metrics["OVERALL"]),
                      ("LSTM_medium (23 feat)", medium_metrics["OVERALL"])]:
        print(f"  {label:30s}  R2={m['r2']:.4f}  MAE={m['mae']:.4f}  rho={m['rho']:.4f}")

    # ── PDF ────────────────────────────────────────────────────────
    with PdfPages(str(PDF_PATH)) as pdf:

        # ── Page 1: Summary table ──────────────────────────────────
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.axis("off")

        rows = [
            ["Image-only (14 feat)", "14 image",
             f"{PREV_IMAGE_ONLY['r2']:.4f}",
             f"{PREV_IMAGE_ONLY['mae']:.4f}",
             f"{PREV_IMAGE_ONLY['rho']:.4f}"],
            ["Both models (32 feat)", "14 image + 18 det",
             f"{PREV_BOTH_MODELS['r2']:.4f}",
             f"{PREV_BOTH_MODELS['mae']:.4f}",
             f"{PREV_BOTH_MODELS['rho']:.4f}"],
            ["LSTM_nano (23 feat)", "14 image + 9 nano",
             f"{nano_metrics['OVERALL']['r2']:.4f}",
             f"{nano_metrics['OVERALL']['mae']:.4f}",
             f"{nano_metrics['OVERALL']['rho']:.4f}"],
            ["LSTM_medium (23 feat)", "14 image + 9 medium",
             f"{medium_metrics['OVERALL']['r2']:.4f}",
             f"{medium_metrics['OVERALL']['mae']:.4f}",
             f"{medium_metrics['OVERALL']['rho']:.4f}"],
        ]

        col_labels = ["Experiment", "Features", "Gap R2", "Gap MAE", "Gap rho"]
        table = ax.table(cellText=rows, colLabels=col_labels,
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.0)

        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")

        fig.suptitle("Cross-Experiment Comparison: Recall Gap Prediction",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 2: Per-weather grouped bar chart ──────────────────
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        weather_cats = ["sunny", "cloudy", "night", "rainy"]
        x = np.arange(len(weather_cats))
        width = 0.18

        # Image-only and both-models don't have per-weather breakdowns,
        # so we only plot nano and medium per-weather, plus overall bars
        bars_nano = [nano_weather_r2.get(w, 0) for w in weather_cats]
        bars_medium = [medium_weather_r2.get(w, 0) for w in weather_cats]

        ax.bar(x - width/2, bars_nano, width, label="LSTM_nano (23 feat)",
               color="steelblue")
        ax.bar(x + width/2, bars_medium, width, label="LSTM_medium (23 feat)",
               color="coral")

        # Reference lines for overall R2
        ax.axhline(PREV_IMAGE_ONLY["r2"], color="gray", ls=":", lw=1.5,
                    label=f"Image-only overall (R2={PREV_IMAGE_ONLY['r2']:.3f})")
        ax.axhline(PREV_BOTH_MODELS["r2"], color="green", ls="--", lw=1.5,
                    label=f"Both-models overall (R2={PREV_BOTH_MODELS['r2']:.3f})")
        ax.axhline(0, color="k", lw=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(weather_cats, fontsize=11)
        ax.set_ylabel("Gap R2", fontsize=12)
        ax.set_title("Per-Weather Gap R2: LSTM_nano vs LSTM_medium",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="best")

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 3: Best 4 val sequences from LSTM_nano ───────────
        nano_seq_r2 = {s: nano_metrics[s]["r2"] for s in VAL_SEQUENCES
                       if s in nano_metrics}
        best_nano = sorted(nano_seq_r2, key=nano_seq_r2.get, reverse=True)[:4]

        fig, axes = plt.subplots(4, 1, figsize=(16, 14))
        for row, seq_name in enumerate(best_nano):
            d = nano_seq_data[seq_name]
            weather = seq_weather.get(seq_name, "unknown")
            m = nano_metrics[seq_name]

            ax = axes[row]
            ax.plot(d["frames_gap"], d["gap_true"], "b-", alpha=0.7, lw=0.8,
                    label="Actual")
            ax.plot(d["frames_gap"], d["gap_pred"], "r-", alpha=0.7, lw=0.8,
                    label="Predicted")
            ax.fill_between(
                d["frames_gap"],
                d["gap_pred"] - np.sqrt(np.clip(d["gap_epis"], 0, None)),
                d["gap_pred"] + np.sqrt(np.clip(d["gap_epis"], 0, None)),
                alpha=0.25, color="orange", label="Epistemic",
            )
            ax.set_ylabel("Recall Gap")
            ax.set_title(f"{seq_name} [{weather}] "
                         f"(MAE={m['mae']:.4f}, R2={m['r2']:.3f}, rho={m['rho']:.3f})")
            ax.legend(fontsize=7, loc="upper right")
            if row == 3:
                ax.set_xlabel("Frame index (local)")

        fig.suptitle("LSTM_nano -- Best 4 Validation Sequences (by R2)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── Page 4: Best 4 val sequences from LSTM_medium ─────────
        medium_seq_r2 = {s: medium_metrics[s]["r2"] for s in VAL_SEQUENCES
                         if s in medium_metrics}
        best_medium = sorted(medium_seq_r2, key=medium_seq_r2.get, reverse=True)[:4]

        fig, axes = plt.subplots(4, 1, figsize=(16, 14))
        for row, seq_name in enumerate(best_medium):
            d = medium_seq_data[seq_name]
            weather = seq_weather.get(seq_name, "unknown")
            m = medium_metrics[seq_name]

            ax = axes[row]
            ax.plot(d["frames_gap"], d["gap_true"], "b-", alpha=0.7, lw=0.8,
                    label="Actual")
            ax.plot(d["frames_gap"], d["gap_pred"], "r-", alpha=0.7, lw=0.8,
                    label="Predicted")
            ax.fill_between(
                d["frames_gap"],
                d["gap_pred"] - np.sqrt(np.clip(d["gap_epis"], 0, None)),
                d["gap_pred"] + np.sqrt(np.clip(d["gap_epis"], 0, None)),
                alpha=0.25, color="orange", label="Epistemic",
            )
            ax.set_ylabel("Recall Gap")
            ax.set_title(f"{seq_name} [{weather}] "
                         f"(MAE={m['mae']:.4f}, R2={m['r2']:.3f}, rho={m['rho']:.3f})")
            ax.legend(fontsize=7, loc="upper right")
            if row == 3:
                ax.set_xlabel("Frame index (local)")

        fig.suptitle("LSTM_medium -- Best 4 Validation Sequences (by R2)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"\nPDF saved -> {PDF_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
