#!/usr/bin/env python3
"""D²-City Step 5 — Shifted-label LSTM vs Random Forest regression.

Ports new_features/new_train_models.py to D²-City.

Task: Given WINDOW=30 sampled frames of spatial features (~3.6 s at 8.3 fps),
      predict the MEAN fn_nano over the NEXT 30 sampled frames (~3.6 s ahead).
      This is the operationally relevant lookahead for an MPC switching controller.

Key design choices (same as DETRAC experiment, all bugs fixed):
  - NO num_gt filter (preserve temporal continuity within clips)
  - NO per-clip mean removal (avoid look-ahead bias)
  - Stride = 15 frames for training (50% overlap), 30 for evaluation (non-overlapping)
  - Clip-level train/val split (zero leakage)
  - StandardScaler fit on training clips only
  - 20 non-redundant features from D2_select_features.py
    (falls back to top-20 from D2_analysis_summary.csv if selection PDF not yet run)

Split strategy: no weather metadata → split by clip hash (first 80 clips train,
  last 20 val by alphabetical order of clip_id hash).

Outputs:
  D2_train_results.pdf
  D2_scaler.pkl
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D2_common import SCRIPT_DIR

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_CSV = SCRIPT_DIR / "D2_all_features.csv"
ANALYSIS_CSV = SCRIPT_DIR / "D2_analysis_summary.csv"
OUT_PDF      = SCRIPT_DIR / "D2_train_results.pdf"
SCALER_PATH  = SCRIPT_DIR / "D2_scaler.pkl"

META_COLS = ["clip_id", "frame_idx", "fn_nano", "num_gt_coco",
             "tp_nano", "tp_medium", "recoverable_fn"]

# ── Task config ───────────────────────────────────────────────────────────────
WINDOW       = 30   # input window (sampled frames ≈ 3.6 s at 8.3 fps)
HORIZON      = 30   # prediction horizon (same length ahead)
TRAIN_STRIDE = 15   # 50% overlap for training windows
EVAL_STRIDE  = 30   # non-overlapping for evaluation
TARGET       = "fn_nano"

# Fallback: top-20 features by within-clip |ρ| from analysis CSV
N_FEATURES_FALLBACK = 20

# Val clips: last 20 alphabetically (deterministic, reproducible)
N_VAL_CLIPS = 20


# ── Feature selection (auto-detect from analysis CSV) ─────────────────────────

def get_selected_features(df):
    """Load top-20 non-redundant features from analysis CSV (or top-20 by |ρ|)."""
    if not ANALYSIS_CSV.exists():
        feat_cols = [c for c in df.columns if c not in META_COLS]
        return feat_cols[:N_FEATURES_FALLBACK]
    summary = pd.read_csv(ANALYSIS_CSV)
    return summary["feature"].head(N_FEATURES_FALLBACK).tolist()


# ── Data pipeline ─────────────────────────────────────────────────────────────

def build_windows(df, feature_cols, target_col, clip_ids, stride, scaler=None):
    """Build shifted-label regression windows.

    Input:  features[t : t+WINDOW]   → (WINDOW, n_feat)
    Target: mean(fn_nano[t+WINDOW : t+WINDOW+HORIZON])   → scalar
    """
    X_list, y_list, seq_ids = [], [], []

    for clip_id in clip_ids:
        cdf = df[df["clip_id"] == clip_id].sort_values("frame_idx").reset_index(drop=True)
        n   = len(cdf)
        if n < WINDOW + HORIZON:
            continue

        feats  = cdf[feature_cols].values
        target = cdf[target_col].values

        if scaler is not None:
            feats = scaler.transform(feats)

        for t in range(0, n - WINDOW - HORIZON + 1, stride):
            X_list.append(feats[t : t + WINDOW])
            y_list.append(target[t + WINDOW : t + WINDOW + HORIZON].mean())
            seq_ids.append((clip_id, cdf["frame_idx"].iloc[t]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, seq_ids


def build_rf_features(X_windows):
    """Window → flat: [mean, std, slope] per feature = 3 × n_feat."""
    n, w, f  = X_windows.shape
    t_c      = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
    t_var    = (t_c ** 2).sum()
    means    = X_windows.mean(axis=1)
    stds     = X_windows.std(axis=1)
    slopes   = np.einsum("ijk,j->ik", X_windows.astype(np.float64),
                         t_c) / t_var
    return np.hstack([means, stds, slopes.astype(np.float32)])


# ── LSTM model ────────────────────────────────────────────────────────────────

class ForecastLSTM(nn.Module):
    """2-layer LSTM → scalar prediction (same arch as new_train_models.py)."""

    def __init__(self, input_size, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden,
                            num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def train_lstm(X_train, y_train, X_val, y_val, n_feat, device):
    print("\n" + "=" * 60)
    print("TRAINING FORECAST LSTM")
    print("=" * 60)

    model = ForecastLSTM(input_size=n_feat).to(device)
    n_p   = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_p:,}  |  Train: {len(X_train):,}  |  Val: {len(X_val):,}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=128)

    opt  = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None
    patience_count = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(200):
        model.train()
        tloss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred   = model(xb)
            loss   = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tloss += loss.item() * len(yb)
        tloss /= len(X_train)

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vloss += loss_fn(model(xb), yb).item() * len(yb)
        vloss /= len(X_val)

        history["train_loss"].append(tloss)
        history["val_loss"].append(vloss)
        sch.step(vloss)

        if vloss < best_val:
            best_val   = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 20 == 0:
            marker = " *" if patience_count == 0 else ""
            print(f"  Ep {epoch:3d}: train={tloss:.4f} val={vloss:.4f}"
                  f"  lr={opt.param_groups[0]['lr']:.1e}{marker}")

        if patience_count >= 25:
            print(f"  Early stop at epoch {epoch} (best val={best_val:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, label=""):
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    print(f"  {label:<25s}  R²={r2:.4f}  MAE={mae:.4f}  ρ={rho:.4f}")
    return {"r2": r2, "mae": mae, "rho": rho}


# ── PDF report ─────────────────────────────────────────────────────────────────

def generate_pdf(history_lstm, y_val, lstm_preds, rf_preds, baseline_preds,
                 feature_cols, rf_importances, metrics):

    with PdfPages(OUT_PDF) as pdf:

        # Page 1: Training curves + scatter
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        ax = axes[0]
        ax.plot(history_lstm["train_loss"], "b-", alpha=0.7, label="Train MSE")
        ax.plot(history_lstm["val_loss"],   "r-", alpha=0.7, label="Val MSE")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.set_title("LSTM Training Curves"); ax.legend()

        for ax, preds, label in zip(axes[1:], [lstm_preds, rf_preds],
                                    ["LSTM", "Random Forest"]):
            ax.scatter(y_val, preds, alpha=0.3, s=10)
            lim = max(y_val.max(), preds.max())
            ax.plot([0, lim], [0, lim], "r--", lw=1, label="Perfect")
            ax.set_xlabel("True mean fn_nano (next 30 frames)")
            ax.set_ylabel("Predicted")
            m = metrics[label]
            ax.set_title(f"{label}\nR²={m['r2']:.3f}  MAE={m['mae']:.3f}  ρ={m['rho']:.3f}")
            ax.legend(fontsize=8)

        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 2: Model comparison bar + RF importances
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        model_names = list(metrics.keys())
        colors      = ["steelblue", "forestgreen", "#FF9800"]

        ax = axes[0]
        r2s   = [metrics[m]["r2"]  for m in model_names]
        bars  = ax.bar(model_names, r2s, color=colors[:len(model_names)])
        ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("R²")
        ax.set_title("Shifted-Label Regression R² (D²-City)")
        for b, v in zip(bars, r2s):
            ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                    ha="center", fontsize=9)

        ax = axes[1]
        top15_rf = rf_importances.head(15)
        y_p      = range(len(top15_rf))
        ax.barh(y_p, top15_rf.values, color="forestgreen")
        ax.set_yticks(y_p)
        ax.set_yticklabels(top15_rf.index, fontsize=7)
        ax.set_xlabel("Importance"); ax.invert_yaxis()
        ax.set_title("RF Top 15 Feature Importances")

        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 3: Summary table
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")
        header = ["Model", "R²", "MAE", "Spearman ρ", "Note"]
        rows_t = [
            ["LSTM (2-layer, h=64)", f"{metrics['LSTM']['r2']:.4f}",
             f"{metrics['LSTM']['mae']:.4f}", f"{metrics['LSTM']['rho']:.4f}",
             "Evidential: no (MSE loss)"],
            ["Random Forest (500 trees)", f"{metrics['Random Forest']['r2']:.4f}",
             f"{metrics['Random Forest']['mae']:.4f}", f"{metrics['Random Forest']['rho']:.4f}",
             "mean/std/slope features"],
            ["Baseline (train mean)", f"{metrics['Baseline']['r2']:.4f}",
             f"{metrics['Baseline']['mae']:.4f}", f"{metrics['Baseline']['rho']:.4f}",
             "constant prediction"],
        ]
        tbl = ax.table(cellText=rows_t, colLabels=header,
                       loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.0, 2.0)
        for j in range(len(header)):
            tbl[0, j].set_facecolor("#37474F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        config = (
            f"\nTask: predict mean(fn_nano) over next {HORIZON} sampled frames "
            f"({HORIZON * 3 / 25:.1f} s at 8.3 fps)\n"
            f"Input: {WINDOW} sampled frames × {len(feature_cols)} features "
            f"({WINDOW * 3 / 25:.1f} s context window)\n"
            f"Train stride: {TRAIN_STRIDE}, Eval stride: {EVAL_STRIDE}\n"
            f"Split: {100 - N_VAL_CLIPS} train clips / {N_VAL_CLIPS} val clips "
            f"(alphabetical hash split, no weather metadata)\n"
            f"No num_gt filter, no per-clip normalization, no look-ahead bias\n"
        )
        ax.text(0.5, 0.25, config, transform=ax.transAxes,
                ha="center", va="top", fontfamily="monospace", fontsize=9)
        ax.set_title("D²-City Shifted-Label Regression Summary",
                     fontsize=14, fontweight="bold", pad=40)
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

    print(f"  PDF: {OUT_PDF}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("D²-CITY SHIFTED-LABEL REGRESSION: LSTM vs RF")
    print("=" * 70)
    print(f"  Device: {device}")

    if not FEATURES_CSV.exists():
        print(f"\n  ERROR: {FEATURES_CSV} not found. Run D2_extract_features.py first.")
        return

    df   = pd.read_csv(FEATURES_CSV)
    feat = get_selected_features(df)
    for c in feat:
        if "temporal" in c or c == "motion_pixel_ratio":
            df[c] = df[c].fillna(0.0)

    print(f"\n  {len(df):,} frames, {df['clip_id'].nunique()} clips")
    print(f"  Features: {len(feat)} — {feat[:4]} ...")

    # Train/val split by alphabetical clip_id
    all_clips = sorted(df["clip_id"].unique())
    val_clips   = all_clips[-N_VAL_CLIPS:]
    train_clips = all_clips[:-N_VAL_CLIPS]
    print(f"\n  Train: {len(train_clips)} clips | Val: {len(val_clips)} clips")

    train_df = df[df["clip_id"].isin(train_clips)]
    val_df   = df[df["clip_id"].isin(val_clips)]

    # Scaler fit on train only
    scaler = StandardScaler()
    scaler.fit(train_df[feat].values)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved: {SCALER_PATH}")

    # Build windows
    print("\n  Building windows ...", flush=True)
    X_tr, y_tr, tr_ids = build_windows(df, feat, TARGET, train_clips,
                                        TRAIN_STRIDE, scaler)
    X_va, y_va, va_ids = build_windows(df, feat, TARGET, val_clips,
                                        EVAL_STRIDE, scaler)
    print(f"  Train: {len(X_tr):,} windows  "
          f"(target mean={y_tr.mean():.2f}, std={y_tr.std():.2f})")
    print(f"  Val:   {len(X_va):,} windows  "
          f"(target mean={y_va.mean():.2f}, std={y_va.std():.2f})")

    X_tr_rf = build_rf_features(X_tr)
    X_va_rf = build_rf_features(X_va)

    # ── Train LSTM ───────────────────────────────────────────────────────────
    lstm_model, history = train_lstm(X_tr, y_tr, X_va, y_va, len(feat), device)

    with torch.no_grad():
        lstm_preds = lstm_model(
            torch.from_numpy(X_va).to(device)).cpu().numpy()

    # ── Train RF ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST REGRESSOR")
    print("=" * 60)
    rf = RandomForestRegressor(n_estimators=500, max_depth=20,
                               min_samples_leaf=10, max_features="sqrt",
                               n_jobs=-1, random_state=42)
    rf.fit(X_tr_rf, y_tr)
    rf_preds = rf.predict(X_va_rf)

    feat_names_rf = ([f"{f}_mean" for f in feat] +
                     [f"{f}_std"  for f in feat] +
                     [f"{f}_slope" for f in feat])
    rf_importances = pd.Series(rf.feature_importances_,
                               index=feat_names_rf).sort_values(ascending=False)

    # ── Baseline ─────────────────────────────────────────────────────────────
    baseline_pred = np.full(len(y_va), y_tr.mean())

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION (non-overlapping val windows)")
    print("=" * 60)
    metrics = {
        "LSTM":          evaluate(y_va, lstm_preds,    "LSTM"),
        "Random Forest": evaluate(y_va, rf_preds,      "Random Forest"),
        "Baseline":      evaluate(y_va, baseline_pred, "Baseline (train mean)"),
    }

    # ── PDF ──────────────────────────────────────────────────────────────────
    print("\n  Generating PDF ...", flush=True)
    generate_pdf(history, y_va, lstm_preds, rf_preds, baseline_pred,
                 feat, rf_importances, metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
