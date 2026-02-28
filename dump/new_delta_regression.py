#!/usr/bin/env python3
"""Delta regression: predict CHANGE in difficulty, not absolute level.

Target: mean(fn_nano over next W frames) - mean(fn_nano over current W frames)
This removes per-sequence bias entirely — persistence baseline predicts 0.

Feature sets tested:
  1. DETRAC-20 (spatial only)
  2. DETRAC-20 + 3 motion features (temporal_diff_mean/std, motion_pixel_ratio)
  3. DETRAC-20 + motion + feature velocity (per-feature slopes as extra inputs)

Window sizes: 0.5s, 1s, 2s, 3s, 5s (same past = same future)
Models: GRU-1L-32 (regression head), Random Forest

Output: new_features/new_delta_regression_results.pdf
"""

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "new_all_features.csv"
OUT_PDF = SCRIPT_DIR / "new_delta_regression_results.pdf"

META_COLS = ["sequence", "frame_id", "weather", "fn_nano", "num_gt",
             "tp_nano", "tp_medium", "recoverable_fn"]
TARGET = "fn_nano"

SPATIAL_20 = [
    "fast_keypoints_half", "edge_density_coarse", "gradient_magnitude_mean",
    "gabor_nyquist_energy", "fast_keypoints_full", "glcm_homogeneity",
    "overall_edge_density", "top_edge_density", "top_high_freq_energy",
    "top_gradient_std", "fft_critical_band_ratio", "total_edge_components",
    "top_laplacian_var", "glcm_entropy", "mid_edge_density",
    "dark_channel_mean", "mid_gradient_std", "foreground_blob_count",
    "mid_high_freq_energy", "rms_contrast",
]

MOTION_FEATURES = ["temporal_diff_mean", "temporal_diff_std", "motion_pixel_ratio"]

VAL_SEQUENCES = [
    "MVI_39361", "MVI_20011", "MVI_20012", "MVI_20032", "MVI_20051",
    "MVI_40172", "MVI_40191", "MVI_40712", "MVI_40244", "MVI_40243",
    "MVI_40992", "MVI_40732", "MVI_39811", "MVI_40775", "MVI_40752",
    "MVI_63521", "MVI_40892", "MVI_63553", "MVI_63525", "MVI_63544",
]

SEEDS = [42, 123, 456]

# Window configs: (window_frames, label, train_stride, eval_stride)
# 30 fps DETRAC
WINDOW_CONFIGS = [
    {"window": 15,  "label": "0.5s", "train_stride": 8,   "eval_stride": 15},
    {"window": 30,  "label": "1.0s", "train_stride": 15,  "eval_stride": 30},
    {"window": 60,  "label": "2.0s", "train_stride": 30,  "eval_stride": 60},
    {"window": 90,  "label": "3.0s", "train_stride": 45,  "eval_stride": 90},
    {"window": 150, "label": "5.0s", "train_stride": 75,  "eval_stride": 150},
]


def build_delta_windows(df, feature_cols, target_col, window, stride):
    """Build windows where target = mean(future W) - mean(current W)."""
    total_need = window * 2  # need current window + future window
    X_list, y_list, seq_ids = [], [], []

    for seq_name, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        n = len(sdf)
        if n < total_need:
            continue
        feat_vals = sdf[feature_cols].values
        target_vals = sdf[target_col].values.astype(float)

        for t in range(0, n - total_need + 1, stride):
            x_window = feat_vals[t : t + window]
            current_mean = target_vals[t : t + window].mean()
            future_mean = target_vals[t + window : t + window * 2].mean()
            delta = future_mean - current_mean

            X_list.append(x_window)
            y_list.append(delta)
            seq_ids.append((seq_name, sdf["frame_id"].iloc[t]))

    if not X_list:
        return None, None, None
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            seq_ids)


def build_rf_features(X_windows):
    """mean/std/slope per feature across window."""
    n, w, f = X_windows.shape
    tc = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
    tv = (tc ** 2).sum()
    means = X_windows.mean(axis=1)
    stds = X_windows.std(axis=1)
    slopes = np.einsum("ijk,j->ik", X_windows.astype(np.float64), tc) / tv
    return np.hstack([means, stds, slopes.astype(np.float32)])


class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1,
                          batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


def train_gru_regression(X_tr, y_tr, X_va, y_va, n_feat, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GRURegressor(n_feat, 32).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    loader_tr = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    loader_va = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=256)

    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    best_loss, best_state, pat, best_ep = 1e9, None, 0, 0

    for ep in range(200):
        model.train()
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in loader_va:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item() * len(yb))
        vl = sum(val_losses) / len(y_va)
        sched.step(vl)
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep; pat = 0
        else:
            pat += 1
        if pat >= 25:
            break

    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, best_ep, n_params


def evaluate_regression(y_true, y_pred, label=""):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Direction accuracy: did we predict increase vs decrease correctly?
    # Only count windows where delta != 0
    nonzero = np.abs(y_true) > 0.01
    if nonzero.sum() > 10:
        dir_correct = (np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero])).mean()
    else:
        dir_correct = float("nan")

    # Persistence baseline: predict delta = 0 (difficulty won't change)
    persist_r2 = r2_score(y_true, np.zeros_like(y_true))
    persist_mae = mean_absolute_error(y_true, np.zeros_like(y_true))

    return {
        "r2": r2, "mae": mae, "dir_acc": dir_correct,
        "persist_r2": persist_r2, "persist_mae": persist_mae,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n", flush=True)

    df = pd.read_csv(FEATURES_CSV)
    for c in SPATIAL_20 + MOTION_FEATURES:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    actual_val = [s for s in VAL_SEQUENCES if s in set(df["sequence"].unique())]
    train_df = df[~df["sequence"].isin(actual_val)].copy()
    val_df = df[df["sequence"].isin(actual_val)].copy()

    # Feature sets
    feat_sets = {
        "Spatial-20": SPATIAL_20,
        "Spatial+Motion (23)": SPATIAL_20 + MOTION_FEATURES,
    }

    # Check all features exist
    for name, feats in feat_sets.items():
        missing = [f for f in feats if f not in df.columns]
        if missing:
            print(f"WARNING: {name} missing: {missing}", flush=True)
            feat_sets[name] = [f for f in feats if f in df.columns]

    # ── Delta statistics ──
    print("=" * 70, flush=True)
    print("  DELTA TARGET STATISTICS", flush=True)
    print("=" * 70, flush=True)

    for wcfg in WINDOW_CONFIGS:
        w = wcfg["window"]
        _, y_all, _ = build_delta_windows(
            df, SPATIAL_20, TARGET, w, w)
        if y_all is None:
            print(f"  {wcfg['label']}: no data", flush=True)
            continue
        nonzero = np.abs(y_all) > 0.01
        print(f"  {wcfg['label']} (w={w}): N={len(y_all)}, "
              f"mean={y_all.mean():.4f}, std={y_all.std():.4f}, "
              f"median={np.median(y_all):.4f}, "
              f"|delta|>0.01: {nonzero.sum()}/{len(y_all)} ({nonzero.mean()*100:.1f}%)",
              flush=True)

    # ── Main experiments ──
    all_results = []

    for wcfg in WINDOW_CONFIGS:
        w = wcfg["window"]
        wlabel = wcfg["label"]
        ts = wcfg["train_stride"]
        es = wcfg["eval_stride"]

        print(f"\n{'='*70}", flush=True)
        print(f"  WINDOW: {wlabel} (w={w}, horizon={w})", flush=True)
        print(f"{'='*70}", flush=True)

        for feat_name, feat_cols in feat_sets.items():
            scaler = StandardScaler()
            scaler.fit(train_df[feat_cols].values)

            # Scale features in dataframes
            train_scaled = train_df.copy()
            val_scaled = val_df.copy()
            train_scaled[feat_cols] = scaler.transform(train_df[feat_cols].values)
            val_scaled[feat_cols] = scaler.transform(val_df[feat_cols].values)

            result = build_delta_windows(train_scaled, feat_cols, TARGET, w, ts)
            if result[0] is None:
                print(f"  {feat_name}: no training data at w={w}", flush=True)
                continue
            X_tr, y_tr, _ = result

            result = build_delta_windows(val_scaled, feat_cols, TARGET, w, es)
            if result[0] is None:
                print(f"  {feat_name}: no val data at w={w}", flush=True)
                continue
            X_va, y_va, va_ids = result

            n_feat = X_tr.shape[2]
            print(f"\n  {feat_name} ({n_feat} features):", flush=True)
            print(f"    Train: {len(X_tr)}, Val: {len(X_va)}", flush=True)
            print(f"    Target stats — train: mean={y_tr.mean():.4f} std={y_tr.std():.4f}  "
                  f"val: mean={y_va.mean():.4f} std={y_va.std():.4f}", flush=True)

            # GRU regression (multi-seed)
            gru_r2s, gru_maes, gru_dirs = [], [], []
            for seed in SEEDS:
                model, ep, n_params = train_gru_regression(
                    X_tr, y_tr, X_va, y_va, n_feat, device, seed)

                with torch.no_grad():
                    preds = []
                    for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X_va)),
                                            batch_size=256):
                        preds.append(model(xb.to(device)).cpu().numpy())
                    y_pred = np.concatenate(preds)

                metrics = evaluate_regression(y_va, y_pred)
                gru_r2s.append(metrics["r2"])
                gru_maes.append(metrics["mae"])
                gru_dirs.append(metrics["dir_acc"])

            print(f"    GRU:  R²={np.mean(gru_r2s):.4f}±{np.std(gru_r2s):.4f}  "
                  f"MAE={np.mean(gru_maes):.4f}  "
                  f"DirAcc={np.nanmean(gru_dirs):.3f}", flush=True)

            # RF regression
            X_tr_rf = build_rf_features(X_tr)
            X_va_rf = build_rf_features(X_va)
            rf = RandomForestRegressor(n_estimators=200, max_depth=15,
                                       min_samples_leaf=10, random_state=42,
                                       n_jobs=-1)
            rf.fit(X_tr_rf, y_tr)
            y_pred_rf = rf.predict(X_va_rf)
            rf_metrics = evaluate_regression(y_va, y_pred_rf)
            print(f"    RF:   R²={rf_metrics['r2']:.4f}  "
                  f"MAE={rf_metrics['mae']:.4f}  "
                  f"DirAcc={rf_metrics['dir_acc']:.3f}", flush=True)

            # Persistence baseline (predict delta = 0)
            persist_metrics = evaluate_regression(y_va, np.zeros_like(y_va))
            print(f"    Persist(0): R²={persist_metrics['r2']:.4f}  "
                  f"MAE={persist_metrics['mae']:.4f}", flush=True)

            # Intra-sequence check: per-sequence R² for GRU best seed
            model_best, _, _ = train_gru_regression(
                X_tr, y_tr, X_va, y_va, n_feat, device, 42)
            with torch.no_grad():
                preds_all = []
                for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X_va)),
                                        batch_size=256):
                    preds_all.append(model_best(xb.to(device)).cpu().numpy())
                y_pred_best = np.concatenate(preds_all)

            # Per-sequence
            seq_preds = defaultdict(list)
            seq_labels = defaultdict(list)
            for i, (sn, _) in enumerate(va_ids):
                seq_preds[sn].append(y_pred_best[i])
                seq_labels[sn].append(y_va[i])

            print(f"    Per-sequence (GRU seed=42):", flush=True)
            for sn in sorted(seq_preds.keys()):
                sp = np.array(seq_preds[sn])
                sl = np.array(seq_labels[sn])
                if len(sl) < 5 or sl.std() < 0.001:
                    print(f"      {sn}: N={len(sl)}, target_std={sl.std():.4f} (flat)", flush=True)
                    continue
                sr2 = r2_score(sl, sp)
                # Direction accuracy for this sequence
                nz = np.abs(sl) > 0.01
                if nz.sum() > 3:
                    sdir = (np.sign(sp[nz]) == np.sign(sl[nz])).mean()
                else:
                    sdir = float("nan")
                print(f"      {sn}: R²={sr2:.4f}, DirAcc={sdir:.3f}, "
                      f"N={len(sl)}, target_std={sl.std():.4f}", flush=True)

            all_results.append({
                "window": wlabel, "features": feat_name,
                "gru_r2": np.mean(gru_r2s), "gru_r2_std": np.std(gru_r2s),
                "gru_mae": np.mean(gru_maes), "gru_dir": np.nanmean(gru_dirs),
                "rf_r2": rf_metrics["r2"], "rf_mae": rf_metrics["mae"],
                "rf_dir": rf_metrics["dir_acc"],
                "persist_r2": persist_metrics["r2"],
                "persist_mae": persist_metrics["mae"],
                "n_train": len(X_tr), "n_val": len(X_va),
            })

    # ── Summary ──
    print(f"\n\n{'='*70}", flush=True)
    print(f"  SUMMARY TABLE", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n  {'Window':<8s} {'Features':<25s} {'GRU R²':>10s} {'RF R²':>10s} "
          f"{'Persist R²':>10s} {'GRU Dir':>8s} {'RF Dir':>8s} {'N_val':>6s}", flush=True)
    print(f"  {'-'*85}", flush=True)
    for r in all_results:
        print(f"  {r['window']:<8s} {r['features']:<25s} "
              f"{r['gru_r2']:>10.4f} {r['rf_r2']:>10.4f} "
              f"{r['persist_r2']:>10.4f} {r['gru_dir']:>8.3f} {r['rf_dir']:>8.3f} "
              f"{r['n_val']:>6d}", flush=True)

    # ── PDF ──
    print(f"\nGenerating PDF ...", flush=True)
    with PdfPages(str(OUT_PDF)) as pdf:
        # Page 1: R² across windows for each feature set
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, metric, title in [(axes[0], "r2", "R²"), (axes[1], "dir", "Direction Accuracy")]:
            for feat_name in feat_sets:
                subset = [r for r in all_results if r["features"] == feat_name]
                if not subset:
                    continue
                windows = [r["window"] for r in subset]
                gru_vals = [r[f"gru_{metric}"] for r in subset]
                rf_vals = [r[f"rf_{metric}"] for r in subset]
                ax.plot(windows, gru_vals, "o-", label=f"GRU ({feat_name})", markersize=6)
                ax.plot(windows, rf_vals, "s--", label=f"RF ({feat_name})", markersize=6)

            if metric == "r2":
                persist_vals = [r["persist_r2"] for r in all_results
                               if r["features"] == list(feat_sets.keys())[0]]
                if persist_vals:
                    ax.plot([r["window"] for r in all_results
                            if r["features"] == list(feat_sets.keys())[0]],
                           persist_vals, "k^--", label="Persist(0)", markersize=6)

            ax.set_xlabel("Window")
            ax.set_ylabel(title)
            ax.set_title(f"Delta Regression — {title}")
            ax.legend(fontsize=7)
            ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
            if metric == "dir":
                ax.axhline(0.5, color="red", linestyle="--", alpha=0.3, label="Random")
                ax.set_ylim(0.3, 0.8)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Summary table
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")
        text = "Delta Regression: Predict CHANGE in difficulty\n"
        text += "Target: mean(fn_nano next W) - mean(fn_nano current W)\n"
        text += "Persistence baseline: always predict delta=0\n\n"
        text += f"{'Window':<8s} {'Features':<25s} {'GRU R²':>10s} {'RF R²':>10s} "
        text += f"{'Persist R²':>10s} {'GRU Dir':>8s} {'RF Dir':>8s}\n"
        text += "-" * 85 + "\n"
        for r in all_results:
            text += (f"{r['window']:<8s} {r['features']:<25s} "
                     f"{r['gru_r2']:>10.4f} {r['rf_r2']:>10.4f} "
                     f"{r['persist_r2']:>10.4f} {r['gru_dir']:>8.3f} {r['rf_dir']:>8.3f}\n")
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nPDF: {OUT_PDF}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
