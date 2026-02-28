#!/usr/bin/env python3
"""Test optical flow features for difficulty prediction.

Feature sets:
  1. Flow-only (11 features)
  2. Spatial-20 + Flow (31 features)
  3. Spatial-20 only (baseline, for comparison)

Models: GRU-1L-32 (3 seeds), RF
Window: 0.5s (w=15, h=15) — best from previous experiments
Also tests binary classification AND delta regression.

Includes intra-sequence signal analysis for best model.

Output: new_features/new_flow_pipeline_results.pdf
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "new_all_features.csv"
FLOW_CSV = SCRIPT_DIR / "new_optical_flow_features.csv"
OUT_PDF = SCRIPT_DIR / "new_flow_pipeline_results.pdf"

META_COLS = ["sequence", "frame_id", "weather", "fn_nano", "num_gt",
             "tp_nano", "tp_medium", "recoverable_fn"]
TARGET = "fn_nano"
N_CLASSES = 2
KL_ANNEALING_EPOCHS = 10

SPATIAL_20 = [
    "fast_keypoints_half", "edge_density_coarse", "gradient_magnitude_mean",
    "gabor_nyquist_energy", "fast_keypoints_full", "glcm_homogeneity",
    "overall_edge_density", "top_edge_density", "top_high_freq_energy",
    "top_gradient_std", "fft_critical_band_ratio", "total_edge_components",
    "top_laplacian_var", "glcm_entropy", "mid_edge_density",
    "dark_channel_mean", "mid_gradient_std", "foreground_blob_count",
    "mid_high_freq_energy", "rms_contrast",
]

FLOW_FEATURES = [
    "flow_magnitude_mean", "flow_magnitude_std", "flow_magnitude_max",
    "flow_high_motion_ratio", "flow_direction_entropy",
    "flow_top_magnitude", "flow_mid_magnitude", "flow_bot_magnitude",
    "flow_spatial_variance", "flow_acceleration_mean", "flow_acceleration_abs",
]

VAL_SEQUENCES = [
    "MVI_39361", "MVI_20011", "MVI_20012", "MVI_20032", "MVI_20051",
    "MVI_40172", "MVI_40191", "MVI_40712", "MVI_40244", "MVI_40243",
    "MVI_40992", "MVI_40732", "MVI_39811", "MVI_40775", "MVI_40752",
    "MVI_63521", "MVI_40892", "MVI_63553", "MVI_63525", "MVI_63544",
]

SEEDS = [42, 123, 456]
WINDOW = 15
HORIZON = 15
TRAIN_STRIDE = 8
EVAL_STRIDE = 15


def build_windows(df, feature_cols, target_col, window, horizon,
                  stride, scaler=None, threshold=None):
    total_need = window + horizon
    X_list, y_list, seq_ids = [], [], []
    tp_nano_last, tp_medium_last = [], []

    for seq_name, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        n = len(sdf)
        if n < total_need:
            continue
        feat_vals = sdf[feature_cols].values
        target_vals = sdf[target_col].values
        tp_n = sdf["tp_nano"].values
        tp_m = sdf["tp_medium"].values
        if scaler is not None:
            feat_vals = scaler.transform(feat_vals)
        for t in range(0, n - total_need + 1, stride):
            X_list.append(feat_vals[t : t + window])
            y_list.append(int(target_vals[t + window : t + window + horizon].mean() >= threshold))
            seq_ids.append((seq_name, sdf["frame_id"].iloc[t]))
            tp_nano_last.append(tp_n[t + window - 1])
            tp_medium_last.append(tp_m[t + window - 1])

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.int64),
            seq_ids,
            {"tp_nano_last": np.array(tp_nano_last),
             "tp_medium_last": np.array(tp_medium_last)})


def build_rf_features(X):
    n, w, f = X.shape
    tc = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
    tv = (tc ** 2).sum()
    return np.hstack([X.mean(1), X.std(1),
                      (np.einsum("ijk,j->ik", X.astype(np.float64), tc) / tv).astype(np.float32)])


def dirichlet_loss(alphas, targets, epoch):
    y_oh = torch.zeros_like(alphas)
    y_oh.scatter_(1, targets.unsqueeze(1), 1.0)
    S = alphas.sum(dim=1, keepdim=True)
    loss_ml = (y_oh * (torch.digamma(S) - torch.digamma(alphas))).sum(dim=1)
    alpha_tilde = y_oh + (1.0 - y_oh) * alphas
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    kl = (torch.lgamma(S_tilde.squeeze(1))
          - torch.lgamma(torch.tensor(float(N_CLASSES)))
          - torch.lgamma(alpha_tilde).sum(dim=1)
          + ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde)
             - torch.digamma(S_tilde))).sum(dim=1))
    lam = min(1.0, epoch / max(KL_ANNEALING_EPOCHS, 1))
    return (loss_ml + lam * kl).mean()


class DirichletGRU(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1,
                          batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, N_CLASSES), nn.Softplus())

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]) + 1.0

    def predict(self, x):
        a = self.forward(x)
        S = a.sum(1, keepdim=True)
        p = a / S
        return p.argmax(1), p[:, 1], N_CLASSES / S.squeeze(1)


def train_gru(X_tr, y_tr, X_va, y_va, n_feat, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DirichletGRU(n_feat, 32).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    loader_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr),
                                          torch.from_numpy(y_tr)),
                           batch_size=64, shuffle=True,
                           generator=torch.Generator().manual_seed(seed))
    loader_va = DataLoader(TensorDataset(torch.from_numpy(X_va),
                                          torch.from_numpy(y_va)),
                           batch_size=256)
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=10, min_lr=1e-6)

    best_acc, best_state, pat, best_ep = 0.0, None, 0, 0
    for ep in range(200):
        model.train()
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            loss = dirichlet_loss(model(xb), yb, ep)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for xb, yb in loader_va:
                xb, yb = xb.to(device), yb.to(device)
                a = model(xb)
                vc += ((a / a.sum(1, keepdim=True)).argmax(1) == yb).sum().item()
                vt += len(yb)
        va = vc / vt
        sched.step(va)
        if va > best_acc:
            best_acc = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep; pat = 0
        else:
            pat += 1
        if pat >= 25:
            break

    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, best_acc, best_ep, n_params


def get_predictions(model, X, device):
    preds, probs, uncs = [], [], []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=256):
            p, prob, u = model.predict(xb.to(device))
            preds.append(p.cpu().numpy())
            probs.append(prob.cpu().numpy())
            uncs.append(u.cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs), np.concatenate(uncs)


def intra_seq_analysis(preds, y_va, va_ids, label=""):
    """Compute per-sequence easy/hard accuracy."""
    seq_data = defaultdict(lambda: {"preds": [], "labels": []})
    for i, (sn, _) in enumerate(va_ids):
        seq_data[sn]["preds"].append(preds[i])
        seq_data[sn]["labels"].append(y_va[i])

    results = []
    total_easy_correct, total_easy, total_hard_correct, total_hard = 0, 0, 0, 0
    transitions, trans_correct = 0, 0

    for sn in sorted(seq_data.keys()):
        labels = np.array(seq_data[sn]["labels"])
        pred_s = np.array(seq_data[sn]["preds"])
        n = len(labels)
        n_hard = labels.sum()
        n_easy = n - n_hard

        if n_hard == 0 or n_hard == n:
            continue  # Skip pure sequences

        easy_mask = labels == 0
        hard_mask = labels == 1
        acc_easy = (pred_s[easy_mask] == labels[easy_mask]).mean()
        acc_hard = (pred_s[hard_mask] == labels[hard_mask]).mean()
        overall = (pred_s == labels).mean()

        total_easy_correct += (pred_s[easy_mask] == labels[easy_mask]).sum()
        total_easy += easy_mask.sum()
        total_hard_correct += (pred_s[hard_mask] == labels[hard_mask]).sum()
        total_hard += hard_mask.sum()

        for j in range(1, n):
            if labels[j] != labels[j-1]:
                transitions += 1
                if pred_s[j] == labels[j]:
                    trans_correct += 1

        signal = acc_easy > 0.5 and acc_hard > 0.5
        results.append({
            "seq": sn, "acc_easy": acc_easy, "acc_hard": acc_hard,
            "overall": overall, "signal": signal, "n": n,
            "hard_pct": n_hard / n * 100,
        })

    return {
        "per_seq": results,
        "agg_easy_acc": total_easy_correct / max(total_easy, 1),
        "agg_hard_acc": total_hard_correct / max(total_hard, 1),
        "n_signal": sum(1 for r in results if r["signal"]),
        "n_mixed": len(results),
        "transition_acc": trans_correct / max(transitions, 1),
        "n_transitions": transitions,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n", flush=True)

    # Load and merge data
    print("Loading data ...", flush=True)
    df_feat = pd.read_csv(FEATURES_CSV)
    df_flow = pd.read_csv(FLOW_CSV)

    # Merge on sequence + frame_id
    df = df_feat.merge(df_flow, on=["sequence", "frame_id"], how="inner")
    print(f"  Merged: {len(df)} frames ({len(df_feat)} features, {len(df_flow)} flow)",
          flush=True)

    # Fill NaN in flow features
    for c in FLOW_FEATURES:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    for c in SPATIAL_20:
        if "temporal" in c or c == "motion_pixel_ratio":
            df[c] = df[c].fillna(0.0)

    # Verify all flow features exist
    flow_available = [f for f in FLOW_FEATURES if f in df.columns]
    print(f"  Flow features: {len(flow_available)}/{len(FLOW_FEATURES)}", flush=True)
    if len(flow_available) < len(FLOW_FEATURES):
        missing = [f for f in FLOW_FEATURES if f not in df.columns]
        print(f"  Missing: {missing}", flush=True)

    # Split
    actual_val = [s for s in VAL_SEQUENCES if s in set(df["sequence"].unique())]
    train_df = df[~df["sequence"].isin(actual_val)].copy()
    val_df = df[df["sequence"].isin(actual_val)].copy()

    # Threshold
    all_targets = []
    for _, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        for t in range(0, len(sdf) - 60 + 1, 30):
            all_targets.append(sdf[TARGET].iloc[t+30:t+60].mean())
    threshold = float(np.median(all_targets))
    print(f"  Threshold: {threshold:.3f}\n", flush=True)

    # Feature sets to test
    feat_sets = {
        "Flow-only (11)": flow_available,
        "Spatial-20 + Flow (31)": SPATIAL_20 + flow_available,
        "Spatial-20 (baseline)": SPATIAL_20,
    }

    # MedSample baseline (compute once)
    scaler_base = StandardScaler()
    scaler_base.fit(train_df[SPATIAL_20].values)
    _, y_va_base, va_ids_base, va_meta_base = build_windows(
        val_df, SPATIAL_20, TARGET, WINDOW, HORIZON, EVAL_STRIDE,
        scaler_base, threshold)
    med_pred = (va_meta_base["tp_medium_last"] > va_meta_base["tp_nano_last"]).astype(int)
    med_acc = accuracy_score(y_va_base, med_pred)
    med_f1 = f1_score(y_va_base, med_pred)
    cm_med = confusion_matrix(y_va_base, med_pred)
    med_miss = cm_med[1, 0] / max(cm_med[1].sum(), 1) * 100
    print(f"MedSample baseline: Acc={med_acc:.3f} F1={med_f1:.3f} Miss={med_miss:.1f}%\n",
          flush=True)

    all_results = {}

    for feat_name, feat_cols in feat_sets.items():
        print(f"{'='*70}", flush=True)
        print(f"  {feat_name}", flush=True)
        print(f"{'='*70}", flush=True)

        scaler = StandardScaler()
        scaler.fit(train_df[feat_cols].values)

        X_tr, y_tr, tr_ids, tr_meta = build_windows(
            train_df, feat_cols, TARGET, WINDOW, HORIZON,
            TRAIN_STRIDE, scaler, threshold)
        X_va, y_va, va_ids, va_meta = build_windows(
            val_df, feat_cols, TARGET, WINDOW, HORIZON,
            EVAL_STRIDE, scaler, threshold)

        n_feat = X_tr.shape[2]
        print(f"  Train: {len(X_tr)} ({y_tr.mean()*100:.1f}% hard), "
              f"Val: {len(X_va)} ({y_va.mean()*100:.1f}% hard)", flush=True)

        # GRU multi-seed
        seed_accs, seed_f1s, seed_aucs, seed_misses = [], [], [], []
        best_model, best_acc_all = None, 0

        for seed in SEEDS:
            model, acc, ep, n_params = train_gru(
                X_tr, y_tr, X_va, y_va, n_feat, device, seed)
            preds, probs, _ = get_predictions(model, X_va, device)
            f1 = f1_score(y_va, preds)
            try:
                auc = roc_auc_score(y_va, probs)
            except ValueError:
                auc = 0.0
            cm = confusion_matrix(y_va, preds)
            miss = cm[1, 0] / max(cm[1].sum(), 1) * 100

            seed_accs.append(acc)
            seed_f1s.append(f1)
            seed_aucs.append(auc)
            seed_misses.append(miss)
            print(f"    seed={seed}: Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f} "
                  f"Miss={miss:.1f}% ep={ep}", flush=True)

            if acc > best_acc_all:
                best_acc_all = acc
                best_model = model

        print(f"  GRU mean: Acc={np.mean(seed_accs):.3f}±{np.std(seed_accs):.3f} "
              f"AUC={np.mean(seed_aucs):.3f} Miss={np.mean(seed_misses):.1f}%", flush=True)

        # RF
        X_tr_rf = build_rf_features(X_tr)
        X_va_rf = build_rf_features(X_va)
        rf = RandomForestClassifier(n_estimators=300, max_depth=15,
                                     min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X_tr_rf, y_tr)
        rf_pred = rf.predict(X_va_rf)
        rf_prob = rf.predict_proba(X_va_rf)[:, 1]
        rf_acc = accuracy_score(y_va, rf_pred)
        rf_f1 = f1_score(y_va, rf_pred)
        rf_auc = roc_auc_score(y_va, rf_prob)
        cm_rf = confusion_matrix(y_va, rf_pred)
        rf_miss = cm_rf[1, 0] / max(cm_rf[1].sum(), 1) * 100
        print(f"  RF: Acc={rf_acc:.3f} F1={rf_f1:.3f} AUC={rf_auc:.3f} Miss={rf_miss:.1f}%",
              flush=True)

        # RF feature importances (top 10)
        imp = rf.feature_importances_
        # RF features are: [means(n_feat), stds(n_feat), slopes(n_feat)]
        feat_names_rf = ([f"{c}_mean" for c in feat_cols] +
                         [f"{c}_std" for c in feat_cols] +
                         [f"{c}_slope" for c in feat_cols])
        top_idx = np.argsort(imp)[::-1][:10]
        print(f"  RF top 10 features:", flush=True)
        for i, idx in enumerate(top_idx):
            print(f"    {i+1}. {feat_names_rf[idx]:40s} {imp[idx]:.4f}", flush=True)

        # Intra-sequence analysis for best GRU
        preds_best, _, _ = get_predictions(best_model, X_va, device)
        intra = intra_seq_analysis(preds_best, y_va, va_ids, feat_name)

        print(f"\n  Intra-sequence signal:", flush=True)
        print(f"    Easy accuracy:      {intra['agg_easy_acc']:.3f}", flush=True)
        print(f"    Hard accuracy:      {intra['agg_hard_acc']:.3f}", flush=True)
        print(f"    Transition accuracy: {intra['transition_acc']:.3f} "
              f"({intra['n_transitions']} transitions)", flush=True)
        print(f"    Sequences with signal (both >50%): "
              f"{intra['n_signal']}/{intra['n_mixed']}", flush=True)

        # Per-sequence detail
        print(f"\n    {'Sequence':<16s} {'AccEasy':>8s} {'AccHard':>8s} "
              f"{'Overall':>8s} {'Signal':>7s}", flush=True)
        for r in sorted(intra["per_seq"], key=lambda x: x["overall"]):
            sig = "YES" if r["signal"] else "no"
            print(f"    {r['seq']:<16s} {r['acc_easy']:>8.3f} {r['acc_hard']:>8.3f} "
                  f"{r['overall']:>8.3f} {sig:>7s}", flush=True)

        all_results[feat_name] = {
            "gru_acc": np.mean(seed_accs), "gru_std": np.std(seed_accs),
            "gru_auc": np.mean(seed_aucs), "gru_miss": np.mean(seed_misses),
            "rf_acc": rf_acc, "rf_f1": rf_f1, "rf_auc": rf_auc, "rf_miss": rf_miss,
            "intra": intra,
        }
        print(flush=True)

    # ── Summary ──
    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*70}\n", flush=True)
    print(f"  {'Feature Set':<30s} {'GRU Acc':>12s} {'RF Acc':>8s} "
          f"{'GRU AUC':>8s} {'IntraSignal':>12s} {'TransAcc':>9s}", flush=True)
    print(f"  {'-'*82}", flush=True)
    for name, r in all_results.items():
        print(f"  {name:<30s} {r['gru_acc']:.3f}±{r['gru_std']:.3f} "
              f"{r['rf_acc']:.3f}    {r['gru_auc']:.3f}    "
              f"{r['intra']['n_signal']}/{r['intra']['n_mixed']}       "
              f"{r['intra']['transition_acc']:.3f}", flush=True)
    print(f"  {'MedSample':<30s} {med_acc:.3f}          {'—':>8s}    {'—':>8s}    "
          f"{'—':>12s} {'—':>9s}", flush=True)

    # ── PDF ──
    print(f"\nGenerating PDF ...", flush=True)
    with PdfPages(str(OUT_PDF)) as pdf:
        # Page 1: Bar chart comparison
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        names = list(all_results.keys()) + ["MedSample"]
        colors = ["#FF5722", "#4CAF50", "#2196F3", "#9C27B0"]

        # Accuracy
        ax = axes[0]
        gru_accs = [all_results[n]["gru_acc"] for n in all_results] + [med_acc]
        gru_stds = [all_results[n]["gru_std"] for n in all_results] + [0]
        bars = ax.bar(range(len(names)), gru_accs, yerr=gru_stds, capsize=5,
                      color=colors[:len(names)], alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split("(")[0].strip() for n in names],
                           rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_title("GRU Accuracy (3-seed)")
        ax.set_ylim(0.65, 0.85)

        # Intra-sequence signal
        ax = axes[1]
        easy_accs = [all_results[n]["intra"]["agg_easy_acc"] for n in all_results]
        hard_accs = [all_results[n]["intra"]["agg_hard_acc"] for n in all_results]
        x = np.arange(len(all_results))
        ax.bar(x - 0.15, easy_accs, 0.3, label="Easy windows", color="#4CAF50", alpha=0.7)
        ax.bar(x + 0.15, hard_accs, 0.3, label="Hard windows", color="#F44336", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([n.split("(")[0].strip() for n in all_results],
                           rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_title("Intra-Sequence: Easy vs Hard Accuracy")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.legend()

        # Transition accuracy
        ax = axes[2]
        trans_accs = [all_results[n]["intra"]["transition_acc"] for n in all_results]
        ax.bar(range(len(all_results)), trans_accs,
               color=colors[:len(all_results)], alpha=0.8)
        ax.set_xticks(range(len(all_results)))
        ax.set_xticklabels([n.split("(")[0].strip() for n in all_results],
                           rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Transition Accuracy")
        ax.set_title("Label Transition Accuracy")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random")
        ax.legend()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Summary table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")
        text = "Optical Flow Feature Experiment\n"
        text += "=" * 60 + "\n\n"
        text += f"{'Feature Set':<30s} {'GRU Acc':>12s} {'RF Acc':>8s} {'AUC':>7s} "
        text += f"{'EasyAcc':>8s} {'HardAcc':>8s} {'TransAcc':>9s}\n"
        text += "-" * 85 + "\n"
        for name, r in all_results.items():
            text += (f"{name:<30s} {r['gru_acc']:.3f}+/-{r['gru_std']:.3f} "
                     f"{r['rf_acc']:.3f}   {r['gru_auc']:.3f}  "
                     f"{r['intra']['agg_easy_acc']:.3f}   "
                     f"{r['intra']['agg_hard_acc']:.3f}   "
                     f"{r['intra']['transition_acc']:.3f}\n")
        text += f"{'MedSample':<30s} {med_acc:.3f}          —       —      —       —       —\n"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nPDF: {OUT_PDF}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
