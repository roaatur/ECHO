#!/usr/bin/env python3
"""LSTM-1L-32 vs LSTM-1L-16 multi-seed validation with per-sequence breakdown.

Also includes GRU-1L-32 for reference.
All at 0.5s window on DETRAC with 20 features.

Output: new_features/new_lstm_comparison_results.pdf
"""

import warnings
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "new_all_features.csv"
OUT_PDF = SCRIPT_DIR / "new_lstm_comparison_results.pdf"

META_COLS = ["sequence", "frame_id", "weather", "fn_nano", "num_gt",
             "tp_nano", "tp_medium", "recoverable_fn"]
TARGET = "fn_nano"
N_CLASSES = 2
KL_ANNEALING_EPOCHS = 10

SELECTED_FEATURES = [
    "fast_keypoints_half", "edge_density_coarse", "gradient_magnitude_mean",
    "gabor_nyquist_energy", "fast_keypoints_full", "glcm_homogeneity",
    "overall_edge_density", "top_edge_density", "top_high_freq_energy",
    "top_gradient_std", "fft_critical_band_ratio", "total_edge_components",
    "top_laplacian_var", "glcm_entropy", "mid_edge_density",
    "dark_channel_mean", "mid_gradient_std", "foreground_blob_count",
    "mid_high_freq_energy", "rms_contrast",
]

VAL_SEQUENCES = [
    "MVI_39361", "MVI_20011", "MVI_20012", "MVI_20032", "MVI_20051",
    "MVI_40172", "MVI_40191", "MVI_40712", "MVI_40244", "MVI_40243",
    "MVI_40992", "MVI_40732", "MVI_39811", "MVI_40775", "MVI_40752",
    "MVI_63521", "MVI_40892", "MVI_63553", "MVI_63525", "MVI_63544",
]

SEEDS = [42, 123, 456, 789, 2024]
WINDOW = 15
HORIZON = 15
TRAIN_STRIDE = 8
EVAL_STRIDE = 15

MODEL_CONFIGS = [
    {"type": "gru",  "layers": 1, "hidden": 32, "label": "GRU-1L-32"},
    {"type": "lstm", "layers": 1, "hidden": 32, "label": "LSTM-1L-32"},
    {"type": "lstm", "layers": 1, "hidden": 16, "label": "LSTM-1L-16"},
]


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


class DirichletSeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, rnn_type):
        super().__init__()
        self.rnn_type = rnn_type
        drop = 0.2 if num_layers > 1 else 0.0
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=drop)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=drop)
        head_hidden = max(16, hidden_size // 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(head_hidden, N_CLASSES),
            nn.Softplus())

    def forward(self, x):
        if self.rnn_type == "lstm":
            _, (h, _) = self.rnn(x)
        else:
            _, h = self.rnn(x)
        return self.head(h[-1]) + 1.0

    def predict(self, x):
        a = self.forward(x)
        S = a.sum(1, keepdim=True)
        p = a / S
        return p.argmax(1), p[:, 1], N_CLASSES / S.squeeze(1)


def train_one(X_tr, y_tr, X_va, y_va, mcfg, n_feat, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DirichletSeqModel(n_feat, mcfg["hidden"], mcfg["layers"],
                               mcfg["type"]).to(device)
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

    best_acc, best_state, pat = 0.0, None, 0
    best_ep = 0

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
    loader = DataLoader(TensorDataset(torch.from_numpy(X)),
                        batch_size=256)
    preds, probs, uncs = [], [], []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            p, prob, u = model.predict(xb)
            preds.append(p.cpu().numpy())
            probs.append(prob.cpu().numpy())
            uncs.append(u.cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs), np.concatenate(uncs)


def per_sequence_accuracy(preds, y_true, seq_ids):
    """Compute per-sequence accuracy."""
    from collections import defaultdict
    seq_correct = defaultdict(int)
    seq_total = defaultdict(int)
    seq_hard_pct = defaultdict(float)
    seq_hard_count = defaultdict(int)

    for i, (seq_name, _) in enumerate(seq_ids):
        seq_total[seq_name] += 1
        if preds[i] == y_true[i]:
            seq_correct[seq_name] += 1
        if y_true[i] == 1:
            seq_hard_count[seq_name] += 1

    results = []
    for seq in sorted(seq_total.keys()):
        n = seq_total[seq]
        acc = seq_correct[seq] / n
        hard_pct = seq_hard_count[seq] / n * 100
        mixed = 0 < seq_hard_count[seq] < n
        results.append({
            "sequence": seq, "acc": acc, "n": n,
            "hard_pct": hard_pct, "mixed": mixed
        })
    return sorted(results, key=lambda x: x["acc"])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n", flush=True)

    df = pd.read_csv(FEATURES_CSV)
    for c in SELECTED_FEATURES:
        if "temporal" in c or c == "motion_pixel_ratio":
            df[c] = df[c].fillna(0.0)

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
    print(f"Threshold: {threshold:.3f}\n", flush=True)

    scaler = StandardScaler()
    scaler.fit(train_df[SELECTED_FEATURES].values)

    # Build data
    X_tr, y_tr, tr_ids, tr_meta = build_windows(
        train_df, SELECTED_FEATURES, TARGET, WINDOW, HORIZON,
        TRAIN_STRIDE, scaler, threshold)
    X_va, y_va, va_ids, va_meta = build_windows(
        val_df, SELECTED_FEATURES, TARGET, WINDOW, HORIZON,
        EVAL_STRIDE, scaler, threshold)

    n_feat = X_tr.shape[2]
    print(f"Train: {len(X_tr)} windows ({y_tr.mean()*100:.1f}% hard)", flush=True)
    print(f"Val:   {len(X_va)} windows ({y_va.mean()*100:.1f}% hard)\n", flush=True)

    # MedSample baseline
    med_pred = (va_meta["tp_medium_last"] > va_meta["tp_nano_last"]).astype(int)
    med_acc = accuracy_score(y_va, med_pred)
    med_f1 = f1_score(y_va, med_pred)
    cm = confusion_matrix(y_va, med_pred)
    med_miss = cm[1, 0] / max(cm[1].sum(), 1) * 100
    print(f"MedSample: Acc={med_acc:.3f} F1={med_f1:.3f} Miss={med_miss:.1f}%\n", flush=True)

    # ── Multi-seed for each model ──
    all_model_results = {}

    for mcfg in MODEL_CONFIGS:
        label = mcfg["label"]
        print(f"{'='*70}", flush=True)
        print(f"  {label} — 5 seeds at 0.5s", flush=True)
        print(f"{'='*70}", flush=True)

        seed_results = []
        best_seed_acc = 0
        best_seed_model = None
        best_seed_id = None

        for seed in SEEDS:
            model, acc, ep, n_params = train_one(
                X_tr, y_tr, X_va, y_va, mcfg, n_feat, device, seed)

            preds, probs, uncs = get_predictions(model, X_va, device)
            f1 = f1_score(y_va, preds)
            try:
                auc = roc_auc_score(y_va, probs)
            except ValueError:
                auc = 0.0
            cm = confusion_matrix(y_va, preds)
            miss = cm[1, 0] / max(cm[1].sum(), 1) * 100

            seed_results.append({
                "seed": seed, "acc": acc, "f1": f1, "auc": auc,
                "miss": miss, "ep": ep, "n_params": n_params
            })

            print(f"  seed={seed}: Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f} "
                  f"Miss={miss:.1f}% ep={ep} ({n_params} params)", flush=True)

            if acc > best_seed_acc:
                best_seed_acc = acc
                best_seed_model = model
                best_seed_id = seed

        # Stats
        accs = [r["acc"] for r in seed_results]
        f1s = [r["f1"] for r in seed_results]
        aucs = [r["auc"] for r in seed_results]
        misses = [r["miss"] for r in seed_results]

        print(f"\n   acc: {np.mean(accs):.3f} +/- {np.std(accs):.3f} "
              f"[{min(accs):.3f}, {max(accs):.3f}]", flush=True)
        print(f"    f1: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}", flush=True)
        print(f"   auc: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}", flush=True)
        print(f"  miss: {np.mean(misses):.1f}% +/- {np.std(misses):.1f}%", flush=True)

        # Per-sequence for best seed
        preds_best, _, _ = get_predictions(best_seed_model, X_va, device)
        per_seq = per_sequence_accuracy(preds_best, y_va, va_ids)

        print(f"\n  Per-sequence accuracy (best seed={best_seed_id}):", flush=True)
        mixed_accs = []
        for r in per_seq:
            tag = "mixed" if r["mixed"] else "pure"
            print(f"    {r['sequence']:20s} {r['acc']:.3f} (n={r['n']:3d}, "
                  f"{tag}, {r['hard_pct']:.1f}% hard)", flush=True)
            if r["mixed"]:
                mixed_accs.append(r["acc"])

        if mixed_accs:
            print(f"\n  Mixed sequences ({len(mixed_accs)}):", flush=True)
            print(f"    Median: {np.median(mixed_accs):.3f}, "
                  f"Mean: {np.mean(mixed_accs):.3f}", flush=True)
            for thresh in [0.5, 0.6, 0.7, 0.75, 0.8]:
                above = sum(1 for a in mixed_accs if a >= thresh)
                print(f"    above {thresh*100:.0f}%: {above}/{len(mixed_accs)}", flush=True)

        all_model_results[label] = {
            "seed_results": seed_results,
            "per_seq": per_seq,
            "mixed_accs": mixed_accs,
            "mean_acc": np.mean(accs),
            "std_acc": np.std(accs),
            "mean_f1": np.mean(f1s),
            "mean_auc": np.mean(aucs),
            "mean_miss": np.mean(misses),
            "n_params": seed_results[0]["n_params"],
        }
        print(flush=True)

    # ── Comparison table ──
    print(f"\n{'='*70}", flush=True)
    print(f"  COMPARISON SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n  {'Model':<20s} {'Params':>7s} {'Acc':>12s} {'F1':>10s} "
          f"{'AUC':>10s} {'Miss':>10s}", flush=True)
    print(f"  {'-'*72}", flush=True)

    for label, res in all_model_results.items():
        print(f"  {label:<20s} {res['n_params']:>7d} "
              f"{res['mean_acc']:.3f}+/-{res['std_acc']:.3f} "
              f"{res['mean_f1']:.3f}      "
              f"{res['mean_auc']:.3f}      "
              f"{res['mean_miss']:.1f}%", flush=True)
    print(f"  {'MedSample':<20s} {'—':>7s} {med_acc:.3f}          "
          f"{med_f1:.3f}      {'—':>10s} {med_miss:.1f}%", flush=True)

    # ── PDF ──
    print(f"\nGenerating PDF ...", flush=True)
    with PdfPages(str(OUT_PDF)) as pdf:
        # Page 1: Accuracy comparison bar chart with error bars
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: accuracy with error bars
        ax = axes[0]
        labels_list = list(all_model_results.keys()) + ["MedSample"]
        means = [all_model_results[l]["mean_acc"] for l in all_model_results] + [med_acc]
        stds = [all_model_results[l]["std_acc"] for l in all_model_results] + [0]
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"][:len(labels_list)]
        bars = ax.bar(labels_list, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax.set_ylabel("Accuracy")
        ax.set_title("Multi-Seed Mean Accuracy (0.5s, 5 seeds)")
        ax.set_ylim(0.7, 0.85)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{m:.3f}", ha="center", fontsize=9)

        # Right: per-sequence boxplot
        ax = axes[1]
        bp_data = [all_model_results[l]["mixed_accs"] for l in all_model_results]
        bp = ax.boxplot(bp_data, labels=list(all_model_results.keys()),
                        patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_ylabel("Per-Sequence Accuracy (mixed only)")
        ax.set_title("Per-Sequence Accuracy Distribution")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.3, label="Random")
        ax.legend()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Per-sequence accuracy comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        # Get all sequences sorted by first model's accuracy
        first_label = list(all_model_results.keys())[0]
        all_seqs = [r["sequence"] for r in all_model_results[first_label]["per_seq"]]
        x = np.arange(len(all_seqs))
        width = 0.25
        for i, (label, res) in enumerate(all_model_results.items()):
            seq_acc_map = {r["sequence"]: r["acc"] for r in res["per_seq"]}
            vals = [seq_acc_map.get(s, 0) for s in all_seqs]
            ax.bar(x + i * width, vals, width, label=label, alpha=0.8,
                   color=colors[i])

        # MedSample line
        med_per_seq = per_sequence_accuracy(med_pred, y_va, va_ids)
        med_seq_map = {r["sequence"]: r["acc"] for r in med_per_seq}
        med_vals = [med_seq_map.get(s, 0) for s in all_seqs]
        ax.plot(x + width, med_vals, "k^--", markersize=6, label="MedSample", alpha=0.7)

        ax.set_xticks(x + width)
        ax.set_xticklabels(all_seqs, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Accuracy")
        ax.set_title("Per-Sequence Accuracy (0.5s, best seed)")
        ax.legend()
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.2)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Summary table
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        text = "LSTM Architecture Comparison — DETRAC 0.5s Window\n"
        text += "=" * 55 + "\n\n"
        text += f"{'Model':<18s} {'Params':>7s} {'Acc (5-seed)':>16s} {'AUC':>8s} {'Miss':>8s}\n"
        text += "-" * 60 + "\n"
        for label, res in all_model_results.items():
            text += (f"{label:<18s} {res['n_params']:>7d} "
                     f"{res['mean_acc']:.3f} +/- {res['std_acc']:.3f}   "
                     f"{res['mean_auc']:.3f}   {res['mean_miss']:.1f}%\n")
        text += f"{'MedSample':<18s} {'—':>7s} {med_acc:.3f}             {'—':>8s}   {med_miss:.1f}%\n"
        text += "\n\nPer-Sequence (mixed, best seed):\n"
        text += "-" * 40 + "\n"
        for label, res in all_model_results.items():
            ma = res["mixed_accs"]
            text += (f"  {label}: median={np.median(ma):.3f}, "
                     f">{'.7'}={sum(1 for a in ma if a>=0.7)}/{len(ma)}, "
                     f">{'.8'}={sum(1 for a in ma if a>=0.8)}/{len(ma)}\n")

        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nPDF: {OUT_PDF}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
