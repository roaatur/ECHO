#!/usr/bin/env python3
"""Check whether the model has intra-sequence signal.

For each mixed val sequence, compute:
  - Accuracy on easy windows (y=0) vs hard windows (y=1) separately
  - If the model just predicts majority class per sequence, one of these will be ~0%
  - If it has real intra-sequence signal, both should be above chance

Also: compute a "transition accuracy" — at windows where the label changes
from the previous window, how often does the model get it right?

Uses the best GRU-1L-32 seed (42) at 0.5s.
"""

import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "new_all_features.csv"

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

WINDOW = 15
HORIZON = 15
EVAL_STRIDE = 15
SEED = 42


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
    print(f"Threshold: {threshold:.3f}", flush=True)

    scaler = StandardScaler()
    scaler.fit(train_df[SELECTED_FEATURES].values)

    # Build data
    X_tr, y_tr, tr_ids, tr_meta = build_windows(
        train_df, SELECTED_FEATURES, TARGET, WINDOW, HORIZON,
        8, scaler, threshold)  # train stride=8
    X_va, y_va, va_ids, va_meta = build_windows(
        val_df, SELECTED_FEATURES, TARGET, WINDOW, HORIZON,
        EVAL_STRIDE, scaler, threshold)

    n_feat = X_tr.shape[2]
    print(f"Train: {len(X_tr)}, Val: {len(X_va)}\n", flush=True)

    # Train GRU-1L-32 with seed 42
    print("Training GRU-1L-32 (seed=42) ...", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = DirichletSeqModel(n_feat, 32, 1, "gru").to(device)
    loader_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr),
                                          torch.from_numpy(y_tr)),
                           batch_size=64, shuffle=True,
                           generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=10, min_lr=1e-6)

    best_acc, best_state, pat, best_ep = 0.0, None, 0, 0
    loader_va = DataLoader(TensorDataset(torch.from_numpy(X_va),
                                          torch.from_numpy(y_va)), batch_size=256)

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
            best_acc = va; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep; pat = 0
        else:
            pat += 1
        if pat >= 25:
            break

    model.load_state_dict(best_state)
    model.to(device).eval()
    print(f"  Best val acc: {best_acc:.3f} at epoch {best_ep}\n", flush=True)

    # Get predictions
    all_preds, all_probs, all_uncs = [], [], []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X_va)), batch_size=256):
            xb = xb.to(device)
            p, prob, u = model.predict(xb)
            all_preds.append(p.cpu().numpy())
            all_probs.append(prob.cpu().numpy())
            all_uncs.append(u.cpu().numpy())
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    uncs = np.concatenate(all_uncs)

    # MedSample predictions
    med_pred = (va_meta["tp_medium_last"] > va_meta["tp_nano_last"]).astype(int)

    # Majority-class baseline per sequence (always predict the sequence's majority)
    seq_majority = {}
    for seq_name, sdf in val_df.groupby("sequence"):
        # compute majority from training data of that sequence... no, this is val
        pass

    # ═══════════════════════════════════════════════════════════════
    # INTRA-SEQUENCE ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70, flush=True)
    print("  INTRA-SEQUENCE SIGNAL ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # Group predictions by sequence
    seq_data = defaultdict(lambda: {"preds": [], "labels": [], "probs": [],
                                     "uncs": [], "med_preds": [], "frame_ids": []})
    for i, (seq_name, fid) in enumerate(va_ids):
        seq_data[seq_name]["preds"].append(preds[i])
        seq_data[seq_name]["labels"].append(y_va[i])
        seq_data[seq_name]["probs"].append(probs[i])
        seq_data[seq_name]["uncs"].append(uncs[i])
        seq_data[seq_name]["med_preds"].append(med_pred[i])
        seq_data[seq_name]["frame_ids"].append(fid)

    print(f"\n  {'Sequence':<16s} {'N':>4s} {'Hard%':>6s} "
          f"{'AccEasy':>8s} {'AccHard':>8s} {'Overall':>8s} "
          f"{'MajBase':>8s} {'MedSmpl':>8s} {'Signal?':>8s}", flush=True)
    print(f"  {'-'*84}", flush=True)

    total_easy_correct_gru = 0
    total_easy_gru = 0
    total_hard_correct_gru = 0
    total_hard_gru = 0
    total_easy_correct_med = 0
    total_hard_correct_med = 0
    total_transitions = 0
    total_transition_correct_gru = 0
    total_transition_correct_med = 0
    mixed_results = []

    for seq_name in sorted(seq_data.keys()):
        sd = seq_data[seq_name]
        labels = np.array(sd["labels"])
        preds_s = np.array(sd["preds"])
        med_s = np.array(sd["med_preds"])
        n = len(labels)

        n_hard = labels.sum()
        n_easy = n - n_hard
        hard_pct = n_hard / n * 100

        # Skip pure sequences
        if n_hard == 0 or n_hard == n:
            # Pure — report but flag
            overall = (preds_s == labels).mean()
            maj_class = 1 if n_hard > n_easy else 0
            maj_acc = max(n_hard, n_easy) / n
            med_acc = (med_s == labels).mean()
            print(f"  {seq_name:<16s} {n:>4d} {hard_pct:>5.1f}% "
                  f"{'—':>8s} {'—':>8s} {overall:>8.3f} "
                  f"{maj_acc:>8.3f} {med_acc:>8.3f} {'PURE':>8s}", flush=True)
            continue

        # Mixed sequence — compute accuracy on easy and hard separately
        easy_mask = labels == 0
        hard_mask = labels == 1

        acc_easy = (preds_s[easy_mask] == labels[easy_mask]).mean()
        acc_hard = (preds_s[hard_mask] == labels[hard_mask]).mean()
        overall = (preds_s == labels).mean()

        # Majority baseline: always predict the majority class of this sequence
        maj_class = 1 if n_hard > n_easy else 0
        maj_acc = max(n_hard, n_easy) / n

        # MedSample accuracy
        med_acc = (med_s == labels).mean()
        med_acc_easy = (med_s[easy_mask] == labels[easy_mask]).mean() if easy_mask.sum() > 0 else 0
        med_acc_hard = (med_s[hard_mask] == labels[hard_mask]).mean() if hard_mask.sum() > 0 else 0

        # Has intra-sequence signal if both acc_easy and acc_hard > 50%
        has_signal = acc_easy > 0.5 and acc_hard > 0.5
        signal_str = "YES" if has_signal else "no"
        # Stronger test: beats majority baseline
        beats_maj = overall > maj_acc + 0.02
        if beats_maj:
            signal_str = "YES+" if has_signal else "weak"

        total_easy_correct_gru += (preds_s[easy_mask] == labels[easy_mask]).sum()
        total_easy_gru += easy_mask.sum()
        total_hard_correct_gru += (preds_s[hard_mask] == labels[hard_mask]).sum()
        total_hard_gru += hard_mask.sum()
        total_easy_correct_med += (med_s[easy_mask] == labels[easy_mask]).sum()
        total_hard_correct_med += (med_s[hard_mask] == labels[hard_mask]).sum()

        # Transition accuracy: when label changes from previous window
        for j in range(1, n):
            if labels[j] != labels[j-1]:
                total_transitions += 1
                if preds_s[j] == labels[j]:
                    total_transition_correct_gru += 1
                if med_s[j] == labels[j]:
                    total_transition_correct_med += 1

        print(f"  {seq_name:<16s} {n:>4d} {hard_pct:>5.1f}% "
              f"{acc_easy:>8.3f} {acc_hard:>8.3f} {overall:>8.3f} "
              f"{maj_acc:>8.3f} {med_acc:>8.3f} {signal_str:>8s}", flush=True)

        mixed_results.append({
            "seq": seq_name, "n": n, "hard_pct": hard_pct,
            "acc_easy": acc_easy, "acc_hard": acc_hard,
            "overall": overall, "maj_acc": maj_acc,
            "med_acc_easy": med_acc_easy, "med_acc_hard": med_acc_hard,
            "has_signal": has_signal, "beats_maj": beats_maj,
        })

    # Aggregate stats
    print(f"\n  AGGREGATE (mixed sequences only):", flush=True)
    print(f"    GRU accuracy on EASY windows:  {total_easy_correct_gru}/{total_easy_gru} "
          f"= {total_easy_correct_gru/max(total_easy_gru,1):.3f}", flush=True)
    print(f"    GRU accuracy on HARD windows:  {total_hard_correct_gru}/{total_hard_gru} "
          f"= {total_hard_correct_gru/max(total_hard_gru,1):.3f}", flush=True)
    print(f"    MedSample on EASY windows:     "
          f"{total_easy_correct_med/max(total_easy_gru,1):.3f}", flush=True)
    print(f"    MedSample on HARD windows:     "
          f"{total_hard_correct_med/max(total_hard_gru,1):.3f}", flush=True)
    print(f"\n    Transition accuracy (label changes):", flush=True)
    print(f"      Total transitions: {total_transitions}", flush=True)
    print(f"      GRU correct:   {total_transition_correct_gru}/{total_transitions} "
          f"= {total_transition_correct_gru/max(total_transitions,1):.3f}", flush=True)
    print(f"      MedSample:     {total_transition_correct_med}/{total_transitions} "
          f"= {total_transition_correct_med/max(total_transitions,1):.3f}", flush=True)

    n_signal = sum(1 for r in mixed_results if r["has_signal"])
    n_beats = sum(1 for r in mixed_results if r["beats_maj"])
    print(f"\n    Sequences with intra-seq signal (both easy&hard >50%): "
          f"{n_signal}/{len(mixed_results)}", flush=True)
    print(f"    Sequences beating majority baseline (+2%):              "
          f"{n_beats}/{len(mixed_results)}", flush=True)

    # Also: does the model just predict a constant per sequence?
    print(f"\n  PREDICTION DIVERSITY CHECK:", flush=True)
    for seq_name in sorted(seq_data.keys()):
        sd = seq_data[seq_name]
        preds_s = np.array(sd["preds"])
        labels = np.array(sd["labels"])
        n = len(labels)
        n_pred_hard = preds_s.sum()
        n_label_hard = labels.sum()
        if labels.sum() == 0 or labels.sum() == n:
            tag = "pure"
        else:
            tag = "mixed"
        print(f"    {seq_name:<16s} pred: {n_pred_hard:>3d}/{n} hard ({n_pred_hard/n*100:5.1f}%)  "
              f"label: {n_label_hard:>3d}/{n} hard ({n_label_hard/n*100:5.1f}%)  [{tag}]", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
