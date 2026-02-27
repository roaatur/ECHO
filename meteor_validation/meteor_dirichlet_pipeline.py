#!/usr/bin/env python3
"""METEOR Dirichlet LSTM + RF pipeline — full honest evaluation.

Key differences from DETRAC pipeline:
  - 10 fps (every 3rd frame subsampled), not 30 fps
  - Window = 10 frames = 1 second (not 30)
  - Target = fn_coco_nano (COCO-class false negatives for nano)
  - 41 features (not 65) — different extraction set
  - 49 clips (not 100 sequences)
  - No weather labels (all "unknown") — random clip split
  - "clip" column instead of "sequence"

Output: meteor_validation/meteor_dirichlet_results.pdf
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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SCRIPT_DIR / "meteor_spatial_features.csv"
OUT_PDF = SCRIPT_DIR / "meteor_dirichlet_results.pdf"

# ── METEOR-specific constants ──────────────────────────────────────
# METEOR is subsampled every 3rd frame → ~10 fps
# Window = 10 frames = 1 second (same temporal duration as DETRAC's 30 frames)
WINDOW = 10
HORIZON = 10
TRAIN_STRIDE = 5    # 50% overlap for training
EVAL_STRIDE = 10    # non-overlapping for evaluation (unbiased)
TARGET = "fn_coco_nano"
N_CLASSES = 2
KL_ANNEALING_EPOCHS = 10
MIN_FRAMES = WINDOW + HORIZON  # need at least input + target window

META_COLS = ["clip", "frame_id", "weather", "road_type",
             "tp_nano", "tp_medium", "recoverable_fn",
             "fn_coco_nano", "fn_coco_medium",
             "fn_total_nano", "fn_total_medium",
             "num_gt", "num_gt_coco"]

# Use all 41 features — individual correlations are near-zero,
# so correlation-based selection would be arbitrary.
# Let the LSTM/RF figure out what's useful from the full set.
FEATURE_COLS = None  # set in main() after reading CSV


# ═══════════════════════════════════════════════════════════════════
# Data Pipeline
# ═══════════════════════════════════════════════════════════════════

def build_windows(df, feature_cols, target_col, stride, scaler=None,
                  threshold=None):
    """Build shifted-label windows with per-window metadata."""
    X_list, y_list, y_raw_list, clip_ids = [], [], [], []
    tp_nano_last, tp_medium_last = [], []

    for clip_name, cdf in df.groupby("clip"):
        cdf = cdf.sort_values("frame_id").reset_index(drop=True)
        n = len(cdf)
        if n < MIN_FRAMES:
            continue

        feat_vals = cdf[feature_cols].values
        target_vals = cdf[target_col].values
        tp_n = cdf["tp_nano"].values
        tp_m = cdf["tp_medium"].values

        if scaler is not None:
            feat_vals = scaler.transform(feat_vals)

        for t in range(0, n - MIN_FRAMES + 1, stride):
            x_window = feat_vals[t : t + WINDOW]
            target_mean = target_vals[t + WINDOW : t + WINDOW + HORIZON].mean()

            X_list.append(x_window)
            y_list.append(int(target_mean >= threshold))
            y_raw_list.append(target_mean)
            clip_ids.append((clip_name, cdf["frame_id"].iloc[t]))

            # Last frame of input window — for medium-sampling baseline
            last_idx = t + WINDOW - 1
            tp_nano_last.append(tp_n[last_idx])
            tp_medium_last.append(tp_m[last_idx])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    y_raw = np.array(y_raw_list, dtype=np.float32)

    meta = {
        "tp_nano_last": np.array(tp_nano_last),
        "tp_medium_last": np.array(tp_medium_last),
    }
    return X, y, y_raw, clip_ids, meta


def build_rf_features(X_windows):
    """Summarize each window as mean/std/slope per feature for RF."""
    n_samples, window_len, n_feat = X_windows.shape
    t_centered = np.arange(window_len, dtype=np.float64) - (window_len - 1) / 2.0
    t_var = (t_centered ** 2).sum()
    means = X_windows.mean(axis=1)
    stds = X_windows.std(axis=1)
    slopes = np.einsum("ijk,j->ik", X_windows.astype(np.float64),
                       t_centered) / t_var
    return np.hstack([means, stds, slopes.astype(np.float32)])


# ═══════════════════════════════════════════════════════════════════
# Dirichlet Loss + LSTM
# ═══════════════════════════════════════════════════════════════════

def dirichlet_loss(alphas, targets, epoch):
    y_oh = torch.zeros_like(alphas)
    y_oh.scatter_(1, targets.unsqueeze(1), 1.0)
    S = alphas.sum(dim=1, keepdim=True)

    # Type II Maximum Likelihood
    loss_ml = (y_oh * (torch.digamma(S) - torch.digamma(alphas))).sum(dim=1)

    # Annealed KL regularizer (Sensoy et al. 2018)
    alpha_tilde = y_oh + (1.0 - y_oh) * alphas
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(S_tilde.squeeze(1))
        - torch.lgamma(torch.tensor(float(N_CLASSES)))
        - torch.lgamma(alpha_tilde).sum(dim=1)
        + ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde)
           - torch.digamma(S_tilde))).sum(dim=1)
    )
    lam = min(1.0, epoch / max(KL_ANNEALING_EPOCHS, 1))
    return (loss_ml + lam * kl).mean()


class DirichletLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, N_CLASSES), nn.Softplus(),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]) + 1.0  # Dirichlet α ≥ 1

    def predict(self, x):
        alphas = self.forward(x)
        S = alphas.sum(dim=1, keepdim=True)
        probs = alphas / S
        uncertainty = N_CLASSES / S.squeeze(1)
        return probs.argmax(dim=1), probs[:, 1], uncertainty


def train_lstm(X_train, y_train, X_val, y_val, n_feat, device):
    model = DirichletLSTM(input_size=n_feat).to(device)
    print(f"  LSTM params: {sum(p.numel() for p in model.parameters()):,}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=64, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=128)

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6)

    best_acc, best_state, patience = 0.0, None, 0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(200):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            alphas = model(xb)
            loss = dirichlet_loss(alphas, yb, epoch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(yb)
            t_correct += ((alphas / alphas.sum(1, keepdim=True)).argmax(1) == yb).sum().item()
            t_total += len(yb)

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                alphas = model(xb)
                v_loss += dirichlet_loss(alphas, yb, epoch).item() * len(yb)
                v_correct += ((alphas / alphas.sum(1, keepdim=True)).argmax(1) == yb).sum().item()
                v_total += len(yb)

        ta, va = t_correct / t_total, v_correct / v_total
        tl, vl = t_loss / t_total, v_loss / v_total
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        scheduler.step(va)

        if va > best_acc:
            best_acc = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 10 == 0 or patience == 0:
            m = " *" if patience == 0 else ""
            print(f"  Epoch {epoch:3d}: acc={ta:.3f}/{va:.3f} "
                  f"loss={tl:.4f}/{vl:.4f}{m}", flush=True)
        if patience >= 25:
            print(f"  Early stop epoch {epoch} (best={best_acc:.4f})")
            break

    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, history


# ═══════════════════════════════════════════════════════════════════
# Evaluation Helpers
# ═══════════════════════════════════════════════════════════════════

def eval_clf(y_true, y_pred, y_prob=None, label=""):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    cm = confusion_matrix(y_true, y_pred)
    print(f"  {label:40s} Acc={acc:.3f} F1={f1:.3f} "
          f"Prec={prec:.3f} Rec={rec:.3f} AUC={auc:.3f}")
    return {"acc": acc, "f1": f1, "prec": prec, "rec": rec, "auc": auc, "cm": cm}


def per_clip_eval(y_true, y_pred, clip_ids):
    clip_data = defaultdict(lambda: {"t": [], "p": []})
    for i, (cn, _) in enumerate(clip_ids):
        clip_data[cn]["t"].append(y_true[i])
        clip_data[cn]["p"].append(y_pred[i])

    clip_accs = {}
    for cn, d in clip_data.items():
        yt, yp = np.array(d["t"]), np.array(d["p"])
        mixed = len(np.unique(yt)) > 1
        clip_accs[cn] = {"acc": accuracy_score(yt, yp), "n": len(yt),
                          "mixed": mixed}
    return clip_accs


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load ──
    print("Loading METEOR features ...", flush=True)
    df = pd.read_csv(FEATURES_CSV)

    # Identify feature columns (everything not in META_COLS)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  {len(df):,} frames, {df['clip'].nunique()} clips, "
          f"{len(feature_cols)} features")

    # Fill NaN in temporal/motion features (first frame of each clip)
    temporal_feats = [c for c in feature_cols
                      if any(k in c for k in ["flow_", "temporal_diff_"])]
    for c in temporal_feats:
        df[c] = df[c].fillna(0.0)

    # Check for remaining NaN
    nan_counts = df[feature_cols].isna().sum()
    if nan_counts.sum() > 0:
        print(f"  WARNING: {nan_counts.sum()} remaining NaN values, filling with 0")
        df[feature_cols] = df[feature_cols].fillna(0.0)

    # ── Train/Val Split ──
    # No weather labels available — use random split with fixed seed
    all_clips = sorted(df["clip"].unique())
    rng = np.random.RandomState(42)
    rng.shuffle(all_clips)
    n_val = max(1, len(all_clips) // 5)  # 20% for val
    val_clips = set(all_clips[:n_val])
    train_clips = set(all_clips[n_val:])

    print(f"  Split: {len(train_clips)} train clips, {len(val_clips)} val clips")
    print(f"  Val clips: {sorted(val_clips)[:5]}{'...' if len(val_clips) > 5 else ''}")

    train_df = df[df["clip"].isin(train_clips)].copy()
    val_df = df[df["clip"].isin(val_clips)].copy()

    # ── Compute threshold on TRAIN data only (no leakage) ──
    train_targets = []
    for _, cdf in train_df.groupby("clip"):
        cdf = cdf.sort_values("frame_id").reset_index(drop=True)
        for t in range(0, len(cdf) - MIN_FRAMES + 1, EVAL_STRIDE):
            train_targets.append(
                cdf[TARGET].iloc[t + WINDOW : t + WINDOW + HORIZON].mean())
    threshold = float(np.median(train_targets))

    # ── Class balance on ALL data (for reporting) ──
    all_targets = []
    for _, cdf in df.groupby("clip"):
        cdf = cdf.sort_values("frame_id").reset_index(drop=True)
        for t in range(0, len(cdf) - MIN_FRAMES + 1, EVAL_STRIDE):
            all_targets.append(
                cdf[TARGET].iloc[t + WINDOW : t + WINDOW + HORIZON].mean())
    all_targets = np.array(all_targets)
    hard_rate = (all_targets >= threshold).mean()

    print(f"\n{'=' * 65}")
    print("CLASS BALANCE")
    print("=" * 65)
    print(f"  Target: {TARGET}")
    print(f"  Threshold (median from train): {threshold:.3f}")
    print(f"  All data:  {len(all_targets)} windows, "
          f"{100*hard_rate:.1f}% hard / {100*(1-hard_rate):.1f}% easy")
    print(f"  Majority class baseline: {max(hard_rate, 1-hard_rate)*100:.1f}%")

    # ── Scaler fit on TRAIN only ──
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    # ── Build windows ──
    X_train, y_train, _, train_ids, train_meta = build_windows(
        train_df, feature_cols, TARGET, TRAIN_STRIDE, scaler, threshold)
    X_val, y_val, y_raw_val, val_ids, val_meta = build_windows(
        val_df, feature_cols, TARGET, EVAL_STRIDE, scaler, threshold)

    print(f"\n  Train: {len(X_train):,} windows ({y_train.mean():.1%} hard)")
    print(f"  Val:   {len(X_val):,} windows ({y_val.mean():.1%} hard)")
    print(f"  Val majority baseline: {max(y_val.mean(), 1-y_val.mean())*100:.1f}%")

    X_train_rf = build_rf_features(X_train)
    X_val_rf = build_rf_features(X_val)

    # ═══════════════════════════════════════════════════════════════
    # TRAIN MODELS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("TRAINING")
    print("=" * 65)

    # Dirichlet LSTM
    lstm, history = train_lstm(X_train, y_train, X_val, y_val,
                                len(feature_cols), device)

    # RF
    print("\n  Training RF ...", flush=True)
    rf = RandomForestClassifier(n_estimators=500, max_depth=20,
                                 min_samples_leaf=10, max_features="sqrt",
                                 n_jobs=-1, random_state=42)
    rf.fit(X_train_rf, y_train)

    # Logistic Regression
    print("  Training LogReg ...", flush=True)
    lr = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    lr.fit(X_train_rf, y_train)

    # ═══════════════════════════════════════════════════════════════
    # PREDICTIONS
    # ═══════════════════════════════════════════════════════════════
    with torch.no_grad():
        lstm_pred, lstm_prob, lstm_unc = lstm.predict(
            torch.from_numpy(X_val).to(device))
    lstm_pred = lstm_pred.cpu().numpy()
    lstm_prob = lstm_prob.cpu().numpy()
    lstm_unc = lstm_unc.cpu().numpy()

    rf_pred = rf.predict(X_val_rf)
    rf_prob = rf.predict_proba(X_val_rf)[:, 1]

    lr_pred = lr.predict(X_val_rf)
    lr_prob = lr.predict_proba(X_val_rf)[:, 1]

    # ═══════════════════════════════════════════════════════════════
    # BASELINES
    # ═══════════════════════════════════════════════════════════════

    # 1. FAST threshold (fit on TRAIN, eval on val)
    #    Use fast_keypoints_full (640px in METEOR, closest to DETRAC's fast_keypoints_half)
    if "fast_keypoints_full" in feature_cols:
        fast_idx = feature_cols.index("fast_keypoints_full")
    elif "fast_keypoints_320" in feature_cols:
        fast_idx = feature_cols.index("fast_keypoints_320")
    else:
        fast_idx = None

    if fast_idx is not None:
        train_fast = X_train[:, :, fast_idx].mean(axis=1)
        val_fast = X_val[:, :, fast_idx].mean(axis=1)
        best_fast_acc, best_fast_thr = 0, 0
        for q in np.arange(0.05, 0.95, 0.02):
            thr = np.quantile(train_fast, q)
            preds = (train_fast >= thr).astype(int)
            acc = accuracy_score(y_train, preds)
            if acc > best_fast_acc:
                best_fast_acc = acc
                best_fast_thr = thr
        fast_pred = (val_fast >= best_fast_thr).astype(int)
    else:
        fast_pred = np.zeros_like(y_val)

    # 2. Majority class
    majority_class = int(y_train.mean() >= 0.5)
    majority_pred = np.full_like(y_val, majority_class)

    # 3. Medium-sampling baseline
    #    Compare tp_medium vs tp_nano on last frame of input window
    train_gap = train_meta["tp_medium_last"] - train_meta["tp_nano_last"]
    val_gap = val_meta["tp_medium_last"] - val_meta["tp_nano_last"]

    # gap > 0: medium found more objects than nano → predict hard
    medium_simple_pred = (val_gap > 0).astype(int)

    # Tuned threshold: find best gap threshold on train
    best_mg_acc, best_mg_thr = 0, 0
    for thr in range(-2, 8):
        preds = (train_gap >= thr).astype(int)
        acc = accuracy_score(y_train, preds)
        if acc > best_mg_acc:
            best_mg_acc = acc
            best_mg_thr = thr
    medium_tuned_pred = (val_gap >= best_mg_thr).astype(int)

    # ═══════════════════════════════════════════════════════════════
    # EVALUATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("OVERALL EVALUATION (non-overlapping val windows)")
    print("=" * 65)

    results = {}
    results["Dirichlet LSTM"] = eval_clf(y_val, lstm_pred, lstm_prob,
                                          "Dirichlet LSTM")
    results["RF Classifier"] = eval_clf(y_val, rf_pred, rf_prob,
                                         "RF Classifier")
    results["Logistic Regression"] = eval_clf(y_val, lr_pred, lr_prob,
                                               "Logistic Regression")
    results["FAST Threshold"] = eval_clf(y_val, fast_pred, None,
                                          "FAST Threshold (train-fit)")
    results["FAST Threshold"]["auc"] = 0.0
    results["Medium Sample (gap>0)"] = eval_clf(y_val, medium_simple_pred,
                                                  None, "Medium Sample (gap>0)")
    results["Medium Sample (gap>0)"]["auc"] = 0.0
    results["Medium Sample (tuned)"] = eval_clf(
        y_val, medium_tuned_pred, None,
        f"Medium Sample (gap>={best_mg_thr}, tuned)")
    results["Medium Sample (tuned)"]["auc"] = 0.0
    results["Majority Class"] = eval_clf(y_val, majority_pred, None,
                                          "Majority Class")
    results["Majority Class"]["auc"] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # MPC ERROR ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("MPC ERROR ANALYSIS")
    print("=" * 65)
    print(f"  'Hard' = {TARGET} >= {threshold:.1f} → should switch to medium")
    print("  False Positive: said 'hard' but was easy → wasted energy")
    print("  False Negative: said 'easy' but was hard → MISSED OBJECTS (safety risk)")

    for name, preds in [("Dirichlet LSTM", lstm_pred),
                         ("RF Classifier", rf_pred),
                         ("Medium Sample (tuned)", medium_tuned_pred)]:
        cm = confusion_matrix(y_val, preds)
        tn, fp, fn_count, tp = cm.ravel()
        total = len(y_val)
        n_easy = tn + fp
        n_hard = fn_count + tp
        print(f"\n  {name}:")
        print(f"    True Pos  (correctly switch to medium): {tp:4d} ({100*tp/total:.1f}%)")
        print(f"    True Neg  (correctly keep nano):        {tn:4d} ({100*tn/total:.1f}%)")
        print(f"    False Pos (unnecessary medium — waste): {fp:4d} ({100*fp/total:.1f}%)")
        print(f"    False Neg (missed hard — SAFETY RISK):  {fn_count:4d} ({100*fn_count/total:.1f}%)")
        if n_easy > 0:
            print(f"    → Energy waste rate: {100*fp/n_easy:.1f}% of easy windows get medium")
        if n_hard > 0:
            print(f"    → Miss rate:         {100*fn_count/n_hard:.1f}% of hard windows missed")

    # ═══════════════════════════════════════════════════════════════
    # PER-CLIP ACCURACY AT MULTIPLE THRESHOLDS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("PER-CLIP ACCURACY")
    print("=" * 65)

    ca_lstm = per_clip_eval(y_val, lstm_pred, val_ids)
    ca_rf = per_clip_eval(y_val, rf_pred, val_ids)
    ca_ms = per_clip_eval(y_val, medium_tuned_pred, val_ids)

    for label, ca in [("Dirichlet LSTM", ca_lstm), ("RF Classifier", ca_rf),
                       ("Medium Sample", ca_ms)]:
        all_a = [v["acc"] for v in ca.values()]
        mixed_a = [v["acc"] for v in ca.values() if v["mixed"]]
        n_pure = sum(1 for v in ca.values() if not v["mixed"])
        print(f"\n  {label}:")
        print(f"    All {len(all_a)} clips:   median={np.median(all_a):.3f}  "
              f"mean={np.mean(all_a):.3f}")
        print(f"    {n_pure} pure clips (all-easy or all-hard)")
        if mixed_a:
            print(f"    {len(mixed_a)} mixed clips: median={np.median(mixed_a):.3f}  "
                  f"mean={np.mean(mixed_a):.3f}")
            for thr_pct in [50, 60, 65, 70, 75, 80]:
                n_above = sum(1 for a in mixed_a if a * 100 >= thr_pct)
                print(f"      above {thr_pct}%: {n_above}/{len(mixed_a)} "
                      f"({100*n_above/len(mixed_a):.0f}%)")
        else:
            print("    No mixed clips in val set!")

        # Print per-clip detail
        print(f"    {'Clip':<40s} {'Acc':>6s} {'N':>4s} {'Mixed':>5s}")
        for cn in sorted(ca, key=lambda c: ca[c]["acc"]):
            v = ca[cn]
            print(f"    {cn:<40s} {v['acc']:6.3f} {v['n']:4d} "
                  f"{'yes' if v['mixed'] else 'no':>5s}")

    # ═══════════════════════════════════════════════════════════════
    # UNCERTAINTY-BASED ABSTENTION (Dirichlet LSTM only)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("UNCERTAINTY-BASED ABSTENTION (Dirichlet LSTM)")
    print("=" * 65)
    print("  When uncertain → default to medium (safe).\n")

    lstm_correct = (lstm_pred == y_val)
    abstention_results = []
    for unc_pct in [0, 10, 20, 30, 40, 50]:
        if unc_pct == 0:
            mask = np.ones(len(y_val), dtype=bool)
        else:
            unc_threshold = np.percentile(lstm_unc, 100 - unc_pct)
            mask = lstm_unc <= unc_threshold

        n_kept = mask.sum()
        if n_kept == 0:
            continue
        acc_kept = accuracy_score(y_val[mask], lstm_pred[mask])

        # Abstained → default to "hard" (safe)
        combined_pred = lstm_pred.copy()
        combined_pred[~mask] = 1
        acc_combined = accuracy_score(y_val, combined_pred)

        hard_mask = y_val == 1
        recall_combined = (combined_pred[hard_mask] == 1).mean() if hard_mask.sum() > 0 else 0.0

        abstention_results.append({
            "abstain_pct": unc_pct, "n_kept": n_kept,
            "acc_confident": acc_kept, "acc_combined": acc_combined,
            "recall_hard": recall_combined,
        })
        print(f"  Abstain {unc_pct:2d}%: keep {n_kept:4d} ({100*n_kept/len(y_val):.0f}%), "
              f"acc(confident)={acc_kept:.3f}, "
              f"acc(combined)={acc_combined:.3f}, "
              f"recall(hard)={recall_combined:.3f}")

    # Uncertainty calibration
    if lstm_correct.sum() > 0 and (~lstm_correct).sum() > 0:
        unc_correct = lstm_unc[lstm_correct].mean()
        unc_incorrect = lstm_unc[~lstm_correct].mean()
        print(f"\n  Uncertainty calibration:")
        print(f"    Correct predictions:   mean uncertainty = {unc_correct:.4f}")
        print(f"    Incorrect predictions: mean uncertainty = {unc_incorrect:.4f}")
        print(f"    Gap: {unc_incorrect - unc_correct:+.4f} "
              f"({'CALIBRATED' if unc_incorrect > unc_correct else 'NOT CALIBRATED'})")

    # ═══════════════════════════════════════════════════════════════
    # SHIFTED-LABEL FEATURE CORRELATION (for context)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("SHIFTED-LABEL FEATURE CORRELATION (on val, for reference)")
    print("=" * 65)
    print("  How well does mean(feature[t:t+10]) correlate with")
    print(f"  mean({TARGET}[t+10:t+20]) across val windows?\n")

    from scipy.stats import spearmanr

    # Use unscaled val features
    val_raw_features = []
    val_raw_targets = []
    for clip_name, cdf in val_df.groupby("clip"):
        cdf = cdf.sort_values("frame_id").reset_index(drop=True)
        n = len(cdf)
        if n < MIN_FRAMES:
            continue
        for t in range(0, n - MIN_FRAMES + 1, EVAL_STRIDE):
            feat_window = cdf[feature_cols].iloc[t:t+WINDOW].values
            val_raw_features.append(feat_window.mean(axis=0))
            val_raw_targets.append(
                cdf[TARGET].iloc[t+WINDOW:t+WINDOW+HORIZON].mean())

    val_raw_features = np.array(val_raw_features)
    val_raw_targets = np.array(val_raw_targets)

    corr_list = []
    for i, feat_name in enumerate(feature_cols):
        feat_vals = val_raw_features[:, i]
        if feat_vals.std() < 1e-10:
            corr_list.append((feat_name, 0.0))
            continue
        rho, _ = spearmanr(feat_vals, val_raw_targets)
        corr_list.append((feat_name, abs(rho) if not np.isnan(rho) else 0.0))

    corr_list.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Feature':<35s} {'Shifted |ρ|':>12s}")
    print(f"  {'-'*47}")
    for feat_name, rho in corr_list[:20]:
        print(f"  {feat_name:<35s} {rho:12.4f}")
    print(f"  ... ({len(feature_cols) - 20} more features)")

    # ═══════════════════════════════════════════════════════════════
    # RF FEATURE IMPORTANCE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("RF FEATURE IMPORTANCE (top 15)")
    print("=" * 65)

    rf_feat_names = (
        [f"{f}_mean" for f in feature_cols] +
        [f"{f}_std" for f in feature_cols] +
        [f"{f}_slope" for f in feature_cols]
    )
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {rf_feat_names[idx]:<40s} {importances[idx]:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # PDF REPORT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  Generating PDF ...", flush=True)

    with PdfPages(OUT_PDF) as pdf:

        # ── Page 1: Overall comparison ──
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        model_names = list(results.keys())
        colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2",
                  "#D32F2F", "#C62828", "#455A64"]

        for ax_idx, metric in enumerate(["acc", "f1"]):
            ax = axes[ax_idx]
            vals = [results[m][metric] for m in model_names]
            x = np.arange(len(model_names))
            bars = ax.bar(x, vals, color=colors[:len(model_names)])
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel(metric.upper())
            ax.set_title(f"METEOR — {'Accuracy' if metric == 'acc' else 'F1 Score'}")
            ax.axhline(max(y_val.mean(), 1 - y_val.mean()), color="gray",
                       ls=":", lw=1, label="Majority baseline")
            ax.legend(fontsize=8)
            ax.set_ylim(0.3, 1.0)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                        f"{v:.3f}", ha="center", fontsize=8)

        fig.suptitle("METEOR Dataset — Dirichlet LSTM vs All Baselines",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

        # ── Page 2: Confusion matrices ──
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for idx, (name, preds) in enumerate([
            ("Dirichlet LSTM", lstm_pred),
            ("RF Classifier", rf_pred),
            ("Medium Sample", medium_tuned_pred),
        ]):
            ax = axes[idx]
            cm = confusion_matrix(y_val, preds)
            im = ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    color = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=16, color=color)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pred Easy", "Pred Hard"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["True Easy", "True Hard"])
            r = results.get(name, results.get("Medium Sample (tuned)", {}))
            ax.set_title(f"{name}\nAcc={r.get('acc', 0):.3f}", fontsize=10)
        fig.suptitle("METEOR — Confusion Matrices", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close()

        # ── Page 3: Uncertainty + training curves ──
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        if lstm_correct.sum() > 0 and (~lstm_correct).sum() > 0:
            ax.hist(lstm_unc[lstm_correct], bins=30, alpha=0.6,
                    label=f"Correct ({lstm_unc[lstm_correct].mean():.3f})",
                    color="green", density=True)
            ax.hist(lstm_unc[~lstm_correct], bins=30, alpha=0.6,
                    label=f"Incorrect ({lstm_unc[~lstm_correct].mean():.3f})",
                    color="red", density=True)
        ax.set_xlabel("Uncertainty (K/S)")
        ax.set_title("Uncertainty: Correct vs Incorrect")
        ax.legend(fontsize=9)

        ax = axes[0, 1]
        if abstention_results:
            abs_pcts = [r["abstain_pct"] for r in abstention_results]
            ax.plot(abs_pcts, [r["acc_confident"] for r in abstention_results],
                    "bo-", label="Acc (confident)")
            ax.plot(abs_pcts, [r["acc_combined"] for r in abstention_results],
                    "gs-", label="Acc (abstain→medium)")
            ax.plot(abs_pcts, [r["recall_hard"] for r in abstention_results],
                    "r^-", label="Recall (hard)")
            ax.legend(fontsize=8)
        ax.set_xlabel("% Abstained")
        ax.set_title("Uncertainty-Based Abstention")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        data, labels = [], []
        for label_name, ca in [("LSTM", ca_lstm), ("RF", ca_rf),
                                ("MedSample", ca_ms)]:
            mixed_a = [v["acc"] for v in ca.values() if v["mixed"]]
            if mixed_a:
                data.append(mixed_a)
                labels.append(f"{label_name}\n(n={len(mixed_a)})")
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"],
                                     ["steelblue", "forestgreen", "salmon"]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax.axhline(0.5, color="r", ls="--", lw=0.5, label="Random (50%)")
        ax.set_ylabel("Per-Clip Accuracy (mixed clips)")
        ax.set_title("Per-Clip Accuracy Distribution")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        ax.plot(history["train_acc"], "b-", alpha=0.7, label="Train")
        ax.plot(history["val_acc"], "r-", alpha=0.7, label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("LSTM Training Curves")
        ax.legend()
        ax.set_ylim(0.4, 1.0)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # ── Page 4: Feature importance + shifted correlation ──
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # RF importance
        ax = axes[0]
        top_n = min(15, len(top_idx))
        y_pos = np.arange(top_n)
        ax.barh(y_pos, [importances[idx] for idx in top_idx[:top_n]],
                color="#1976D2")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([rf_feat_names[idx] for idx in top_idx[:top_n]],
                           fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title("RF Feature Importance (top 15)")
        ax.invert_yaxis()

        # Shifted-label correlation
        ax = axes[1]
        top_corr = corr_list[:15]
        y_pos = np.arange(len(top_corr))
        ax.barh(y_pos, [c[1] for c in top_corr], color="#388E3C")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([c[0] for c in top_corr], fontsize=8)
        ax.set_xlabel("Shifted-Label |ρ|")
        ax.set_title(f"Shifted-Label Spearman (w={WINDOW})")
        ax.invert_yaxis()

        fig.suptitle("METEOR — What Features Matter?",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

        # ── Page 5: Summary ──
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis("off")

        lstm_m = results["Dirichlet LSTM"]
        rf_m = results["RF Classifier"]
        ms_m = results["Medium Sample (tuned)"]
        maj_m = results["Majority Class"]

        lines = [
            "=" * 75,
            "METEOR DATASET — FULL EVALUATION SUMMARY",
            "=" * 75,
            "",
            f"Dataset:     49 clips, ~10 fps, {len(df):,} frames, {len(feature_cols)} features",
            f"Task:        predict mean({TARGET}[t+{WINDOW}:t+{WINDOW+HORIZON}]) >= {threshold:.1f}",
            f"             from {WINDOW} frames (1 sec) of spatial features",
            f"Class split: {y_val.mean():.1%} hard / {1-y_val.mean():.1%} easy",
            f"Val set:     {len(val_clips)} clips, {len(X_val)} non-overlapping windows",
            "",
            f"{'Model':<30s} {'Acc':>6s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'AUC':>6s}",
            "-" * 60,
        ]
        for name, m in results.items():
            auc_str = f"{m['auc']:.3f}" if m['auc'] > 0 else "  —  "
            lines.append(
                f"{name:<30s} {m['acc']:6.3f} {m['f1']:6.3f} "
                f"{m['prec']:6.3f} {m['rec']:6.3f} {auc_str}")
        lines.extend([
            "",
            "KEY COMPARISON:",
            f"  Dirichlet LSTM:     {lstm_m['acc']:.1%} accuracy",
            f"  Medium-sampling:    {ms_m['acc']:.1%} accuracy",
            f"  Majority baseline:  {maj_m['acc']:.1%} accuracy",
            "",
            "VERDICT:",
        ])

        if lstm_m["acc"] > ms_m["acc"] + 0.03:
            lines.append("  LSTM meaningfully beats medium-sampling on METEOR.")
        elif lstm_m["acc"] > ms_m["acc"]:
            lines.append("  LSTM marginally beats medium-sampling (< 3pp difference).")
        elif abs(lstm_m["acc"] - ms_m["acc"]) <= 0.01:
            lines.append("  LSTM and medium-sampling are effectively tied.")
        else:
            lines.append("  Medium-sampling beats the LSTM.")

        if lstm_m["acc"] <= maj_m["acc"] + 0.05:
            lines.append("  WARNING: LSTM barely beats majority baseline.")

        lines.append("=" * 75)

        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontfamily="monospace", fontsize=7.5, verticalalignment="top")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"  PDF: {OUT_PDF}")
    print("\nDone.")


if __name__ == "__main__":
    main()
