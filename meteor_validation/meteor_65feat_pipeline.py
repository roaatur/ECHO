#!/usr/bin/env python3
"""METEOR pipeline using DETRAC's 65 features — Dirichlet LSTM + RF.

Key improvements over meteor_dirichlet_pipeline.py:
  1. Uses DETRAC's 65 features (extracted at 960×540) instead of 41
  2. Tests both DETRAC's selected 20 features AND all 65
  3. Val split guarantees mixed clips (both easy+hard windows)
  4. Includes RF as primary comparison (not just LSTM)

Output: meteor_validation/meteor_65feat_results.pdf
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
FEATURES_CSV = SCRIPT_DIR / "meteor_65_features.csv"
OUT_PDF = SCRIPT_DIR / "meteor_65feat_results.pdf"

# ── Constants ──────────────────────────────────────────────────────
# METEOR is ~10 fps (every 3rd frame)
WINDOW = 10         # 10 frames = 1 second
HORIZON = 10        # predict 1 second ahead
TRAIN_STRIDE = 5    # 50% overlap
EVAL_STRIDE = 10    # non-overlapping
TARGET = "fn_coco_nano"
N_CLASSES = 2
KL_ANNEALING_EPOCHS = 10
MIN_FRAMES = WINDOW + HORIZON

META_COLS = ["clip", "frame_id", "fn_coco_nano", "fn_total_nano",
             "num_gt", "num_gt_coco", "tp_nano", "tp_medium",
             "fn_coco_medium", "fn_total_medium", "recoverable_fn"]

# DETRAC's 20 selected features (from hierarchical clustering)
DETRAC_20 = [
    "fast_keypoints_half", "edge_density_coarse", "gradient_magnitude_mean",
    "gabor_nyquist_energy", "fast_keypoints_full", "glcm_homogeneity",
    "overall_edge_density", "top_edge_density", "top_high_freq_energy",
    "top_gradient_std", "fft_critical_band_ratio", "total_edge_components",
    "top_laplacian_var", "glcm_entropy", "mid_edge_density",
    "dark_channel_mean", "mid_gradient_std", "foreground_blob_count",
    "mid_high_freq_energy", "rms_contrast",
]


# ═══════════════════════════════════════════════════════════════════
# Data Pipeline
# ═══════════════════════════════════════════════════════════════════

def build_windows(df, feature_cols, target_col, stride, scaler=None,
                  threshold=None):
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
    n_samples, window_len, n_feat = X_windows.shape
    t_centered = np.arange(window_len, dtype=np.float64) - (window_len - 1) / 2.0
    t_var = (t_centered ** 2).sum()
    means = X_windows.mean(axis=1)
    stds = X_windows.std(axis=1)
    slopes = np.einsum("ijk,j->ik", X_windows.astype(np.float64),
                       t_centered) / t_var
    return np.hstack([means, stds, slopes.astype(np.float32)])


# ═══════════════════════════════════════════════════════════════════
# Dirichlet LSTM
# ═══════════════════════════════════════════════════════════════════

def dirichlet_loss(alphas, targets, epoch):
    y_oh = torch.zeros_like(alphas)
    y_oh.scatter_(1, targets.unsqueeze(1), 1.0)
    S = alphas.sum(dim=1, keepdim=True)
    loss_ml = (y_oh * (torch.digamma(S) - torch.digamma(alphas))).sum(dim=1)
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
        return self.head(h_n[-1]) + 1.0

    def predict(self, x):
        alphas = self.forward(x)
        S = alphas.sum(dim=1, keepdim=True)
        probs = alphas / S
        uncertainty = N_CLASSES / S.squeeze(1)
        return probs.argmax(dim=1), probs[:, 1], uncertainty


def train_lstm(X_train, y_train, X_val, y_val, n_feat, device, label=""):
    model = DirichletLSTM(input_size=n_feat).to(device)
    print(f"  {label} LSTM params: {sum(p.numel() for p in model.parameters()):,}")

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
            print(f"    Epoch {epoch:3d}: acc={ta:.3f}/{va:.3f} "
                  f"loss={tl:.4f}/{vl:.4f}{m}", flush=True)
        if patience >= 25:
            print(f"    Early stop epoch {epoch} (best={best_acc:.4f})")
            break

    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, history


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def eval_clf(y_true, y_pred, y_prob=None, label=""):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    cm = confusion_matrix(y_true, y_pred)
    print(f"  {label:45s} Acc={acc:.3f} F1={f1:.3f} "
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
    print("Loading METEOR 65-feature CSV ...", flush=True)
    df = pd.read_csv(FEATURES_CSV)
    all_feature_cols = [c for c in df.columns if c not in META_COLS]

    print(f"  {len(df):,} frames, {df['clip'].nunique()} clips, "
          f"{len(all_feature_cols)} features")

    # Fill NaN in temporal features
    temporal_feats = [c for c in all_feature_cols
                      if any(k in c for k in ["temporal_diff_", "motion_pixel_"])]
    for c in temporal_feats:
        df[c] = df[c].fillna(0.0)
    # Fill any remaining NaN
    df[all_feature_cols] = df[all_feature_cols].fillna(0.0)

    # Verify DETRAC's 20 features exist
    detrac_20_available = [f for f in DETRAC_20 if f in all_feature_cols]
    detrac_20_missing = [f for f in DETRAC_20 if f not in all_feature_cols]
    if detrac_20_missing:
        print(f"  WARNING: Missing DETRAC features: {detrac_20_missing}")
    print(f"  DETRAC-20 available: {len(detrac_20_available)}/20")
    print(f"  All features: {len(all_feature_cols)}")

    # ── Compute threshold on all data first (to identify mixed clips) ──
    clip_labels = {}  # clip_name -> list of binary labels
    for clip_name, cdf in df.groupby("clip"):
        cdf = cdf.sort_values("frame_id").reset_index(drop=True)
        n = len(cdf)
        if n < MIN_FRAMES:
            continue
        targets = []
        for t in range(0, n - MIN_FRAMES + 1, EVAL_STRIDE):
            targets.append(cdf[TARGET].iloc[t + WINDOW : t + WINDOW + HORIZON].mean())
        clip_labels[clip_name] = targets

    # Compute median threshold across all windows
    all_targets = []
    for targets in clip_labels.values():
        all_targets.extend(targets)
    all_targets = np.array(all_targets)
    threshold = float(np.median(all_targets))

    # Identify mixed clips (have both easy and hard windows)
    mixed_clips = []
    pure_clips = []
    for clip_name, targets in clip_labels.items():
        labels = [int(t >= threshold) for t in targets]
        if len(set(labels)) > 1:
            mixed_clips.append(clip_name)
        else:
            pure_clips.append(clip_name)

    print(f"\n  Threshold (median): {threshold:.3f}")
    print(f"  Mixed clips: {len(mixed_clips)} (both easy+hard windows)")
    print(f"  Pure clips:  {len(pure_clips)} (all-easy or all-hard)")

    # ── Val split: ensure mixed clips in val ──
    # Put at least 40% of mixed clips in val, plus some pure clips
    rng = np.random.RandomState(42)
    rng.shuffle(mixed_clips)
    rng.shuffle(pure_clips)

    n_val_mixed = max(3, len(mixed_clips) * 2 // 5)  # ~40% of mixed
    n_val_pure = max(1, len(pure_clips) // 5)         # ~20% of pure

    val_clips_list = mixed_clips[:n_val_mixed] + pure_clips[:n_val_pure]
    train_clips_list = mixed_clips[n_val_mixed:] + pure_clips[n_val_pure:]

    val_clips = set(val_clips_list)
    train_clips = set(train_clips_list)

    n_val_mixed_actual = sum(1 for c in val_clips if c in set(mixed_clips))
    n_train_mixed_actual = sum(1 for c in train_clips if c in set(mixed_clips))

    print(f"\n  Val split:")
    print(f"    Train: {len(train_clips)} clips "
          f"({n_train_mixed_actual} mixed, {len(train_clips) - n_train_mixed_actual} pure)")
    print(f"    Val:   {len(val_clips)} clips "
          f"({n_val_mixed_actual} mixed, {len(val_clips) - n_val_mixed_actual} pure)")

    train_df = df[df["clip"].isin(train_clips)].copy()
    val_df = df[df["clip"].isin(val_clips)].copy()

    # METEOR-selected features (from shifted-label correlation + clustering)
    METEOR_11 = [
        "colorfulness", "top_gradient_std", "top_high_freq_energy",
        "mean_brightness", "keypoint_loss_ratio", "mscn_kurtosis",
        "saturation_mean", "glcm_energy", "edge_fine_coarse_ratio",
        "foreground_blob_count", "mscn_skewness",
    ]
    meteor_11_available = [f for f in METEOR_11 if f in all_feature_cols]

    # ── Run all three feature sets ──
    feature_sets = {
        "METEOR-11": meteor_11_available,
        "DETRAC-20": detrac_20_available,
        "All-65": all_feature_cols,
    }

    all_results = {}

    for feat_label, feat_cols in feature_sets.items():
        print(f"\n{'=' * 70}")
        print(f"  FEATURE SET: {feat_label} ({len(feat_cols)} features)")
        print(f"{'=' * 70}")

        # Scaler
        scaler = StandardScaler()
        scaler.fit(train_df[feat_cols].values)

        # Build windows
        X_train, y_train, _, train_ids, train_meta = build_windows(
            train_df, feat_cols, TARGET, TRAIN_STRIDE, scaler, threshold)
        X_val, y_val, y_raw_val, val_ids, val_meta = build_windows(
            val_df, feat_cols, TARGET, EVAL_STRIDE, scaler, threshold)

        print(f"  Train: {len(X_train):,} windows ({y_train.mean():.1%} hard)")
        print(f"  Val:   {len(X_val):,} windows ({y_val.mean():.1%} hard)")
        print(f"  Val majority baseline: {max(y_val.mean(), 1-y_val.mean())*100:.1f}%")

        X_train_rf = build_rf_features(X_train)
        X_val_rf = build_rf_features(X_val)

        # ── Train ──
        print(f"\n  Training {feat_label} LSTM ...")
        lstm, history = train_lstm(X_train, y_train, X_val, y_val,
                                    len(feat_cols), device, label=feat_label)

        print(f"\n  Training {feat_label} RF ...")
        rf = RandomForestClassifier(n_estimators=500, max_depth=20,
                                     min_samples_leaf=10, max_features="sqrt",
                                     n_jobs=-1, random_state=42)
        rf.fit(X_train_rf, y_train)

        print(f"  Training {feat_label} LogReg ...")
        lr = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
        lr.fit(X_train_rf, y_train)

        # ── Predictions ──
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

        # ── Baselines ──
        # Medium-sampling
        train_gap = train_meta["tp_medium_last"] - train_meta["tp_nano_last"]
        val_gap = val_meta["tp_medium_last"] - val_meta["tp_nano_last"]
        best_mg_acc, best_mg_thr = 0, 0
        for thr in range(-2, 8):
            preds = (train_gap >= thr).astype(int)
            acc = accuracy_score(y_train, preds)
            if acc > best_mg_acc:
                best_mg_acc = acc
                best_mg_thr = thr
        medium_pred = (val_gap >= best_mg_thr).astype(int)

        # FAST threshold
        if "fast_keypoints_half" in feat_cols:
            fast_idx = feat_cols.index("fast_keypoints_half")
        elif "fast_keypoints_full" in feat_cols:
            fast_idx = feat_cols.index("fast_keypoints_full")
        else:
            fast_idx = None

        if fast_idx is not None:
            train_fast = X_train[:, :, fast_idx].mean(axis=1)
            val_fast = X_val[:, :, fast_idx].mean(axis=1)
            best_fa, best_ft = 0, 0
            for q in np.arange(0.05, 0.95, 0.02):
                thr = np.quantile(train_fast, q)
                preds = (train_fast >= thr).astype(int)
                a = accuracy_score(y_train, preds)
                if a > best_fa:
                    best_fa = a
                    best_ft = thr
            fast_pred = (val_fast >= best_ft).astype(int)
        else:
            fast_pred = np.zeros_like(y_val)

        # Majority
        majority_class = int(y_train.mean() >= 0.5)
        majority_pred = np.full_like(y_val, majority_class)

        # ── Evaluate ──
        print(f"\n  {feat_label} — OVERALL EVALUATION:")
        results = {}
        results[f"LSTM ({feat_label})"] = eval_clf(
            y_val, lstm_pred, lstm_prob, f"LSTM ({feat_label})")
        results[f"RF ({feat_label})"] = eval_clf(
            y_val, rf_pred, rf_prob, f"RF ({feat_label})")
        results[f"LogReg ({feat_label})"] = eval_clf(
            y_val, lr_pred, lr_prob, f"LogReg ({feat_label})")

        # Only add baselines once (same for both feature sets)
        if feat_label == list(feature_sets.keys())[0]:
            results["Medium Sample"] = eval_clf(
                y_val, medium_pred, None,
                f"Medium Sample (gap>={best_mg_thr})")
            results["Medium Sample"]["auc"] = 0.0
            results["FAST Threshold"] = eval_clf(
                y_val, fast_pred, None, "FAST Threshold")
            results["FAST Threshold"]["auc"] = 0.0
            results["Majority Class"] = eval_clf(
                y_val, majority_pred, None, "Majority Class")
            results["Majority Class"]["auc"] = 0.0

        # MPC analysis
        print(f"\n  {feat_label} — MPC ERROR ANALYSIS:")
        for name, preds in [(f"LSTM ({feat_label})", lstm_pred),
                             (f"RF ({feat_label})", rf_pred)]:
            cm = confusion_matrix(y_val, preds)
            tn, fp, fn_count, tp = cm.ravel()
            total = len(y_val)
            n_easy = tn + fp
            n_hard = fn_count + tp
            print(f"    {name}:")
            print(f"      TP={tp} TN={tn} FP={fp} FN={fn_count}")
            if n_easy > 0:
                print(f"      Energy waste: {100*fp/n_easy:.1f}%")
            if n_hard > 0:
                print(f"      Miss rate:    {100*fn_count/n_hard:.1f}%")

        # Per-clip
        print(f"\n  {feat_label} — PER-CLIP ACCURACY:")
        for name, preds in [(f"LSTM ({feat_label})", lstm_pred),
                             (f"RF ({feat_label})", rf_pred)]:
            ca = per_clip_eval(y_val, preds, val_ids)
            mixed_a = [v["acc"] for v in ca.values() if v["mixed"]]
            all_a = [v["acc"] for v in ca.values()]
            print(f"    {name}:")
            print(f"      All {len(all_a)} clips: median={np.median(all_a):.3f}")
            if mixed_a:
                print(f"      {len(mixed_a)} mixed: median={np.median(mixed_a):.3f} "
                      f"mean={np.mean(mixed_a):.3f}")
                for thr_pct in [50, 60, 70, 80]:
                    n_above = sum(1 for a in mixed_a if a * 100 >= thr_pct)
                    print(f"        above {thr_pct}%: {n_above}/{len(mixed_a)}")

            # Print per-clip detail
            for cn in sorted(ca, key=lambda c: ca[c]["acc"]):
                v = ca[cn]
                print(f"      {cn:<40s} {v['acc']:.3f} (n={v['n']}, "
                      f"{'mixed' if v['mixed'] else 'pure'})")

        # Uncertainty
        lstm_correct = (lstm_pred == y_val)
        if lstm_correct.sum() > 0 and (~lstm_correct).sum() > 0:
            print(f"\n  {feat_label} — UNCERTAINTY:")
            print(f"    Correct:   {lstm_unc[lstm_correct].mean():.4f}")
            print(f"    Incorrect: {lstm_unc[~lstm_correct].mean():.4f}")
            gap = lstm_unc[~lstm_correct].mean() - lstm_unc[lstm_correct].mean()
            print(f"    Gap: {gap:+.4f} "
                  f"({'CALIBRATED' if gap > 0 else 'NOT CALIBRATED'})")

        # RF importance
        rf_feat_names = (
            [f"{f}_mean" for f in feat_cols] +
            [f"{f}_std" for f in feat_cols] +
            [f"{f}_slope" for f in feat_cols]
        )
        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        print(f"\n  {feat_label} — RF TOP 10 FEATURES:")
        for rank, idx in enumerate(top_idx, 1):
            print(f"    {rank:2d}. {rf_feat_names[idx]:<45s} {importances[idx]:.4f}")

        all_results[feat_label] = {
            "results": results,
            "lstm_pred": lstm_pred,
            "rf_pred": rf_pred,
            "medium_pred": medium_pred,
            "lstm_unc": lstm_unc,
            "lstm_correct": lstm_correct,
            "history": history,
            "val_ids": val_ids,
            "y_val": y_val,
        }

    # ═══════════════════════════════════════════════════════════════
    # COMPARISON SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("COMPARISON: DETRAC-20 vs ALL-65 FEATURES")
    print("=" * 70)

    print(f"\n  {'Model':<45s} {'Acc':>6s} {'F1':>6s} {'AUC':>6s}")
    print(f"  {'-'*63}")

    for feat_label in feature_sets:
        for name, m in all_results[feat_label]["results"].items():
            auc_str = f"{m['auc']:.3f}" if m['auc'] > 0 else "  —  "
            print(f"  {name:<45s} {m['acc']:6.3f} {m['f1']:6.3f} {auc_str}")

    # ═══════════════════════════════════════════════════════════════
    # PDF
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  Generating PDF ...", flush=True)

    with PdfPages(OUT_PDF) as pdf:

        # Page 1: Bar chart comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        all_model_names = []
        all_accs = []
        all_f1s = []
        colors = []
        color_map = {
            "LSTM": "#1976D2", "RF": "#388E3C", "LogReg": "#F57C00",
            "Medium": "#D32F2F", "FAST": "#7B1FA2", "Majority": "#455A64",
        }

        for feat_label in feature_sets:
            for name, m in all_results[feat_label]["results"].items():
                all_model_names.append(name)
                all_accs.append(m["acc"])
                all_f1s.append(m["f1"])
                for key in color_map:
                    if key in name:
                        colors.append(color_map[key])
                        break
                else:
                    colors.append("#9E9E9E")

        x = np.arange(len(all_model_names))
        for ax_idx, (metric, vals) in enumerate([("Accuracy", all_accs),
                                                    ("F1", all_f1s)]):
            ax = axes[ax_idx]
            bars = ax.bar(x, vals, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(all_model_names, rotation=45, ha="right", fontsize=6)
            ax.set_ylabel(metric)
            ax.set_title(f"METEOR — {metric}")
            ax.set_ylim(0.3, 1.0)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                        f"{v:.3f}", ha="center", fontsize=7)

        fig.suptitle("METEOR — Feature Set Comparison",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

        # Page 2: Training curves (all feature sets)
        n_sets = len(feature_sets)
        fig, axes = plt.subplots(1, n_sets, figsize=(7 * n_sets, 5))
        if n_sets == 1:
            axes = [axes]
        for idx, feat_label in enumerate(feature_sets):
            ax = axes[idx]
            h = all_results[feat_label]["history"]
            ax.plot(h["train_acc"], "b-", alpha=0.7, label="Train")
            ax.plot(h["val_acc"], "r-", alpha=0.7, label="Val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"LSTM Training — {feat_label}")
            ax.legend()
            ax.set_ylim(0.4, 1.0)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 3: Per-clip accuracy boxplot
        fig, ax = plt.subplots(figsize=(12, 6))
        data, labels = [], []
        for feat_label in feature_sets:
            for model_type in ["lstm", "rf"]:
                pred = all_results[feat_label][f"{model_type}_pred"]
                y_v = all_results[feat_label]["y_val"]
                v_ids = all_results[feat_label]["val_ids"]
                ca = per_clip_eval(y_v, pred, v_ids)
                mixed_a = [v["acc"] for v in ca.values() if v["mixed"]]
                if mixed_a:
                    data.append(mixed_a)
                    labels.append(f"{model_type.upper()}\n{feat_label}")

        # Add medium-sampling
        ca_ms = per_clip_eval(
            all_results[list(feature_sets.keys())[0]]["y_val"],
            all_results[list(feature_sets.keys())[0]]["medium_pred"],
            all_results[list(feature_sets.keys())[0]]["val_ids"])
        mixed_ms = [v["acc"] for v in ca_ms.values() if v["mixed"]]
        if mixed_ms:
            data.append(mixed_ms)
            labels.append("Medium\nSample")

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            box_colors = ["steelblue", "forestgreen", "dodgerblue",
                          "limegreen", "salmon"]
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(box_colors[i % len(box_colors)])
                patch.set_alpha(0.6)
        ax.axhline(0.5, color="r", ls="--", lw=0.5, label="Random")
        ax.set_ylabel("Per-Clip Accuracy (mixed clips only)")
        ax.set_title("Per-Clip Accuracy — Mixed Clips Only")
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 4: Summary
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis("off")

        lines = [
            "=" * 75,
            "METEOR — DETRAC-20 vs ALL-65 FEATURES — SUMMARY",
            "=" * 75,
            "",
            f"Dataset:     {df['clip'].nunique()} clips, ~10 fps, "
            f"{len(df):,} frames",
            f"Target:      {TARGET} >= {threshold:.1f} (median split)",
            f"Window:      {WINDOW} frames input, {HORIZON} frames target (1 sec each)",
            f"Val split:   {len(val_clips)} clips "
            f"({n_val_mixed_actual} mixed, "
            f"{len(val_clips) - n_val_mixed_actual} pure)",
            "",
            f"{'Model':<45s} {'Acc':>6s} {'F1':>6s} {'AUC':>6s}",
            "-" * 63,
        ]

        for feat_label in feature_sets:
            for name, m in all_results[feat_label]["results"].items():
                auc_str = f"{m['auc']:.3f}" if m['auc'] > 0 else "  —  "
                lines.append(
                    f"{name:<45s} {m['acc']:6.3f} {m['f1']:6.3f} {auc_str}")

        lines.extend(["", "=" * 75])

        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontfamily="monospace", fontsize=8, verticalalignment="top")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"  PDF: {OUT_PDF}")
    print("\nDone.")


if __name__ == "__main__":
    main()
