#!/usr/bin/env python3
"""Full VisDrone evaluation on ALL 73 sequences (50 train / 23 val).

Previous experiments used only 6 val sequences — too few for reliable results.
This script uses the official VisDrone VID splits properly:
  - Train: 50 sequences (official train split minus 4 missing eval results)
  - Val: 23 sequences (6 official val + 17 test-dev)

Runs three experiment variants:
  1. DETRAC-20 (blind cross-domain — features selected on DETRAC, no VisDrone leakage)
  2. Jay's-48 features with VisDrone-based feature selection
  3. Combined 113 features (DETRAC-65 + Jay's-48) with multiple selection methods

Models: GRU-1L-32 Dirichlet (3 seeds) + Random Forest
Baselines: MedSample (tp_medium > tp_nano), majority class

Output:
  - viss_full_evaluation_results.pdf
  - viss_full_evaluation_results.csv
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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ECHO_DIR = SCRIPT_DIR.parent
EVAL_CSV = ECHO_DIR / "yolo_runs_on_uavdt" / "visdrone_classagnostic_results.csv"
DETRAC_FEAT_CSV = SCRIPT_DIR / "viss_detrac65_features.csv"
JAYS_FEAT_CSV = ECHO_DIR / "yolo_runs_on_uavdt" / "jays" / "jays_visdrone_features.csv"
OUT_PDF = SCRIPT_DIR / "viss_full_evaluation_results.pdf"
OUT_CSV = SCRIPT_DIR / "viss_full_evaluation_results.csv"

# 23 val sequences = 6 official val + 17 test-dev
# (uav0000086_00000_v is official val but has no eval results — excluded)
VAL_SEQS = [
    # Official VisDrone VID val (6 of 7 — uav0000086 missing eval)
    "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
    "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v",
    # Official VisDrone VID test-dev (17)
    "uav0000009_03358_v", "uav0000073_00600_v", "uav0000073_04464_v",
    "uav0000077_00720_v", "uav0000088_00290_v", "uav0000119_02301_v",
    "uav0000120_04775_v", "uav0000161_00000_v", "uav0000188_00000_v",
    "uav0000201_00000_v", "uav0000249_00001_v", "uav0000249_02688_v",
    "uav0000297_00000_v", "uav0000297_02761_v", "uav0000306_00230_v",
    "uav0000355_00001_v", "uav0000370_00001_v",
]

# DETRAC-20: selected on DETRAC — zero VisDrone leakage
DETRAC_20 = [
    "fast_keypoints_half", "edge_density_coarse", "gradient_magnitude_mean",
    "gabor_nyquist_energy", "fast_keypoints_full", "glcm_homogeneity",
    "overall_edge_density", "top_edge_density", "top_high_freq_energy",
    "top_gradient_std", "fft_critical_band_ratio", "total_edge_components",
    "top_laplacian_var", "glcm_entropy", "mid_edge_density",
    "dark_channel_mean", "mid_gradient_std", "foreground_blob_count",
    "mid_high_freq_energy", "rms_contrast",
]

JAYS_DROP_COLS = [
    "GLCM_contrast", "GLCM_homogeneity", "GLCM_energy", "GLCM_entropy",
    "GLCM_correlation", "LBP_entropy", "LBP_uniformity", "wavelet_detail_energy",
]

WINDOW = 15   # 0.5s at 30fps
HORIZON = 15  # predict 0.5s ahead
TRAIN_STRIDE = 8
EVAL_STRIDE = 15
SEEDS = [42, 123, 7]


# ── Data Loading ──────────────────────────────────────────────────────────


def load_eval_data():
    """Load YOLO evaluation results and merge nano + medium."""
    ev = pd.read_csv(EVAL_CSV)
    nano = ev[ev["model"] == "yolo11n.pt"][
        ["sequence", "frame_id", "tp", "fp", "fn", "num_gt"]].copy()
    nano.columns = ["sequence", "frame_id", "tp_nano", "fp_nano", "fn_nano", "num_gt"]
    med = ev[ev["model"] == "yolo11m.pt"][["sequence", "frame_id", "tp"]].copy()
    med.columns = ["sequence", "frame_id", "tp_medium"]
    return nano.merge(med, on=["sequence", "frame_id"])


def load_all_data():
    """Load eval results + both feature sets."""
    print("Loading data ...", flush=True)
    eval_merged = load_eval_data()

    # DETRAC-65 features
    detrac = pd.read_csv(DETRAC_FEAT_CSV).fillna(0.0)
    detrac_feat_cols = [c for c in detrac.columns if c not in ("sequence", "frame_id")]

    # Jay's features
    jays = pd.read_csv(JAYS_FEAT_CSV)
    jays = jays.drop(columns=[c for c in JAYS_DROP_COLS if c in jays.columns], errors="ignore")
    jays = jays.fillna(0.0)
    jays_feat_cols = [c for c in jays.columns if c not in ("sequence", "frame_id")]

    # Merge eval + DETRAC features
    df_detrac = eval_merged.merge(detrac, on=["sequence", "frame_id"], how="inner")

    # Merge eval + Jay's features
    df_jays = eval_merged.merge(jays, on=["sequence", "frame_id"], how="inner")

    # Combined: merge both feature sets with eval
    detrac_prefixed = detrac.rename(columns={c: f"D_{c}" for c in detrac_feat_cols})
    jays_prefixed = jays.rename(columns={c: f"J_{c}" for c in jays_feat_cols})
    df_combined = eval_merged.merge(detrac_prefixed, on=["sequence", "frame_id"], how="inner")
    df_combined = df_combined.merge(jays_prefixed, on=["sequence", "frame_id"], how="inner")
    combined_cols = [f"D_{c}" for c in detrac_feat_cols] + [f"J_{c}" for c in jays_feat_cols]

    all_seqs = sorted(eval_merged["sequence"].unique())
    train_seqs = [s for s in all_seqs if s not in VAL_SEQS]
    val_seqs = [s for s in all_seqs if s in VAL_SEQS]

    print(f"  Total: {len(all_seqs)} sequences, {len(eval_merged)} frames", flush=True)
    print(f"  Train: {len(train_seqs)} sequences", flush=True)
    print(f"  Val:   {len(val_seqs)} sequences", flush=True)
    print(f"  DETRAC features: {len(detrac_feat_cols)}", flush=True)
    print(f"  Jay's features:  {len(jays_feat_cols)}", flush=True)
    print(f"  Combined:        {len(combined_cols)}", flush=True)

    return {
        "detrac": (df_detrac, detrac_feat_cols),
        "jays": (df_jays, jays_feat_cols),
        "combined": (df_combined, combined_cols),
    }


# ── Windowing ─────────────────────────────────────────────────────────────


def compute_threshold(df):
    """Global median of future fn_nano across all windows."""
    all_targets = []
    for seq, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        fn = sdf["fn_nano"].values.astype(np.float32)
        n = len(sdf)
        for t in range(0, n - WINDOW - HORIZON + 1, WINDOW):
            all_targets.append(fn[t + WINDOW:t + WINDOW + HORIZON].mean())
    return float(np.median(all_targets))


def build_windows(df, feature_cols, threshold):
    """Build shifted-label windows with train/val split."""
    sequences = sorted(df["sequence"].unique())
    train_seqs = [s for s in sequences if s not in VAL_SEQS]
    val_seqs = [s for s in sequences if s in VAL_SEQS]

    def _make(seqs):
        X_all, y_all, seq_labels = [], [], []
        tp_nano_last, tp_medium_last = [], []
        for seq in seqs:
            sdf = df[df["sequence"] == seq].sort_values("frame_id")
            feats = sdf[feature_cols].values.astype(np.float32)
            fn = sdf["fn_nano"].values.astype(np.float32)
            tp_n = sdf["tp_nano"].values
            tp_m = sdf["tp_medium"].values
            n = len(sdf)
            s = TRAIN_STRIDE if seq not in VAL_SEQS else EVAL_STRIDE
            for i in range(0, n - WINDOW - HORIZON + 1, s):
                x = feats[i:i + WINDOW]
                future_fn = fn[i + WINDOW:i + WINDOW + HORIZON]
                y = 1 if future_fn.mean() >= threshold else 0
                X_all.append(x)
                y_all.append(y)
                seq_labels.append(seq)
                tp_nano_last.append(tp_n[i + WINDOW - 1])
                tp_medium_last.append(tp_m[i + WINDOW - 1])
        return (np.array(X_all), np.array(y_all), seq_labels,
                np.array(tp_nano_last), np.array(tp_medium_last))

    X_tr, y_tr, tr_seqs, _, _ = _make(train_seqs)
    X_va, y_va, va_seqs, tp_n_last, tp_m_last = _make(val_seqs)
    return X_tr, y_tr, X_va, y_va, va_seqs, tp_n_last, tp_m_last


# ── Feature Selection ─────────────────────────────────────────────────────


def spearman_select(df, feature_cols, threshold, min_rho=0.05, cluster_thresh=0.5):
    """Spearman correlation + hierarchical clustering on TRAIN sequences only."""
    train_df = df[~df["sequence"].isin(VAL_SEQS)]
    feats_list, targets_list = [], []
    for seq, sdf in train_df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        fn = sdf["fn_nano"].values.astype(np.float32)
        n = len(sdf)
        for t in range(0, n - WINDOW - HORIZON + 1, WINDOW):
            x = sdf[feature_cols].iloc[t:t + WINDOW].mean().values
            future_fn = fn[t + WINDOW:t + WINDOW + HORIZON].mean()
            feats_list.append(x)
            targets_list.append(future_fn)
    X = np.array(feats_list)
    y_target = np.array(targets_list)

    corrs = {}
    for j, col in enumerate(feature_cols):
        rho, pval = spearmanr(X[:, j], y_target)
        corrs[col] = (rho, pval)

    # Filter by min correlation
    candidates = [(f, rho) for f, (rho, p) in corrs.items() if abs(rho) >= min_rho]
    if len(candidates) < 3:
        min_rho = 0.03
        candidates = [(f, rho) for f, (rho, p) in corrs.items() if abs(rho) >= min_rho]
    candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    cand_names = [f for f, _ in candidates]

    if len(cand_names) <= 3:
        return cand_names, corrs

    # Hierarchical clustering to remove redundant features
    cand_data = train_df[cand_names].values
    corr_matrix = np.abs(np.corrcoef(cand_data.T))
    np.fill_diagonal(corr_matrix, 0)
    dist = np.clip(1 - corr_matrix, 0, None)
    np.fill_diagonal(dist, 0)
    condensed = dist[np.triu_indices(len(cand_names), k=1)]
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=cluster_thresh, criterion="distance")

    rho_map = {f: abs(r) for f, r in candidates}
    selected = []
    for cl in sorted(set(clusters)):
        members = [cand_names[i] for i in range(len(cand_names)) if clusters[i] == cl]
        best = max(members, key=lambda f: rho_map[f])
        selected.append(best)
    selected.sort(key=lambda f: rho_map[f], reverse=True)
    return selected, corrs


def lasso_select(df, feature_cols, threshold):
    """L1-penalized logistic regression on TRAIN windows."""
    train_df = df[~df["sequence"].isin(VAL_SEQS)]
    feats_list, labels_list = [], []
    for seq, sdf in train_df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        fn = sdf["fn_nano"].values.astype(np.float32)
        n = len(sdf)
        for t in range(0, n - WINDOW - HORIZON + 1, WINDOW):
            x = sdf[feature_cols].iloc[t:t + WINDOW].mean().values
            future_fn = fn[t + WINDOW:t + WINDOW + HORIZON].mean()
            feats_list.append(x)
            labels_list.append(1 if future_fn >= threshold else 0)
    X = np.array(feats_list)
    y = np.array(labels_list)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    clf = LogisticRegressionCV(
        penalty="l1", solver="saga", Cs=20, cv=5,
        max_iter=5000, random_state=42, scoring="accuracy",
    )
    clf.fit(X_sc, y)
    coefs = clf.coef_.ravel()
    surviving = [(col, coef) for col, coef in zip(feature_cols, coefs) if abs(coef) > 1e-6]
    surviving.sort(key=lambda x: abs(x[1]), reverse=True)
    selected = [f for f, _ in surviving]
    return selected, dict(zip(feature_cols, coefs))


# ── Model ─────────────────────────────────────────────────────────────────


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class DirichletGRU(nn.Module):
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, 2)
    def forward(self, x):
        _, h = self.gru(x)
        return torch.exp(self.fc(h[-1])) + 1.0


def dirichlet_loss(alpha, y, epoch, max_epoch=60):
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    nll = -torch.log(p.gather(1, y.unsqueeze(1)) + 1e-8).squeeze(1)
    lam = min(1.0, epoch / (max_epoch * 0.6))
    alpha0 = torch.ones_like(alpha)
    kl = (torch.lgamma(alpha.sum(1)) - torch.lgamma(alpha0.sum(1))
          - (torch.lgamma(alpha) - torch.lgamma(alpha0)).sum(1)
          + ((alpha - alpha0) * (torch.digamma(alpha) - torch.digamma(S))).sum(1))
    return (nll + lam * kl).mean()


def train_gru(X_tr, y_tr, X_va, y_va, seed, n_features):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirichletGRU(n_features, hidden=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    ds = SeqDataset(X_tr, y_tr)
    dl = DataLoader(ds, batch_size=128, shuffle=True)
    best_acc, best_state = 0, None
    for epoch in range(60):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = dirichlet_loss(model(xb), yb, epoch, 60)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
            av = model(Xv)
            preds = (av[:, 1] / av.sum(1)).cpu().numpy()
            acc = ((preds >= 0.5).astype(int) == y_va).mean()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
        av = model(Xv)
        preds = (av[:, 1] / av.sum(1)).cpu().numpy()
        pred_labels = (preds >= 0.5).astype(int)
    return best_acc, pred_labels


def train_rf(X_tr, y_tr, X_va, y_va, seed):
    X_tr_flat = X_tr.reshape(len(X_tr), -1)
    X_va_flat = X_va.reshape(len(X_va), -1)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1)
    rf.fit(X_tr_flat, y_tr)
    pred = rf.predict(X_va_flat)
    return (pred == y_va).mean(), pred, rf.feature_importances_


# ── Intra-sequence Analysis ──────────────────────────────────────────────


def intra_sequence_analysis(y_va, pred_labels, va_seqs):
    """Per-sequence breakdown: easy/hard/transition accuracy."""
    seq_names = sorted(set(va_seqs))
    results = {}
    for seq in seq_names:
        mask = np.array([s == seq for s in va_seqs])
        y_s, p_s = y_va[mask], pred_labels[mask]
        n = len(y_s)
        if n == 0:
            continue
        easy_mask, hard_mask = y_s == 0, y_s == 1
        easy_acc = (p_s[easy_mask] == 0).mean() if easy_mask.sum() > 0 else None
        hard_acc = (p_s[hard_mask] == 1).mean() if hard_mask.sum() > 0 else None
        transitions, trans_correct = 0, 0
        for j in range(1, n):
            if y_s[j] != y_s[j - 1]:
                transitions += 1
                if p_s[j] == y_s[j]:
                    trans_correct += 1
        trans_acc = trans_correct / transitions if transitions > 0 else None
        results[seq] = {
            "n": n, "acc": (y_s == p_s).mean(), "hard_frac": y_s.mean(),
            "easy_acc": easy_acc, "hard_acc": hard_acc,
            "trans_acc": trans_acc, "transitions": transitions,
            "pred_hard_frac": p_s.mean(),
        }
    return results


def compute_intra_summary(intra):
    """Compute summary statistics from intra-sequence analysis."""
    mixed_signal, mixed_total = 0, 0
    for seq, r in intra.items():
        if 0.05 < r["hard_frac"] < 0.95:
            mixed_total += 1
            if r["easy_acc"] is not None and r["hard_acc"] is not None:
                if r["easy_acc"] > 0.5 and r["hard_acc"] > 0.5:
                    mixed_signal += 1
    total_trans = sum(r["transitions"] for r in intra.values() if r["transitions"])
    total_trans_correct = sum(
        int(r["trans_acc"] * r["transitions"])
        for r in intra.values()
        if r["trans_acc"] is not None and r["transitions"] > 0)
    agg_trans_acc = total_trans_correct / total_trans if total_trans > 0 else 0
    return mixed_signal, mixed_total, agg_trans_acc, total_trans


# ── Run one feature set ──────────────────────────────────────────────────


def evaluate_feature_set(fs_name, fs_cols, df, threshold):
    """Train GRU + RF on a feature set, return results."""
    print(f"\n{'='*60}", flush=True)
    print(f"Feature set: {fs_name} ({len(fs_cols)} features)", flush=True)
    print(f"{'='*60}", flush=True)

    scaler = StandardScaler()
    train_data = df[~df["sequence"].isin(VAL_SEQS)]
    scaler.fit(train_data[fs_cols].values)
    df_copy = df.copy()
    df_copy[fs_cols] = scaler.transform(df_copy[fs_cols].values)

    X_tr, y_tr, X_va, y_va, va_seqs, tp_n, tp_m = build_windows(df_copy, fs_cols, threshold)
    print(f"  Windows: train={len(X_tr)} (hard={y_tr.mean():.1%})  "
          f"val={len(X_va)} (hard={y_va.mean():.1%})", flush=True)

    # GRU 3-seed
    gru_accs, best_gru_acc, best_gru_preds = [], 0, None
    for seed in SEEDS:
        acc, preds = train_gru(X_tr, y_tr, X_va, y_va, seed, len(fs_cols))
        gru_accs.append(acc)
        print(f"  GRU seed={seed}: {acc:.1%}", flush=True)
        if acc > best_gru_acc:
            best_gru_acc = acc
            best_gru_preds = preds

    # RF
    rf_acc, rf_preds, rf_imp = train_rf(X_tr, y_tr, X_va, y_va, 42)
    print(f"  RF: {rf_acc:.1%}", flush=True)

    # MedSample baseline
    med_pred = (tp_m > tp_n).astype(int)
    med_acc = (med_pred == y_va).mean()

    # Majority baseline
    majority_label = int(y_tr.mean() >= 0.5)
    majority_acc = (y_va == majority_label).mean()

    # Intra-sequence analysis
    intra = intra_sequence_analysis(y_va, best_gru_preds, va_seqs)
    mixed_signal, mixed_total, trans_acc, total_trans = compute_intra_summary(intra)

    print(f"  MedSample: {med_acc:.1%}  |  Majority: {majority_acc:.1%}", flush=True)
    print(f"  Intra signal: {mixed_signal}/{mixed_total} mixed seqs", flush=True)
    print(f"  Transition accuracy: {trans_acc:.1%} ({total_trans} transitions)", flush=True)

    # Per-sequence detail
    for seq in sorted(intra.keys()):
        r = intra[seq]
        ea = f"{r['easy_acc']:.0%}" if r['easy_acc'] is not None else "N/A"
        ha = f"{r['hard_acc']:.0%}" if r['hard_acc'] is not None else "N/A"
        ta = f"{r['trans_acc']:.0%}" if r['trans_acc'] is not None else "N/A"
        print(f"    {seq}: acc={r['acc']:.0%} hard={r['hard_frac']:.0%} "
              f"easy={ea} hard={ha} trans={ta} ({r['transitions']}t)", flush=True)

    return {
        "gru_accs": gru_accs, "gru_mean": np.mean(gru_accs),
        "gru_std": np.std(gru_accs), "rf_acc": rf_acc,
        "med_acc": med_acc, "majority_acc": majority_acc,
        "rf_importances": dict(zip(fs_cols, rf_imp)) if len(rf_imp) == len(fs_cols) else {},
        "intra": intra, "mixed_signal": mixed_signal,
        "mixed_total": mixed_total, "trans_acc": trans_acc,
        "total_trans": total_trans,
        "n_train": len(X_tr), "n_val": len(X_va),
        "train_hard_frac": y_tr.mean(), "val_hard_frac": y_va.mean(),
    }


# ── Intra-sequence feature variance ─────────────────────────────────────


def intra_seq_feature_variance(df, feature_cols):
    """Check within vs between sequence feature variance."""
    seq_means = df.groupby("sequence")[feature_cols].mean()
    between_var = seq_means.var()
    total_var = df[feature_cols].var()
    within_var = total_var - between_var
    ratio = within_var / (total_var + 1e-10)
    return ratio


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    data = load_all_data()
    df_detrac, detrac_feat_cols = data["detrac"]
    df_jays, jays_feat_cols = data["jays"]
    df_combined, combined_cols = data["combined"]

    # Use DETRAC df for threshold (largest overlap)
    threshold = compute_threshold(df_detrac)
    print(f"\nThreshold (median future fn_nano): {threshold:.3f}", flush=True)

    # Intra-sequence feature variance (Jay's features on Jay's df)
    var_ratio = intra_seq_feature_variance(df_jays, jays_feat_cols)

    # ── Experiment 1: DETRAC-20 Blind ────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 1: DETRAC-20 Blind Cross-Domain Test", flush=True)
    print("  Features selected on DETRAC, applied to VisDrone — zero leakage", flush=True)
    print("=" * 70, flush=True)

    r_blind = evaluate_feature_set("DETRAC-20 (blind)", DETRAC_20, df_detrac, threshold)

    # Also test all 65 DETRAC features
    r_all65 = evaluate_feature_set(f"All-{len(detrac_feat_cols)} DETRAC",
                                    detrac_feat_cols, df_detrac, threshold)

    # ── Experiment 2: Jay's Features + Selection ─────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 2: Jay's Features with VisDrone-Based Selection", flush=True)
    print("=" * 70, flush=True)

    jays_selected, jays_corrs = spearman_select(df_jays, jays_feat_cols, threshold)
    print(f"\n  Spearman+Clustering selected: {len(jays_selected)} features", flush=True)
    for f in jays_selected[:10]:
        print(f"    {f:35s}  ρ={jays_corrs[f][0]:+.4f}", flush=True)

    r_jays_sel = evaluate_feature_set(
        f"Jay's-Selected ({len(jays_selected)})", jays_selected, df_jays, threshold)
    r_jays_all = evaluate_feature_set(
        f"All-{len(jays_feat_cols)} Jay's", jays_feat_cols, df_jays, threshold)

    # ── Experiment 3: Combined Features + Selection ──────────────────────
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 3: Combined Feature Selection (DETRAC-65 + Jay's-48)", flush=True)
    print("=" * 70, flush=True)

    comb_spearman, comb_corrs = spearman_select(df_combined, combined_cols, threshold)
    print(f"\n  Spearman+Clustering: {len(comb_spearman)} features", flush=True)
    n_d = sum(1 for f in comb_spearman if f.startswith("D_"))
    n_j = sum(1 for f in comb_spearman if f.startswith("J_"))
    print(f"  Sources: {n_d} DETRAC + {n_j} Jay's", flush=True)

    comb_lasso, lasso_coefs = lasso_select(df_combined, combined_cols, threshold)
    print(f"  Lasso: {len(comb_lasso)} features", flush=True)

    r_comb_spear = evaluate_feature_set(
        f"Combined-Spearman ({len(comb_spearman)})", comb_spearman, df_combined, threshold)
    r_comb_lasso = evaluate_feature_set(
        f"Combined-Lasso ({len(comb_lasso)})", comb_lasso, df_combined, threshold) if comb_lasso else None
    r_comb_all = evaluate_feature_set(
        f"All-{len(combined_cols)} Combined", combined_cols, df_combined, threshold)

    # ── Collect all results ──────────────────────────────────────────────

    all_results = {
        "DETRAC-20 (blind)": r_blind,
        f"All-{len(detrac_feat_cols)} DETRAC": r_all65,
        f"Jay's-Selected ({len(jays_selected)})": r_jays_sel,
        f"All-{len(jays_feat_cols)} Jay's": r_jays_all,
        f"Combined-Spearman ({len(comb_spearman)})": r_comb_spear,
    }
    if r_comb_lasso is not None:
        all_results[f"Combined-Lasso ({len(comb_lasso)})"] = r_comb_lasso
    all_results[f"All-{len(combined_cols)} Combined"] = r_comb_all

    # ── CSV Output ───────────────────────────────────────────────────────

    csv_rows = []
    for fs_name, r in all_results.items():
        for seq in sorted(r["intra"].keys()):
            ir = r["intra"][seq]
            csv_rows.append({
                "feature_set": fs_name, "sequence": seq,
                "n_windows": ir["n"], "accuracy": ir["acc"],
                "hard_frac": ir["hard_frac"], "easy_acc": ir["easy_acc"],
                "hard_acc": ir["hard_acc"], "trans_acc": ir["trans_acc"],
                "transitions": ir["transitions"], "pred_hard_frac": ir["pred_hard_frac"],
            })
    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved CSV: {OUT_CSV}", flush=True)

    # ── PDF Report ───────────────────────────────────────────────────────

    print("Generating PDF ...", flush=True)
    with PdfPages(str(OUT_PDF)) as pdf:

        # Page 1: Master accuracy comparison
        fig, ax = plt.subplots(figsize=(14, 7))
        fs_names = list(all_results.keys())
        gru_means = [all_results[n]["gru_mean"] for n in fs_names]
        gru_stds = [all_results[n]["gru_std"] for n in fs_names]
        rf_accs = [all_results[n]["rf_acc"] for n in fs_names]
        med_accs = [all_results[n]["med_acc"] for n in fs_names]
        x = np.arange(len(fs_names))
        width = 0.25
        bars1 = ax.bar(x - width, gru_means, width, yerr=gru_stds,
                        label="GRU-1L-32", capsize=3, color="#4C72B0")
        bars2 = ax.bar(x, rf_accs, width, label="RF", color="#DD8452")
        bars3 = ax.bar(x + width, med_accs, width, label="MedSample", color="#C44E52", alpha=0.7)
        # Majority baseline (same across all)
        maj = all_results[fs_names[0]]["majority_acc"]
        ax.axhline(maj, color="gray", ls=":", lw=1.5, label=f"Majority ({maj:.1%})")
        ax.set_xticks(x)
        ax.set_xticklabels(fs_names, fontsize=7, rotation=15, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"VisDrone Full Evaluation — 50 train / 23 val sequences\n"
                     f"Binary difficulty prediction (0.5s window, threshold={threshold:.2f})")
        ax.set_ylim(0.3, 1.0)
        ax.legend(fontsize=8)
        for bars in [bars1, bars2, bars3]:
            for b in bars:
                ax.annotate(f"{b.get_height():.1%}",
                            (b.get_x() + b.get_width()/2, b.get_height()),
                            ha="center", va="bottom", fontsize=6)
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 2: Combined Spearman top 30
        ranked = sorted(comb_corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)
        top30 = ranked[:30]
        fig, ax = plt.subplots(figsize=(12, 10))
        names30 = [f[0].replace("D_", "").replace("J_", "") for f in top30]
        rhos30 = [v[0] for _, v in top30]
        src30 = ["DETRAC" if f[0].startswith("D_") else "Jay's" for f in top30]
        colors30 = ["#4C72B0" if s == "DETRAC" else "#55A868" for s in src30]
        ax.barh(range(len(top30)), rhos30, color=colors30)
        ax.set_yticks(range(len(top30)))
        ax.set_yticklabels([f"{n}  [{s[0]}]" for n, s in zip(names30, src30)], fontsize=7)
        ax.set_xlabel("Spearman ρ (train sequences)")
        ax.set_title("Combined Feature Ranking: Spearman ρ vs Future fn_nano\n"
                     "(Blue=DETRAC-65, Green=Jay's-48)")
        ax.axvline(0, color="black", lw=0.5)
        ax.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 3: Within-sequence variance ratio
        fig, ax = plt.subplots(figsize=(10, 8))
        sorted_ratio = var_ratio.sort_values(ascending=False)
        top_n = min(25, len(sorted_ratio))
        feats_var = sorted_ratio.index[:top_n]
        vals_var = sorted_ratio.values[:top_n]
        ax.barh(range(top_n), vals_var, color="#55A868")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feats_var, fontsize=7)
        ax.set_xlabel("Within-Sequence / Total Variance Ratio")
        ax.set_title("VisDrone: Features with Most Within-Sequence Variation\n"
                     "(Higher = more intra-sequence signal, useful for frame-level prediction)")
        ax.axvline(0.5, color="red", ls="--", lw=0.8, label="50% ratio")
        ax.legend()
        ax.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 4: Per-sequence accuracy heatmap for best method
        best_fs = max(all_results.keys(), key=lambda k: all_results[k]["gru_mean"])
        intra = all_results[best_fs]["intra"]
        seq_names = sorted(intra.keys())

        if seq_names:
            fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(seq_names) * 0.35)))

            easy_accs = [intra[s]["easy_acc"] if intra[s]["easy_acc"] is not None else 0
                         for s in seq_names]
            hard_accs = [intra[s]["hard_acc"] if intra[s]["hard_acc"] is not None else 0
                         for s in seq_names]
            hard_fracs = [intra[s]["hard_frac"] for s in seq_names]

            ax = axes[0]
            y_pos = np.arange(len(seq_names))
            ax.barh(y_pos - 0.2, easy_accs, 0.4, label="Easy acc", color="#55A868")
            ax.barh(y_pos + 0.2, hard_accs, 0.4, label="Hard acc", color="#C44E52")
            ax.axvline(0.5, color="black", ls="--", lw=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{s[:20]}  ({h:.0%} hard)"
                                for s, h in zip(seq_names, hard_fracs)], fontsize=5)
            ax.set_xlabel("Accuracy")
            ax.set_title(f"Per-Sequence Easy/Hard Accuracy\n({best_fs})")
            ax.legend(fontsize=8)

            ax = axes[1]
            trans_accs = [intra[s]["trans_acc"] if intra[s]["trans_acc"] is not None else 0
                          for s in seq_names]
            n_trans = [intra[s]["transitions"] for s in seq_names]
            colors_t = ["#4C72B0" if t > 0 else "#CCCCCC" for t in n_trans]
            ax.barh(y_pos, trans_accs, color=colors_t)
            ax.axvline(0.5, color="red", ls="--", lw=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{s[:20]} ({n})" for s, n in zip(seq_names, n_trans)],
                               fontsize=5)
            ax.set_xlabel("Transition Accuracy")
            ax.set_title("Transition Accuracy (n transitions)")

            plt.tight_layout()
            pdf.savefig(fig); plt.close()

        # Page 5: Comparison table (text page)
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")
        headers = ["Feature Set", "GRU (3-seed)", "RF", "MedSample", "Maj.",
                    "Intra Signal", "Trans Acc", "n_val"]
        rows = []
        for fs_name, r in all_results.items():
            rows.append([
                fs_name,
                f"{r['gru_mean']:.1%} ± {r['gru_std']:.1%}",
                f"{r['rf_acc']:.1%}",
                f"{r['med_acc']:.1%}",
                f"{r['majority_acc']:.1%}",
                f"{r['mixed_signal']}/{r['mixed_total']}",
                f"{r['trans_acc']:.1%} ({r['total_trans']}t)",
                f"{r['n_val']}",
            ])
        table = ax.table(cellText=rows, colLabels=headers, loc="center",
                         cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.5)
        # Color header
        for j in range(len(headers)):
            table[0, j].set_facecolor("#4C72B0")
            table[0, j].set_text_props(color="white", fontweight="bold")
        # Highlight best GRU
        best_idx = max(range(len(rows)),
                       key=lambda i: float(rows[i][1].split("±")[0].strip().replace("%", "")) / 100)
        for j in range(len(headers)):
            table[best_idx + 1, j].set_facecolor("#E6F3FF")

        ax.set_title("Full Evaluation Summary — VisDrone (50 train / 23 val)",
                     fontsize=12, fontweight="bold", pad=20)
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

    print(f"\nSaved PDF: {OUT_PDF}", flush=True)

    # ── Final Summary ────────────────────────────────────────────────────

    print(f"\n{'='*70}", flush=True)
    print("FINAL SUMMARY — VisDrone Full Evaluation (50 train / 23 val)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Threshold: {threshold:.3f}", flush=True)

    for fs_name, r in all_results.items():
        print(f"\n{fs_name}:", flush=True)
        print(f"  GRU: {r['gru_mean']:.1%} ± {r['gru_std']:.1%}  "
              f"(seeds: {[f'{a:.1%}' for a in r['gru_accs']]})", flush=True)
        print(f"  RF:  {r['rf_acc']:.1%}  |  MedSample: {r['med_acc']:.1%}  |  "
              f"Majority: {r['majority_acc']:.1%}", flush=True)
        print(f"  Intra: {r['mixed_signal']}/{r['mixed_total']} mixed seqs  |  "
              f"Trans: {r['trans_acc']:.1%} ({r['total_trans']}t)", flush=True)

    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
